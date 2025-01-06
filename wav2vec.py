import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from Modules.config import Wav2Vec2Config  
from Modules import Features_encoder
from Modules import quantizationModule
from Modules import mask
from Modules import transformerEncoder

class Wav2Vec2(nn.Module):
    def __init__(self, config: Wav2Vec2Config):
        # Call parent class initialization first
        super().__init__()
        
        self.config = config
        
        # Feature encoder with layer norm and dropout
        self.feature_encoder = Features_encoder.FeatureEncoder(config.conv_layers)
        
        # Calculate the encoder output dimension from the last conv layer
        last_conv_channels = config.conv_layers[-1][0]
        
        # Projection layers
        self.proj = nn.Linear(last_conv_channels, config.d_model)
        self.quantizer_proj = nn.Linear(config.num_codebooks * config.codebook_size, config.d_model)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.d_model)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Context network components
        # 1. Convolutional layer for relative positional embedding
        kernel_size = 128
        padding = kernel_size
        self.context_pos_conv = nn.Sequential(
            nn.Conv1d(
                config.d_model,
                config.d_model,
                kernel_size=kernel_size,
                padding=padding,
                groups=16,
                padding_mode='replicate'
            ),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # 2. Transformer encoder with layer drop
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = transformerEncoder.TransformerEncoder(
            encoder_layer, 
            config.num_encoder_layers,
        )
        
        # Quantizer
        self.quantizer = quantizationModule.ProductQuantizer(
            config.d_model,
            num_codebooks=config.num_codebooks,
            codebook_size=config.codebook_size,
            temp=config.temp,
            min_temp=config.min_temp,
            temp_decay=config.temp_decay
        )

        # Masking module
        self.mask = mask.Mask(config.mask_prob, config.mask_length)
        
        # Learnable mask embedding
        self.mask_emb = nn.Parameter(torch.FloatTensor(config.d_model).uniform_())
        
    def forward(self, x, maskBool=True):
        # Feature encoder
        x = self.feature_encoder(x)
        
        # Project to transformer dimension
        x = self.proj(x)
        x = self.dropout(x)

        # Get quantized representations
        q = self.quantizer(x)
        q = self.quantizer_proj(q)
        
        # Initialize mask_indices
        mask_indices = None
        
        # Apply masking if requested
        if maskBool:
            mask_indices = self.mask(x)
            x = torch.where(
                mask_indices.unsqueeze(-1),
                self.mask_emb.view(1, 1, -1).expand(x.shape[0], -1, -1),
                x
            )
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # Add relative positional embeddings
        x_t = x.transpose(1, 2)
        orig_len = x_t.size(2)
        
        pos_embedding = self.context_pos_conv(x_t)
        
        # Ensure output length matches input
        if pos_embedding.size(2) > orig_len:
            excess = pos_embedding.size(2) - orig_len
            start = excess // 2
            pos_embedding = pos_embedding[:, :, start:start + orig_len]
        elif pos_embedding.size(2) < orig_len:
            pad_size = orig_len - pos_embedding.size(2)
            pad_left = pad_size // 2
            pad_right = pad_size - pad_left
            pos_embedding = F.pad(pos_embedding, (pad_left, pad_right), mode='replicate')
            
        pos_embedding = pos_embedding.transpose(1, 2)
        
        assert x.shape == pos_embedding.shape, f"Shape mismatch: x={x.shape}, pos_embedding={pos_embedding.shape}"
        x = x + pos_embedding
        
        # Transformer processing
        c = self.transformer(x)
        
        if not maskBool:
            mask_indices = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)
        
        return c, q, mask_indices
        
    def compute_loss(self, c, q, mask_indices, K=100, temperature=0.1, eps=1e-7):

        device = c.device

        if mask_indices.sum() == 0:
            return torch.tensor(0.0, device=c.device, requires_grad=True)


        c = F.normalize(c, dim=-1).to(device)
        q = F.normalize(q, dim=-1).to(device)

        diversity_loss = self.calculate_diversity_loss(q)
        
        flat_mask = mask_indices.view(-1)
        masked_indices = torch.nonzero(flat_mask).squeeze(-1).to(device)


        
        if len(masked_indices) == 0: 
            return torch.tensor(0.0, device=c.device, requires_grad=True)

        
        batch_indices, seq_indices = torch.where(mask_indices)
        ct = c[batch_indices, seq_indices]  # [num_masked_positions, c_dim]
        qt = q[batch_indices, seq_indices]  
    
        contrastive_loss = 0.0
        num_masked_positions = len(batch_indices)
    
        for i_mask in range(num_masked_positions):
            i_batch = batch_indices[i_mask]
            i_seq = seq_indices[i_mask]
            
            all_indices = torch.arange(0, c.size(1), device=device)
            valid_indices = all_indices[all_indices != i_seq]
            negative_indices = valid_indices[torch.randint(0, len(valid_indices), (K,), device=device)]

            distractors = q[i_batch, negative_indices].to(device)

            sim_correct = F.cosine_similarity(ct[i_mask].unsqueeze(0), qt[i_mask].unsqueeze(0))
            sim_distractors = F.cosine_similarity(ct[i_mask].unsqueeze(0), distractors)

            numerator = torch.exp(sim_correct / temperature).to(device)
            denominator = torch.sum(torch.exp(sim_distractors / temperature)).to(device) + numerator
    
            p_correct = numerator / denominator
    
            contrastive_loss += -torch.log(p_correct + eps)

        # Normaliiize la lossssss
        num_masked_positions = mask_indices.sum().item()  
        contrastive_loss /= num_masked_positions

        loss = contrastive_loss + 0.1 * diversity_loss
        return loss, contrastive_loss, diversity_loss

    def calculate_diversity_loss(self, qt):
        B, G, V = qt.shape
        softmax_probs = F.softmax(qt, dim=-1)
        entropy = -torch.sum(softmax_probs * torch.log(softmax_probs + 1e-10), dim=-1)
        return entropy.mean()

    
    def _sample_negatives(self, pos_indices, num_masked, num_negatives):
        with torch.no_grad():
            all_indices = torch.arange(num_masked, device=pos_indices.device) # Tous les indices possibles

            # For each positive, sample K distractors from other masked positions
            neg_indices = []
            for i in range(len(pos_indices)):
                # Exclude the current positive index
                valid_indices = torch.cat([all_indices[:i], all_indices[i+1:]])
                # Sample K indices
                sampled = valid_indices[torch.randperm(len(valid_indices))[:num_negatives]]
                neg_indices.append(sampled)

            return torch.stack(neg_indices)