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
        
    def compute_loss(self, c, q, mask_indices):
        """Compute contrastive loss with configurations from config"""
        if mask_indices.sum() == 0:
            return torch.tensor(0.0, device=c.device, requires_grad=True)
        
        flat_mask = mask_indices.view(-1)
        masked_indices = torch.nonzero(flat_mask).squeeze(-1)
        
        if len(masked_indices) == 0:
            return torch.tensor(0.0, device=c.device, requires_grad=True)
        
        # Get positive samples
        c_masked = c.view(-1, c.size(-1))[masked_indices]
        q_masked = q.view(-1, q.size(-1))[masked_indices]
        
        # Sample negative indices
        with torch.no_grad():
            neg_indices = self._sample_negatives(
                masked_indices, 
                len(flat_mask), 
                self.config.num_negatives
            )
            negatives = q.view(-1, q.size(-1))[neg_indices]
        
        # Compute cosine similarity
        eps = 1e-7
        c_masked = F.normalize(c_masked + eps, dim=-1)
        q_masked = F.normalize(q_masked + eps, dim=-1)
        negatives = F.normalize(negatives + eps, dim=-1)
        
        # Compute logits
        pos_logits = torch.sum(c_masked * q_masked, dim=-1, keepdim=True)
        neg_logits = torch.bmm(
            c_masked.unsqueeze(1), 
            negatives.transpose(1, 2)
        ).squeeze(1)
        
        logits = torch.cat([pos_logits, neg_logits], dim=1)
        logits = logits / self.config.contrastive_temperature
        
        targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        
        # Compute losses
        contrastive_loss = F.cross_entropy(logits, targets)
        
        try:
            prob_perplexity = self.compute_prob_perplexity()
            diversity_loss = -torch.log(prob_perplexity + eps) * self.config.diversity_weight
            diversity_loss = torch.clamp(diversity_loss, min=-10, max=10)
        except Exception as e:
            print(f"Warning: Error computing diversity loss: {e}")
            diversity_loss = torch.tensor(0.0, device=c.device, requires_grad=True)
        
        return contrastive_loss + diversity_loss

    def compute_prob_perplexity(self, eps=1e-7):
        """Compute codebook perplexity"""
        logits = self.quantizer.weight_proj.weight
        
        logits = logits.view(
            self.config.num_codebooks,
            self.config.codebook_size,
            -1
        )
        
        logits = torch.clamp(logits, min=-100, max=100)
        probs = F.softmax(logits, dim=1)
        
        avg_probs = probs.mean(dim=-1)
        avg_probs = avg_probs + eps
        
        perplexities = []
        for g in range(self.config.num_codebooks):
            p = avg_probs[g]
            p = p / p.sum()
            perplexity = torch.exp(-torch.sum(p * torch.log(p)))
            perplexities.append(perplexity)
        
        return torch.stack(perplexities).mean()
        
    def _sample_negatives(self, pos_indices, num_masked, num_negatives):
        """Sample negative indices from other masked positions"""
        with torch.no_grad():
            all_indices = torch.arange(num_masked, device=pos_indices.device)
            
            neg_indices = []
            for i in range(len(pos_indices)):
                valid_indices = torch.cat([all_indices[:i], all_indices[i+1:]])
                sampled = valid_indices[torch.randperm(len(valid_indices))[:num_negatives]]
                neg_indices.append(sampled)
            
            return torch.stack(neg_indices)

# if __name__ == "__main__":
#     # Initialize model
#     try:
#         config = Wav2Vec2Config()
#         model = Wav2Vec2(config)
#         print("Model initialized successfully.")
#     except Exception as e:
#         print("Error during model initialization:", e)
#         exit(1)

#     # Generate dummy input
#     batch_size = 32
#     seq_length = 48000  # Input sequence length
#     input_tensor = torch.randn(batch_size, seq_length)

#     # Test forward pass
#     try:
#         c, q, mask_indices = model(input_tensor)
#         print("Forward pass works!")
#         print(f"c.shape: {c.shape}, q.shape: {q.shape}, mask_indices.shape: {mask_indices.shape}")
#     except Exception as e:
#         print("Error during forward pass:", e)