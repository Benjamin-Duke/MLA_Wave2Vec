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
        
    def compute_loss(self, c, q, mask_indices, num_negatives=100, temperature=0.1, eps=1e-7):
        """
        Compute contrastive loss:
        L_m = -log(exp(sim(c_t, q_t)/κ) / sum_k(exp(sim(c_t, q_k)/κ)))
        where:
        - c_t is the context network output at masked position t
        - q_t is the correct quantized representation at position t
        - q_k are the distractors (including q_t)
        - κ is the temperature (set to 0.1)
        - sim(a,b) is the cosine similarity between a and b
        """
        # Check if we have any masked indices
        if mask_indices.sum() == 0:
            return torch.tensor(0.0, device=c.device, requires_grad=True)

        # Get masked indices in flattened form
        flat_mask = mask_indices.view(-1)
        masked_indices = torch.nonzero(flat_mask).squeeze(-1)

        if len(masked_indices) == 0:  # No masked positions
            return torch.tensor(0.0, device=c.device, requires_grad=True)

        # Get positive samples (c_t and q_t pairs)
        c_masked = c.view(-1, c.size(-1))[masked_indices]  # c_t
        q_masked = q.view(-1, q.size(-1))[masked_indices]  # q_t

        # Sample negative indices for each positive
        with torch.no_grad():
            neg_indices = self._sample_negatives(masked_indices, len(flat_mask), num_negatives)
            negatives = q.view(-1, q.size(-1))[neg_indices]  # q_k distractors

        # Compute cosine similarity with numerical stability
        c_masked = F.normalize(c_masked + eps, dim=-1)
        q_masked = F.normalize(q_masked + eps, dim=-1)
        negatives = F.normalize(negatives + eps, dim=-1)

        # Compute sim(c_t, q_t) for positives
        pos_logits = torch.sum(c_masked * q_masked, dim=-1, keepdim=True)  # [num_masked, 1]

        # Compute sim(c_t, q_k) for negatives
        neg_logits = torch.bmm(c_masked.unsqueeze(1), negatives.transpose(1, 2)).squeeze(1)  # [num_masked, num_negatives]

        # Concatenate positive and negative logits
        logits = torch.cat([pos_logits, neg_logits], dim=1)  # [num_masked, 1 + num_negatives]

        # Scale by temperature κ
        logits = logits / temperature

        # Targets are zeros (positive pair should be selected)
        targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        # Compute contrastive loss using cross entropy (equivalent to -log(exp(pos)/sum(exp(all))))
        contrastive_loss = F.cross_entropy(logits, targets)

        # Compute diversity loss (weight α = 0.1 as per paper)
        try:
            diversity_loss = self.calculate_diversity_loss(q)*0.1
        except Exception as e:
            print(f"Warning: Error computing diversity loss: {e}")
            diversity_loss = torch.tensor(0.0, device=c.device, requires_grad=True)

        # Total loss
        loss = contrastive_loss + diversity_loss

        # # Print loss components only if they're valid
        if self.training and not torch.isnan(loss) and not torch.isinf(loss):
        #     print(f"\nLoss components:")
             # print(f"Contrastive loss: {contrastive_loss.item():.4f}")
             # print(f"Diversity loss: {diversity_loss.item():.4f}")
        #     print(f"Total loss: {loss.item():.4f}")
        #     print(f"Prob perplexity: {prob_perplexity.item():.2f}")
        #     print(f"Number of masked positions: {len(masked_indices)}")
        #     print(f"Average positive logit: {pos_logits.mean().item():.4f}")
        #     print(f"Average negative logit: {neg_logits.mean().item():.4f}")

        return loss, contrastive_loss, diversity_loss

    def calculate_diversity_loss(self, qt):
        B, G, V = qt.shape
        softmax_probs = F.softmax(qt, dim=-1)
        entropy = -torch.sum(softmax_probs * torch.log(softmax_probs + 1e-10), dim=-1)
        return entropy.mean()

    
    def _sample_negatives(self, pos_indices, num_masked, num_negatives):
        """Sample negative indices from other masked positions."""
        with torch.no_grad():
            # Create a range of all masked indices
            all_indices = torch.arange(num_masked, device=pos_indices.device)

            # For each positive, sample K distractors from other masked positions
            neg_indices = []
            for i in range(len(pos_indices)):
                # Exclude the current positive index
                valid_indices = torch.cat([all_indices[:i], all_indices[i+1:]])
                # Sample K indices
                sampled = valid_indices[torch.randperm(len(valid_indices))[:num_negatives]]
                neg_indices.append(sampled)

            return torch.stack(neg_indices)