"""04/12/2024 Versions
   Author: Edouard
   Changes: "Corrected the handling of the masked time steps by the transformer, 
   and added a linear layer to ensure that the contextualized output has the desired shape
   to correspond to the shape of the quantization module"
"""

import torch
import torch.nn as nn


import numpy as np



import math
import torch.nn.functional as F


class MaskingWithLearnableEmbedding(nn.Module):
    def __init__(self):
        super(MaskingWithLearnableEmbedding, self).__init__()

    def forward(self, x, mask_prob, mask_length):
        B, S, D = x.shape  # B: batch_size, S: sequence_length, D: embedding_dim
        
        # Créer masque de base
        mask = torch.ones((B, S), device=x.device)
        
        # Liste pour stocker les indices des frames masquées
        masked_indices = []
    
        # Nombre de segments à masquer
        num_masks = int(S * mask_prob / mask_length)
        
        for b in range(B):
            # Positions de début des masques
            starts = torch.randperm(S - mask_length)[:num_masks]
            
            # Appliquer les masques et enregistrer les indices des frames masquées
            for start in starts:
                mask[b, start:start+mask_length] = 0
                masked_indices.append((b, start, start + mask_length))  # Enregistrer les indices du segment masqué
        
        # Le masque résultant est de forme [batch_size, seq_len, 1]
        mask = mask.unsqueeze(-1).expand_as(x)  # Extension du masque à la forme [batch_size, seq_len, embedding_dim]

        # Appliquer simplement le masque sans l'embedding appris
        x_masked = x * mask  # Multiplier chaque frame masquée par 0
        
        # Retourner les données masquées ainsi que les indices des frames masquées
        return x_masked, masked_indices

"""

class MaskingWithLearnableEmbedding(nn.Module):
    def __init__(self, embed_size):
        super(MaskingWithLearnableEmbedding, self).__init__()
        self.mask_embedding = nn.Parameter(torch.randn(embed_size))  # Embedding appris pour le masque

    def forward(self, x, mask_prob, mask_length):
        B, S, D = x.shape  # B: batch_size, S: sequence_length, D: embedding_dim
        
        # Créer masque de base
        mask = torch.ones((B, S), device=x.device)
        
        # Liste pour stocker les indices des frames masquées
        masked_indices = []
    
        # Nombre de segments à masquer
        num_masks = int(S * mask_prob / mask_length)
        
        for b in range(B):
            # Positions de début des masques
            starts = torch.randperm(S - mask_length)[:num_masks]
            
            # Appliquer les masques et enregistrer les indices des frames masquées
            for start in starts:
                mask[b, start:start+mask_length] = 0
                masked_indices.append((b, start, start + mask_length))  # Enregistrer les indices du segment masqué
        
        # Le masque résultant est de forme [batch_size, seq_len, 1]
        mask = mask.unsqueeze(-1).expand_as(x)  # Extension du masque à la forme [batch_size, seq_len, embedding_dim]

        # Application de l'embedding appris pour les masques
        mask_embedded = mask * self.mask_embedding  # Multiplier chaque frame masquée par l'embedding appris
        
        # Retourner le masque ainsi que les indices des frames masquées
        return mask_embedded, masked_indices

"""
"""

    def forward(self, latent_reps, mask_prob, mask_length):
        batch_size, seq_length, embed_size = latent_reps.shape
        mask = torch.ones(batch_size, seq_length, embed_size, device=latent_reps.device)  # Le masque est de même taille que latent_reps
        mask_indices = []

        for b in range(batch_size):
            # Choisir aléatoirement les indices à masquer
            indices = np.random.choice(seq_length, size=int(mask_prob * seq_length), replace=False)
            mask_indices.append(indices)
            
            # Appliquer le masque sur les indices choisis
            for idx in indices:
                start = idx
                end = min(start + mask_length, seq_length)
                mask[b, start:end] = 0  # Masquer les valeurs (mettre à zéro)
        
        # Créer une copie de `latent_reps` et remplacer les éléments masqués par `mask_embedding`
        masked_reps = latent_reps.clone()
        for b, indices in enumerate(mask_indices):
            for idx in indices:
                start = idx
                end = min(start + mask_length, seq_length)
                masked_reps[b, start:end] = self.mask_embedding  # Remplacer les valeurs masquées par l'embedding du masque
        
        return masked_reps, mask
"""


class ConvolutionalPositionalEmbedding(nn.Module):
    
    def __init__(self, embed_size, kernel_size, groups, max_relative_position):
        super(ConvolutionalPositionalEmbedding, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=embed_size,
            out_channels=embed_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups
        )
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(embed_size)
        self.max_relative_position = max_relative_position

        self.relative_positions = nn.Embedding(2 * max_relative_position + 1, embed_size)

    def forward(self, x):
        # x shape: (batch_size, seq_length, embed_size)
        batch_size, seq_length, embed_size = x.shape

        x = x.permute(0, 2, 1)  # Switch to (batch_size, embed_size, seq_length)
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # Back to (batch_size, seq_length, embed_size)
        x = self.activation(x)
        x = self.norm(x)

        relative_positions = torch.arange(-seq_length + 1, seq_length, device=x.device)
        relative_positions = relative_positions.clamp(-self.max_relative_position, self.max_relative_position)
        relative_positions = self.relative_positions(relative_positions + self.max_relative_position)

        return x + relative_positions[:seq_length]





class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.head_dim = embed_size // num_heads

        assert self.head_dim * num_heads == embed_size, "Embedding size must be divisible by number of heads"

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, queries, mask=None):
        batch_size = 8
        seq_len = queries.shape[1]
        
        # Split into heads
        values = values.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        keys = keys.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        queries = queries.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        values = values.permute(0, 2, 1, 3)    # [batch_size, num_heads, seq_len, head_dim]
        keys = keys.permute(0, 2, 1, 3)        # [batch_size, num_heads, seq_len, head_dim]
        queries = queries.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
        

        # Calcul de l'énergie d'attention
        energy = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # energy shape: [batch_size, num_heads, seq_len, seq_len]
        

        
        # Attention weights
        attention = torch.softmax(energy, dim=-1)
        
        # Calcul de la sortie
        out = torch.matmul(attention, values)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Reshape final
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        
        # Projection finale
        out = self.fc_out(out)
        
        return out





class TransformerBlockW(nn.Module):
    def __init__(self, d_model, num_heads, dropout, forward_expansion):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, forward_expansion * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(forward_expansion * d_model, d_model)
        )
        
    def forward(self, value, key, query, mask=None):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

