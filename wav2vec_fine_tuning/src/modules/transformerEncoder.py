import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size=768, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        assert embed_size % num_heads == 0, 
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        # Linear layers for Query, Key, and Value
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        
        # Final linear layer after concatenating heads
        self.fc_out = nn.Linear(embed_size, embed_size)
    
    def forward(self, value, key, query, mask=None):
        N, seq_length, embed_size = query.shape
        Q = self.query(query).reshape(N, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.key(key).reshape(N, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.value(value).reshape(N, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)  # Scaled dot-product
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention = torch.softmax(scores, dim=-1)
        out = torch.matmul(attention, V)  # [N, num_heads, seq_length, head_dim]
        out = out.permute(0, 2, 1, 3).reshape(N, seq_length, embed_size)  # Recombine heads
        out = self.fc_out(out)
        return out

# Transformer Block with Convolutional Positional Embedding
class TransformerEncoderLayer(nn.Module):
    def __init__(self):
        super(TransformerEncoderLayer, self).__init__()
        embed_size, num_heads, dropout, forward_expansion, kernel_size, groups=768,8,.1,3072,128,16
        self.attention = MultiHeadAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        mask=None
        value, key, query=x.clone(),x.clone(),x.clone()
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_encoder_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_encoder_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
    
