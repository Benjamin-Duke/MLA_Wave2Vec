import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
       Multi head attention computes attention (i.e the importance of a 
       given sets of tokens to predict other tokens) over a number of "heads"
       where each head's attention could be specialized in a given aspect of the sequence, such as phoneme recognition 
       At the end all the attentions are concatenated
    """
    def __init__(self, embed_size=768, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        assert embed_size % num_heads == 0, "Embed size must be divisible by the number of heads"
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
    """
    During the pre-training of the whole model, only the encoder of the transformer is used
    This due to the fact that the purpose isn't to generate output but rather to identify (i.e Discriminate)
    the correct quantized laten audio representation. Hence, the decoder block will be added in
    the future versions of the hereby script 
    """
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
    """
    Transformer that uses layers of TransformerEncoderLayer
    """
    def __init__(self, encoder_layer, num_encoder_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_encoder_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
    

# class TransformerEncoder(nn.Module):
#     def __init__(self, encoder_layer, num_encoder_layers):
#         super().__init__()
#         self.encoder_layer = encoder_layer
#         self.num_encoder_layers = num_encoder_layers
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

#     def forward(self,x):
#         return self.transformer(x)