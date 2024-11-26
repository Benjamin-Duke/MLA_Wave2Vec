
import torch
import torch.nn as nn
import numpy as np
# Masking with Learnable Embedding
class MaskingWithLearnableEmbedding(nn.Module):
    """In the paper wav2vec there's an aditional that can be learned and that's the masking
        Instead of zeroing values within a sequence, the masking uses a common vector m 
        which is added in the sequence such as x=[x1,....,m,xk,m,...xn] 
        where x is the sequence and n is length of the sequence (i.e Length of the encoder's output)
        m is learnable parameter
    """
    def __init__(self, embed_size):
        super(MaskingWithLearnableEmbedding, self).__init__()
        self.mask_embedding = nn.Parameter(torch.randn(embed_size))  # Learnable mask embedding

    def forward(self, latent_reps, mask_prob, mask_length):
        batch_size, seq_length, embed_size = latent_reps.shape
        mask = torch.ones_like(latent_reps)
        mask_indices = []

        for b in range(batch_size):
            indices = np.random.choice(seq_length, size=int(mask_prob * seq_length), replace=False)
            mask_indices.append(indices)
            for idx in indices:
                start = idx
                end = min(start + mask_length, seq_length)
                mask[b, start:end] = 0  # Set to zero (masked)
        
        masked_reps = latent_reps.clone()
        for b, indices in enumerate(mask_indices):
            for idx in indices:
                start = idx
                end = min(start + mask_length, seq_length)
                masked_reps[b, start:end] = self.mask_embedding
        
        return masked_reps, mask


# Convolutional Positional Embedding
class ConvolutionalPositionalEmbedding(nn.Module):
    """Instead of using positional Embedding that uses sines and cosines to implement the positional 
        embedding the wav2vec article uses a convolutional layer to perform that  
        """
    def __init__(self, embed_size, kernel_size, groups):
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

    def forward(self, x):
        # x: [batch_size, seq_length, embed_size] -> [batch_size, embed_size, seq_length]
        x = x.permute(0, 2, 1)
        x = self.conv(x)  # Convolution along the sequence dimension
        x = x.permute(0, 2, 1)  # Back to [batch_size, seq_length, embed_size]
        x = self.activation(x)
        x = self.norm(x)
        return x

# Multi-Head Attention 
class MultiHeadAttention(nn.Module):
    """
       Multi head attention computes attention (i.e the importance of a 
       given sets of tokens to predict other tokens) over a number of "heads"
       where each head's attention could be specialized in a given aspect of the sequence, such as phoneme recognition 
       At the end all the attentions are concatenated
    """
    def __init__(self, embed_size, num_heads):
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
class TransformerBlockW(nn.Module):
    """
    During the pre-training of the whole model, only the encoder of the transformer is used
    This due to the fact that the purpose isn't to generate output but rather to identify (i.e Discriminate)
    the correct quantized laten audio representation. Hence, the decoder block will be added in
    the future versions of the hereby script 
    """
    def __init__(self, embed_size, num_heads, dropout, forward_expansion, kernel_size, groups):
        super(TransformerBlockW, self).__init__()
        self.positional_embedding = ConvolutionalPositionalEmbedding(embed_size, kernel_size, groups)
        self.attention = MultiHeadAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask=None):
        pos_embedded = self.positional_embedding(query)  # Add positional embeddings
        attention = self.attention(value, key, pos_embedded, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

"""
# Parameters (To be adjusted to the output of the encoder block)
batch_size = 16
seq_length = 49 # In the paper the output of the encoder block has a frequency of 49Hz, we are assuming that each input is a 1s input 
embed_size = 64
mask_prob = 0.15
mask_length = 10
num_heads = 8
dropout = 0.1
forward_expansion = 4
kernel_size = 31
groups = 16

# Random latent encoder output
latent_reps = torch.rand(batch_size, seq_length, embed_size)

# Apply masking
masking_layer = MaskingWithLearnableEmbedding(embed_size)
masked_reps, mask = masking_layer(latent_reps, mask_prob, mask_length)

# Define and apply Transformer with Convolutional Positional Embedding
transformer_block = TransformerBlockW(
    embed_size, num_heads, dropout, forward_expansion, kernel_size, groups
)
contextualized_reps = transformer_block(masked_reps, masked_reps, masked_reps, None)

# Output dimensions
print("Contextualized Representations Shape:", contextualized_reps.shape)
"""