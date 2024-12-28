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
        batch_size, seq_length, hidden_size = x.shape
        
        # Create boolean mask (True = keep, False = mask)
        time_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=x.device)
        
        # Calculate spans to mask
        num_spans = int(seq_length * mask_prob / mask_length)
        
        # Apply masking per batch
        for batch_idx in range(batch_size):
            # Get random starting positions
            starts = torch.randperm(seq_length - mask_length + 1)[:num_spans]
            
            # Create spans
            for start in starts:
                end = start + mask_length
                time_mask[batch_idx, start:end] = False
        
        # Expand mask for hidden states
        expanded_mask = time_mask.unsqueeze(-1).expand_as(x)
        
        # Apply mask to input
        masked_x = x * expanded_mask
        
        return masked_x, time_mask


class ConvolutionalPositionalEmbedding(nn.Module):

    def __init__(self, embed_size, kernel_size, groups):
        super(ConvolutionalPositionalEmbedding, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=embed_size,
            out_channels=embed_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=16
        )
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, x):
        # x: [batch_size, seq_length, embed_size] -> [batch_size, embed_size, seq_length]

        x = x.permute(0, 2, 1)

        x = self.conv(x)
        
        x = x.permute(0, 2, 1)  # Back to [batch_size, seq_length, embed_size]
        
        x = self.activation(x[:,:-1,:])

        #x = self.norm(x)
                     
        return x

#We add the output of the convolution followed by a GELU to the inputs and then apply layer normalization.





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

        
        # Vérification de la forme de key
    
        # Calcul des têtes
        head_dim = embed_size // self.num_heads
    
        # Vérification de la forme de Q, K et V avant reshape
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
    
        # Appliquer reshape et permute
        Q = Q.reshape(N, seq_length, self.num_heads, head_dim).permute(0, 2, 1, 3)
        K = K.reshape(N, seq_length, self.num_heads, head_dim).permute(0, 2, 1, 3)
        V = V.reshape(N, seq_length, self.num_heads, head_dim).permute(0, 2, 1, 3)
        
    
        # Calcul des scores d'attention
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (head_dim ** 0.5)

        #if mask is not None:
        #    scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = torch.softmax(scores, dim=-1)
        out = torch.matmul(attention, V)  # [N, num_heads, seq_length, head_dim]
        

        
        out = out.permute(0, 2, 1, 3).reshape(N, seq_length, embed_size)  # Recombine heads

        
        out = self.fc_out(out)

    
        return out
    
    



class TransformerBlockW(nn.Module):
    def __init__(self, embed_size, num_heads, dropout, forward_expansion,max_relative_position):
        super().__init__()

        #embed_size, kernel_size, groups):
        self.positional_embedding = ConvolutionalPositionalEmbedding(embed_size, 128, 2)
        
        self.attention = MultiHeadAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout) #embed_size == d_model ?
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        
    def forward(self, value, key, query, mask=None):
       # print("key",key.shape)
      #  print("query",query.shape)
        query = self.positional_embedding(query)

     #   print("queryap",query.shape)
    #    print("keyapre",key.shape)
        #print(f"pos_emb.shape (tbw): {pos_emb.shape}")
    
        attention = self.attention(value, key, query, mask)
   #     print("keyapressss",key.shape)
        
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

