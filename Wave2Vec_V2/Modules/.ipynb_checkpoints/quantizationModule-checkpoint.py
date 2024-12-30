import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ProductQuantizer(nn.Module):
    def __init__(self, input_dim, num_codebooks=2, codebook_size=320, temp=2.0, min_temp=0.5, temp_decay=0.999995):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.temp = temp
        self.min_temp = min_temp
        self.temp_decay = temp_decay

        self.codebook_dim = input_dim // num_codebooks

        self.codebooks = nn.Parameter(torch.FloatTensor(num_codebooks * codebook_size, self.codebook_dim))
        nn.init.uniform_(self.codebooks)

        self.weight_proj = nn.Linear(input_dim, num_codebooks * codebook_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.dropout(x)
        bsz, tsz, fsz = x.shape

        # Project to G x V logits
        x = self.weight_proj(x)
        x = x.view(bsz * tsz, self.num_codebooks, self.codebook_size)

        if self.training:
            # Gumbel noise
            uniform_noise = torch.rand_like(x)
            gumbel = -torch.log(-torch.log(uniform_noise + 1e-10) + 1e-10)

            # Apply formula: exp((l_{g,v} + n_v)/τ) / sum_k(exp((l_{g,k} + n_k)/τ))
            logits_with_noise = (x + gumbel) / self.temp
            numerator = torch.exp(logits_with_noise)
            denominator = numerator.sum(dim=-1, keepdim=True)
            x = numerator / denominator

            # Update temperature
            self.temp = max(self.temp * self.temp_decay, self.min_temp)
        else:
            # During inference, use straight-through estimator
            logits = x / self.temp
            x = F.softmax(logits, dim=-1)

        # Straight-through Gumbel-Softmax
        indices = x.max(dim=-1)[1]
        x_hard = torch.zeros_like(x).scatter_(-1, indices.unsqueeze(-1), 1.0)
        x = (x_hard - x).detach() + x

        return x.view(bsz, tsz, -1)