import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Mask(nn.Module):
    def __init__(self, mask_prob=0.065, mask_length=10):
        super().__init__()
        self.mask_prob = mask_prob
        self.mask_length = mask_length

    def forward(self, x):
        B, T, C = x.shape
        
        # Calculate how many starting indices to sample
        num_mask = int(T * self.mask_prob)
        
        # Sample starting indices
        mask_starts = torch.randperm(T)[:num_mask]
        
        # Create mask tensor
        mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
        
        # For each starting index, mask the subsequent M time steps
        for start in mask_starts:
            end = min(start + self.mask_length, T)
            mask[:, start:end] = True
            
        return mask