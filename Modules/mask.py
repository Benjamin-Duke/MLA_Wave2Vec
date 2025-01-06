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
        
        # Calculate the total number of masks needed
        # We want mask_prob percentage of the total sequence to be masked
        total_masked_length = int(T * self.mask_prob)
        # Calculate how many mask segments we need
        num_masks = math.ceil(total_masked_length / self.mask_length)
        
        # Create mask tensor
        mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
        
        # For each batch
        for batch_i in range(B):
            # Calculate valid starting positions (ensure we don't go out of bounds)
            valid_starts = T - self.mask_length + 1
            if valid_starts <= 0:
                continue
                
            # Sample starting indices without replacement
            if num_masks < valid_starts:
                starts = torch.randperm(valid_starts, device=x.device)[:num_masks]
            else:
                starts = torch.randperm(valid_starts, device=x.device)
                
            # Apply masks
            for start in starts:
                end = min(start + self.mask_length, T)
                mask[batch_i, start:end] = True
                
        return mask