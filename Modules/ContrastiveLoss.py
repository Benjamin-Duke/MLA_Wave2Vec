import torch
from torch import nn
import random

class Wav2vec2Loss(nn.Module):
    # (K , k temp, G codevectorgroup, Vcodevectorpergroup, a 0,05)
    def __init__(self, K, k, G, V, a):
        super().__init__()
        self.K = K  # Number of negative samples
        self.k = k  # Temperature parameter
        self.G = G  # Number of code vector groups
        self.V = V  # Number of code vectors per group
        self.alpha = 0.4  # Weight for diversity loss
        self.cos = nn.CosineSimilarity(dim=-1)  # Cosine similarity function

    def forward(self, context_repr, quantized_features, diversity_loss, time_mask):
        # Get masked positions for each batch
        masked_indices = ~time_mask  # Convert to True where masked
        
        # Extract masked features
        masked_contexts = []
        masked_quantized = []
        
        for b in range(context_repr.size(0)):
            batch_indices = masked_indices[b].nonzero().squeeze(-1)
            masked_contexts.append(context_repr[b, batch_indices])
            masked_quantized.append(quantized_features[b, batch_indices])
        
        target_context_repr = torch.cat(masked_contexts, dim=0)
        labels = torch.cat(masked_quantized, dim=0)
        
        # Get number of targets per batch
        num_targets_per_batch = [int(masked_indices[i].sum()) for i in range(masked_indices.size(0))]
        
        # Generate negatives and compute losses
        negative_samples = self.negative_sampler(labels, num_targets_per_batch)
        negative_samples = torch.cat([labels.unsqueeze(1), negative_samples], dim=1)
        
        contrastive_loss = self.contrastive_loss(target_context_repr, labels, negative_samples)
        #diversity_loss = self.diversity_loss(perplexity)
        
        return contrastive_loss + self.alpha * diversity_loss


    
    def contrastive_loss(self, context_repr, labels, negative_samples):

        # Compute similarity for positive samples (positive similarity)
        positive_similarity = torch.exp(self.cos(context_repr, labels) / self.k)
        
        # Compute similarity for negative samples (negative similarity)
        negative_similarity = torch.sum(torch.exp(self.cos(context_repr.unsqueeze(1), negative_samples) / self.k), dim=1)

        # Compute the contrastive loss using the formula
        contrastive_loss = -torch.log(positive_similarity / negative_similarity).mean()
        
        return contrastive_loss

    
    def negative_sampler(self, labels, num_targets_per_batch):
        negative_samples = []
        start_idx = 0
        
        for num_targets in num_targets_per_batch:
            if num_targets == 0:  # Skip empty batches
                continue
                
            # Create mask for valid negative samples
            candidates = torch.arange(num_targets, device=labels.device)
            mask = torch.ones((num_targets, num_targets), device=labels.device, dtype=torch.bool)
            mask[torch.arange(num_targets), torch.arange(num_targets)] = False
            
            # Sample K negatives for each target
            for i in range(num_targets):
                valid_negatives = candidates[mask[i]]
                if len(valid_negatives) >= self.K:
                    neg_indices = valid_negatives[torch.randperm(len(valid_negatives))[:self.K]]
                else:
                    # If not enough negatives, sample with replacement
                    neg_indices = valid_negatives[torch.randint(len(valid_negatives), (self.K,))]
                
                negative_samples.append(labels[start_idx + neg_indices])
                
            start_idx += num_targets
        
        # Reshape to [N, K, D] where N is total number of targets
        return torch.stack(negative_samples).view(-1, self.K, labels.size(-1))
