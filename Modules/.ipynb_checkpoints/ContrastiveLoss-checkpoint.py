import torch
import torch.nn.functional as F

class LossW2V:
    def __init__(self, K, temperature=0.1):
        self.K = K
        self.temperature = temperature

    def contrastive_loss(self, context_repr, quantized_repr, mask_indices):
        
        #context_repr = context_repr.unsqueeze(2)  # [batch_size, seq_len, 1, feature_dim ]

        #print("[batch_size, seq_len, 1, feature_dim ] ? input context_repr shape : ", context_repr.shape)

        cos_sim = F.cosine_similarity(context_repr, quantized_repr, dim=-1)  # [ batch_size, seq_len, K+1 ]

        #print("[cos_sim ", cos_sim.shape)

        cos_sim_scaled = cos_sim / self.temperature
        logits = cos_sim_scaled 
        probabilities = F.softmax(logits, dim=-1)

        probabilities = probabilities.unsqueeze(-1)
        
        #target = mask_indices  # [batch_size, seq_len]
        target = mask_indices.long()

        #print("[DEBUG] mask_indices shape:", mask_indices.shape)



        #target = mask_indices[:, :, 0]  # Par exemple, choisissez la première dimension ou une autre

        # Ensuite, vous pouvez appliquer F.one_hot
        target_one_hot = F.one_hot(target, num_classes=self.K+1)  #  [batch_size, seq_len, K+1]

        #print("[DEBUG] target_one_hot shape:", target_one_hot.shape)
        #print("[DEBUG] probabilities shape:", probabilities.shape)

        loss = -torch.sum(target_one_hot * torch.log(probabilities + 1e-8))  # [batch_size, seq_len]
        
        return torch.mean(loss) #MEAN Batched loss 

    def diversity_loss(self, quantized_repr, batch_size):
        #print("[batch_size * seq_len, K+1, feature_dim] ? input quantized_repr shape : ", quantized_repr.shape)
        quantized_repr_flat = quantized_repr.view(-1, quantized_repr.size(-1))  #[batch_size * seq_len, K+1, feature_dim]
        _, counts = quantized_repr_flat.unique(dim=0, return_counts=True)
        probabilities = counts.float() / (batch_size * quantized_repr.size(1))
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8))
        return entropy

    def compute_loss(self, context_repr, quantized_repr, mask_indices, batch_size):
        contrastive = self.contrastive_loss(context_repr, quantized_repr, mask_indices)
        diversity = self.diversity_loss(quantized_repr, batch_size)
        total_loss = contrastive + diversity
        return total_loss
