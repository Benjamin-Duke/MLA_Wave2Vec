import torch
import torch.nn.functional as F

class LossW2V:
    def __init__(self, K, temperature=1.0):
        self.K = K
        self.temperature = temperature
    
    def contrastive_loss(self, context_repr, quantized_repr, mask_indices):
        # context_repr: [batch_size, seq_len, feature_dim]
        # quantized_repr devrait être: [batch_size, seq_len, K+1, feature_dim]
        # mask_indices: [batch_size, seq_len, feature_dim]
        
        batch_size, seq_len, feature_dim = context_repr.shape
        
        # Reshape quantized_repr pour inclure la dimension K+1
        # On va simuler les K négatifs en prenant des échantillons aléatoires
        # du batch comme négatifs
        quantized_expanded = quantized_repr.unsqueeze(2)  # [batch_size, seq_len, 1, feature_dim]
        
        # Créer les négatifs en mélangeant le batch
        negative_samples = []
        for _ in range(self.K):
            perm = torch.randperm(batch_size)
            negative_samples.append(quantized_repr[perm].unsqueeze(2))
        
        # Concaténer le positif et les négatifs
        quantized_with_negatives = torch.cat([quantized_expanded] + negative_samples, dim=2)
        # Maintenant shape: [batch_size, seq_len, K+1, feature_dim]
        
        # Normaliser
        context_repr = F.normalize(context_repr, p=2, dim=-1)
        quantized_with_negatives = F.normalize(quantized_with_negatives, p=2, dim=-1)
        
        # Expand context pour le produit
        context_expanded = context_repr.unsqueeze(2)  # [batch_size, seq_len, 1, feature_dim]
        
        # Calcul similarité cosinus
        cos_sim = torch.sum(context_expanded * quantized_with_negatives, dim=-1)  # [batch_size, seq_len, K+1]
        
        # Température
        logits = cos_sim / self.temperature
        
        # Labels (premier index = positif)
        labels = torch.zeros(batch_size, seq_len, dtype=torch.long, device=context_repr.device)
        

        # Reshape pour cross entropy
        logits = logits.reshape(-1, self.K + 1)
        labels = labels.reshape(-1)
        
        loss = F.cross_entropy(logits, labels)
        return loss
    
    def diversity_loss(self, quantized_repr, batch_size):
        # Reshape pour avoir tous les vecteurs en 2D
        quantized_flat = quantized_repr.reshape(-1, quantized_repr.size(-1))
        
        # Calculer les centroïdes moyens
        num_vectors = quantized_flat.size(0)
        
        # Calculer la matrice de similarité cosinus entre tous les vecteurs
        similarity_matrix = F.cosine_similarity(
            quantized_flat.unsqueeze(1),
            quantized_flat.unsqueeze(0),
            dim=2
        )
        
        # Masquer la diagonale
        mask = ~torch.eye(num_vectors, device=similarity_matrix.device).bool()
        similarity_matrix = similarity_matrix * mask
        
        # Calculer la diversité comme la négative de la similarité moyenne
        diversity = -torch.mean(similarity_matrix)
        
        return diversity
        
    def compute_loss(self, context_repr, quantized_repr, mask_indices, batch_size):
        contrastive = self.contrastive_loss(context_repr, quantized_repr, mask_indices)
        diversity = self.diversity_loss(quantized_repr, batch_size)
        
        lambda_div = 0.1
        total_loss = contrastive + lambda_div * diversity
        
        return total_loss