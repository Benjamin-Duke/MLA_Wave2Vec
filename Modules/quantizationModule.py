import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantizationModule(nn.Module):
    def __init__(self, input_dim, codebook_size, num_codebooks, output_dim, temperature=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        self.temperature = temperature
        
        # Dimension de chaque codebook
        self.codebook_dim = input_dim // num_codebooks
        
        # Projection vers les logits (ajustée pour la dimension de séquence)
        self.logits_projection = nn.Linear(self.codebook_dim, codebook_size)
        
        # Codebooks
        self.codebooks = nn.Parameter(torch.randn(num_codebooks, codebook_size, self.codebook_dim))
        
        # Projection finale
        self.output_projection = nn.Linear(input_dim, output_dim)

    def forward(self, z):
        batch_size, seq_length, _ = z.shape
        
        # Reshape pour traiter chaque groupe séparément
        z = z.reshape(batch_size, seq_length, self.num_codebooks, self.codebook_dim)
        
        # Calculer les logits pour chaque groupe
        logits = self.logits_projection(z)  # [batch, seq, num_codebooks, codebook_size]
        
        if self.training:
            # Gumbel noise
            uniform_noise = torch.rand_like(logits)
            gumbel_noise = -torch.log(-torch.log(uniform_noise + 1e-10) + 1e-10)
            logits = (logits + gumbel_noise) / self.temperature
        
        # Softmax sur les logits
        probs = F.softmax(logits, dim=-1)
        
        # Straight-through estimator
        indices = torch.argmax(probs, dim=-1)
        hard_probs = F.one_hot(indices, self.codebook_size).float()
        quantized = hard_probs - probs.detach() + probs
        
        # Sélection des entrées du codebook
        quantized = torch.einsum('bsgv,gvd->bsgd', quantized, self.codebooks)
        
        # Reshape et projection finale
        quantized = quantized.reshape(batch_size, seq_length, -1)
        output = self.output_projection(quantized)
        
        return output