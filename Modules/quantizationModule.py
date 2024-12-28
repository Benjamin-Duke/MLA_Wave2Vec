import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

class STEQuantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, codebook):
        batch_size, num_tokens, feature_dim = z.shape
        V, _ = codebook.shape  # V is the number of codebook entries
        Wq = codebook.t()  
        logits = torch.matmul(z, Wq)  # Project z to logits using Wq
        ctx.save_for_backward(z, logits, codebook)
        return logits

    @staticmethod
    def backward(ctx, grad_output):
        # STE principle
        z, logits, codebook = ctx.saved_tensors
        grad_z = torch.matmul(grad_output, codebook)  # Gradient w.r.t. z (Shape: (batch_size, num_tokens, feature_dim))
        return grad_z, None


class QuantizationModule(nn.Module):
    def __init__(self, input_dim, codebook_size, num_codebooks, output_dim, temperature=1.0):
        super().__init__()
        
        self.model_input_dimension = input_dim  # Dimension de sortie du feature encoder
        self.codebook_size = codebook_size  # Nombre de clusters du codebook
        self.num_codebooks = num_codebooks  # Nombre de codebooks
        self.temperature = temperature
        
        # Dimension de chaque codebook
        self.codebook_dim = input_dim // num_codebooks
        
        # Projection finale
        self.W_p = nn.Parameter(torch.randn(self.codebook_size, self.model_input_dimension))

    def forward(self, z): 
        batch_size, num_tokens, feature_dim = z.shape  # Ex: 8,49,512
        z_flat = z.view(batch_size * num_tokens, feature_dim)

        kmeans = KMeans(n_clusters=self.codebook_size, random_state=42).fit(z_flat.cpu().detach().numpy())  # Clustering K-means
        codebook = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, requires_grad=True).to(z.device)

        # Application du STE 
        logits = STEQuantization.apply(z, codebook)

        one_hot_vect = F.gumbel_softmax(logits, tau=1.0, hard=True)

        quantized_output = torch.matmul(one_hot_vect, self.W_p)  # Shape: (batch_size, num_tokens, output_dim)

        avg_softmax_probs = one_hot_vect.mean(dim=0)  # Moyenne sur le batch et les tokens (forme: [num_codebooks, V])
        diversity_loss = -torch.sum(avg_softmax_probs * torch.log(avg_softmax_probs + 1e-10), dim=-1).mean()

        return quantized_output, diversity_loss