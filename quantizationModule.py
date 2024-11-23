import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedGumbelQuantizer(nn.Module):
    """
    Module de quantification basé sur le softmax de Gumbel pour discrétiser 
    des représentations continues tout en restant différentiable.

    Arguments :
    - num_codebooks (int) : Nombre de codebooks (groupes de codes).
    - num_codes (int) : Nombre de vecteurs de code par codebook.
    - code_dim (int) : Dimension des vecteurs de code.
    - output_dim (int) : Dimension de la sortie après transformation linéaire.
    - temperature (float) : Température pour contrôler la discrétisation.

    Méthode :
    - forward(z) : Effectue la quantification en utilisant les codebooks 
      et renvoie la sortie quantifiée après une transformation linéaire.
    """
    def __init__(self, num_codebooks=2, num_codes=320, code_dim=128, output_dim=256, temperature=1.0):
        super(EnhancedGumbelQuantizer, self).__init__()
        self.num_codebooks = num_codebooks
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.temperature = temperature
        
        # Initialisation des codebooks
        self.codebooks = nn.Parameter(torch.randn(num_codebooks, num_codes, code_dim))
        
        # Transformation linéaire finale pour obtenir la dimension souhaitée
        self.linear = nn.Linear(num_codebooks * code_dim, output_dim)

    def forward(self, z):
        """
        Effectue le passage avant du quantiseur. Prend en entrée un tenseur `z`,
        calcule les logits entre `z` et les codebooks, applique le bruit de Gumbel,
        puis sélectionne les codes via un softmax différentiable. Enfin, la sortie
        est projetée dans la dimension de sortie via une transformation linéaire.

        Arguments :
        - z (Tensor) : Tenseur d'entrée de forme (batch_size, sequence_length, num_codebooks * code_dim)

        Sortie :
        - Tensor : Tenseur quantifié de forme (batch_size, sequence_length, output_dim)
        """
        z = z.view(z.shape[0], z.shape[1], self.num_codebooks, self.code_dim)
        
        # Calcul des logits et ajout du bruit de Gumbel
        logits = torch.einsum('btsd,gvd->btsg', z, self.codebooks)
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
        logits = (logits + gumbel_noise) / self.temperature
        
        # Sélection différentiable des codes via le softmax de Gumbel
        softmax_weights = F.softmax(logits, dim=-1)
        quantized = torch.einsum('btsg,gvd->btsd', softmax_weights, self.codebooks)
        quantized = quantized.view(z.shape[0], z.shape[1], -1)
        
        # Transformation linéaire finale pour la dimension de sortie
        quantized = self.linear(quantized)
        
        return quantized