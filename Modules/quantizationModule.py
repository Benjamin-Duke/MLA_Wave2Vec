import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantizationModule(nn.Module):
    """
    Quantization module based on Gumbel-Softmax for discretizing 
    continuous representations while remaining differentiable.

    Arguments:
    - num_codebooks (int): Number of codebooks (groups of codes).
    - num_codes (int): Number of code vectors per codebook.
    - code_dim (int): Dimension of the code vectors.
    - output_dim (int): Dimension of the output after linear transformation.
    - temperature (float): Temperature to control the discretization.

    Method:
    - forward(z): Performs quantization using the codebooks 
      and returns the quantized output after a linear transformation.
    """
    def __init__(self, num_codebooks, num_codes, code_dim=256, output_dim=256, temperature=1.0):
        super(QuantizationModule, self).__init__()
        self.num_codebooks = num_codebooks
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.temperature = temperature
        
        # Initialize the codebooks
        self.codebooks = nn.Parameter(torch.randn(num_codebooks, num_codes, code_dim))
        
        # Final linear transformation to achieve the desired output dimension
        self.linear = nn.Linear(num_codebooks * code_dim, output_dim)

    def forward(self, z):
        """
        Performs the forward pass of the quantizer. Takes an input tensor `z`,
        computes the logits between `z` and the codebooks, applies Gumbel noise,
        then selects codes using differentiable softmax. Finally, the output
        is projected into the output dimension using a linear transformation.

        Arguments:
        - z (Tensor): Input tensor of shape (batch_size, sequence_length, num_codebooks * code_dim)

        Returns:
        - Tensor: Quantized tensor of shape (batch_size, sequence_length, output_dim)
        """

        print("la",self.num_codebooks, self.code_dim)
        z = z.view(z.shape[0], z.shape[1], 2, 256)
        
        # Compute logits and add Gumbel noise
        logits = torch.einsum('btsd,gvd->btsg', z, self.codebooks)
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
        logits = (logits + gumbel_noise) / self.temperature
        
        # Differentiable selection of codes using Gumbel-Softmax
        softmax_weights = F.softmax(logits, dim=-1)
        quantized = torch.einsum('btsg,gvd->btsd', softmax_weights, self.codebooks)
        quantized = quantized.view(z.shape[0], z.shape[1], -1)
        
        # Final linear transformation to the output dimension
        quantized = self.linear(quantized)
        
        return quantized
