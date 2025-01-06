import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_encoder_layers):
        super().__init__()
        self.encoder_layer = encoder_layer
        self.num_encoder_layers = num_encoder_layers
        self.transformer = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

    def forward(self,x):
        return self.transformer(x)