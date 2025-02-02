import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FeatureEncoder(nn.Module):
    def __init__(self, conv_layers=[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)*2]):
        super().__init__()
        
        layers = []
        in_channels = 1
    
        layers.append(
            nn.Sequential(
                nn.Conv1d(in_channels, conv_layers[0][0], conv_layers[0][1], stride=conv_layers[0][2]),
                nn.GELU(),
                nn.Dropout(0.1),
            )
        )
        self.layer_norm = nn.LayerNorm(conv_layers[0][0])
        in_channels = conv_layers[0][0]
        for out_channels, kernel_size, stride in conv_layers[1:]:
            layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride),
                    nn.GELU(),
                    nn.Dropout(0.1),
                )
            )
            in_channels = out_channels
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        x = x.unsqueeze(1)  
        x = self.layers[0](x)

        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        x = x.transpose(1, 2)

        for layer in self.layers[1:]:
            x = layer(x)
        
        return x.transpose(1, 2) 