import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class FeatureEncoder(nn.Module):
    def __init__(self):
        super(FeatureEncoder, self).__init__()
        # Les sept blocs convolutifs avec 512 canaux et strides spécifiés
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=512, kernel_size=10, stride=5),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=2),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=2),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=2),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=2),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=2, stride=2),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=2, stride=2),
            nn.LayerNorm(512),
            nn.GELU()
        )
        
        # Convolution pour l'embedding de position relative
        # ne doit pas etre mis pour la quantification
        self.positional_embedding = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=128, groups=16)

    def forward(self, x):
        # x : entrée audio brut de forme (batch_size, 1, sequence_length)
        x = self.conv_layers(x)
        
        # Embedding de position relative
        pos_embed = self.positional_embedding(x)
        x = x + F.gelu(pos_embed)  # Ajout de l'embedding de position après activation GELU
        return x


if __name__ == '__main__':
# Paramètres
batch_size = 1
sample_rate = 16000  # Fréquence d'échantillonnage (16 kHz)
duration = 1.0  # Durée du signal audio en secondes
frequency = 440  # Fréquence du signal sinusoïdal en Hz (ex : 440 Hz pour un La)

# Calcul du nombre d'échantillons
sequence_length = int(sample_rate * duration)  # 16000 échantillons pour 1 seconde

# Génération d'un signal sinusoïdal
t = np.linspace(0, duration, sequence_length, endpoint=False)  # Temps
audio_signal = np.sin(2 * np.pi * frequency * t)  # Signal sinusoïdal

# Affichage du signal audio généré
plt.plot(t, audio_signal)
plt.title(f"Signal Audio: {frequency} Hz")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.show()


encoder = FeatureEncoder()

print("Dimension de la sortie encodée :", encoded_output.shape)

