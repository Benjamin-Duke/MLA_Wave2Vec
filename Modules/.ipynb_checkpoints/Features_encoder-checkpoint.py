
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn

class FeatureEncoder(nn.Module):
    """
    Encodeur de caractéristiques audio avec des convolutions, normalisation et GELU.
    Transforme les formes d'onde brutes en représentations compactes.
    """
    def __init__(self, input_channels=1, feature_dim=512):
        super(FeatureEncoder, self).__init__()

        # Définition explicite des couches convolutionnelles
        self.conv1 = nn.Conv1d(1, 512, kernel_size=10, stride=5, padding=0)
        self.norm1 = nn.LayerNorm([512])  # Normalisation sur les canaux
        self.gelu1 = nn.GELU()

        self.conv2 = nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=0)
        self.norm2 = nn.LayerNorm([512])
        self.gelu2 = nn.GELU()

        self.conv3 = nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=0)
        self.norm3 = nn.LayerNorm([512])
        self.gelu3 = nn.GELU()

        self.conv4 = nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=0)
        self.norm4 = nn.LayerNorm([512])
        self.gelu4 = nn.GELU()

        self.conv5 = nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=0)
        self.norm5 = nn.LayerNorm([512])
        self.gelu5 = nn.GELU()

        self.conv6 = nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=0)
        self.norm6 = nn.LayerNorm([512])
        self.gelu6 = nn.GELU()

        self.conv7 = nn.Conv1d(512, feature_dim, kernel_size=2, stride=2, padding=0)
        self.norm7 = nn.LayerNorm([512])
        self.gelu7 = nn.GELU()

        # Convolution pour l'encoding positionnel relatif
        self.positional_encoding = nn.Conv1d(512, 512, kernel_size=128, groups=16, padding=0)

    def forward(self, x):
        """
        Passage avant de l'encodeur.
        Args:
            x (torch.Tensor): Formes d'onde (batch_size, 1, sequence_length)
        Returns:
            torch.Tensor: Représentations latentes (batch_size, feature_dim, reduced_sequence_length)
        """
        # Passage explicite par chaque couche sans boucle

        # Première couche : Convolution -> Permute -> Normalisation -> GELU
        x = self.conv1(x)
        x = x.permute(0, 2, 1)  # Permutation pour changer les dimensions (batch, seq_len, channels)
        x = self.norm1(x)       # Normalisation
        x = self.gelu1(x)       # Activation GELU

    
      #  print("1",x.shape)
        
        # Deuxième couche : Convolution -> Permute -> Normalisation -> GELU
        x = self.conv2(x.permute(0, 2, 1))  # Vous devez permuter avant d'appliquer conv2
        x = x.permute(0, 2, 1)  # Permuter après la convolution
        x = self.norm2(x)
        x = self.gelu2(x)

       # print("2",x.shape)
        
        #print(x.shape)
        # Troisième couche : Convolution -> Permute -> Normalisation -> GELU
        x = self.conv3(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = self.norm3(x)
        x = self.gelu3(x)

       # print("3",x.shape)

        # Quatrième couche : Convolution -> Permute -> Normalisation -> GELU
        x = self.conv4(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = self.norm4(x)
        x = self.gelu4(x)


       # print("4",x.shape)

        
        # Cinquième couche : Convolution -> Permute -> Normalisation -> GELU
        x = self.conv5(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = self.norm5(x)
        x = self.gelu5(x)
        #print(x.shape)

       # print("5",x.shape)

        # Sixième couche : Convolution -> Permute -> Normalisation -> GELU
        x = self.conv6(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = self.norm6(x)
        x = self.gelu6(x)

       # print("6",x.shape)

        # Septième couche : Convolution -> Permute -> Normalisation -> GELU
        x = self.conv7(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = self.norm7(x)
        x = self.gelu7(x)


       # print("7", x.shape)
        # Ajout des embeddings positionnels relatifs
        #x = self.positional_encoding(x.permute(0, 2, 1))
        #x = x.permute(0, 2, 1)

        #print("pos embde feature encoder :", x.shape)
        
        return x



"""
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

        
        """
"""
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

"""
