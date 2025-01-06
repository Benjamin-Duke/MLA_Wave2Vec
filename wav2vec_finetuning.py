import torch
import torch.nn as nn
import torch.nn.functional as F
from wave2vec import Wav2Vec2
from Modules.config import Wav2Vec2Config

class Wav2Vec2ForCTC(nn.Module):
    def __init__(self, config: Wav2Vec2Config, num_chars=26):
        super().__init__()
        
        # Load pre-trained model
        self.wav2vec = Wav2Vec2(config)
        
        # Add output layer for character prediction
        # num_chars + 1 for blank token in CTC
        self.char_proj = nn.Linear(config.d_model, num_chars + 1)
        
        # Dropout before the output layer
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize the new projection layer
        self.char_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.char_proj.bias.data.zero_()
        
    def forward(self, x, maskBool=False):
        # Get wav2vec features (we don't need q or mask_indices for fine-tuning)
        features, _, _ = self.wav2vec(x, maskBool=maskBool)
        
        # Apply dropout and project to character probabilities
        x = self.dropout(features)
        logits = self.char_proj(x)
        
        # Apply log softmax for CTC loss
        log_probs = F.log_softmax(logits, dim=-1)
        
        return log_probs
    
    def freeze_feature_encoder(self):
        """Freeze the feature encoder parameters as specified in the paper."""
        for param in self.wav2vec.feature_encoder.parameters():
            param.requires_grad = False
            
    def prepare_for_fine_tuning(self):
        """Prepare model for fine-tuning according to the paper:
        1. Freeze feature encoder
        2. Set appropriate dropout rates
        3. Apply LayerDrop to transformer
        """
        # 1. Freeze feature encoder
        self.freeze_feature_encoder()
        
        # 2. Set dropout rates
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.1  # Standard dropout rate for fine-tuning
                
        # 3. Set LayerDrop rate (already set in config)
        # The transformer already uses the layer_drop rate from config 