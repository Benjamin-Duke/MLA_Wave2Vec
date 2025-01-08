import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.wav2vec import Wav2Vec2
from src.config.model_config import Wav2Vec2Config

class Wav2Vec2ForCTC(nn.Module):
    def __init__(self, config: Wav2Vec2Config, num_chars=26):
        super().__init__()
        self.wav2vec = Wav2Vec2(config)
        self.char_proj = nn.Linear(config.d_model, num_chars + 1)
        self.dropout = nn.Dropout(config.dropout)
        self.char_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.char_proj.bias.data.zero_()
        
    def forward(self, x, maskBool=False):
        features, _, _ = self.wav2vec(x, maskBool=maskBool)
        x = self.dropout(features)
        logits = self.char_proj(x)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs
    
    def freeze_feature_encoder(self):
        for param in self.wav2vec.feature_encoder.parameters():
            param.requires_grad = False
            
    def prepare_for_fine_tuning(self):
        self.freeze_feature_encoder()
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.1
