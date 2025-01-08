import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class TransformerLMConfig:
    def __init__(self):
        self.n_layers = 4
        self.d_model = 256
        self.d_ff = 1024
        self.n_heads = 8
        self.dropout = 0.1
        self.vocab_size = 27
        self.max_seq_length = 512

class LambdaLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        assert vocab_size == 27, "Vocabulary size must be 27 (blank + 26 letters)"
        self.vocab_size = vocab_size
        print(f"Initialized LambdaLM with vocab_size: {vocab_size} (0=blank, 1-26=a-z)")
    
    def forward(self, input_ids, attention_mask=None):
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        return torch.full((batch_size, seq_len, self.vocab_size), 
                        fill_value=-math.log(self.vocab_size), 
                        device=input_ids.device)
    
    def score_sequence(self, sequence):
        with torch.no_grad():
            print(f"LambdaLM.score_sequence - Input shape: {sequence.shape}")
            batch_size, beam_size, seq_len = sequence.size()
            scores = torch.full(
                (batch_size * beam_size, seq_len, self.vocab_size),
                fill_value=-math.log(self.vocab_size),
                device=sequence.device
            )
            scores[..., 0] += 0.1
            print(f"LambdaLM.score_sequence - Output shape: {scores.shape}")
            return scores

class TransformerLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        position = torch.arange(config.max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.d_model, 2) * (-math.log(10000.0) / config.d_model))
        pe = torch.zeros(1, config.max_seq_length, config.d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        if config.d_model % 2 == 1:
            pe[0, :, 1::2] = torch.cos(position * div_term)[..., :-1]
        else:
            pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('position_embedding', pe)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.n_layers)
        
        self.output_layer = nn.Linear(config.d_model, config.vocab_size)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, input_ids, attention_mask=None):
        x = self.token_embedding(input_ids)
        seq_len = input_ids.size(1)
        x = x + self.position_embedding[:, :seq_len, :]
        x = self.dropout(x)
        
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            attention_mask = ~attention_mask
        
        x = self.transformer(x, src_key_padding_mask=attention_mask)
        logits = self.output_layer(x)
        return logits
    
    def score_sequence(self, sequence):
        with torch.no_grad():
            logits = self(sequence)
            log_probs = F.log_softmax(logits, dim=-1)
            return log_probs
