import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class TransformerLMConfig:
    def __init__(self):
        self.n_layers = 20
        self.d_model = 1280
        self.d_ff = 6144
        self.n_heads = 16
        self.dropout = 0.1
        self.vocab_size = 32000  # Will be set based on your vocabulary
        self.max_seq_length = 2048

class TransformerLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, config.max_seq_length, config.d_model))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.n_layers)
        
        # Output layer
        self.output_layer = nn.Linear(config.d_model, config.vocab_size)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, input_ids, attention_mask=None):
        # Get embeddings
        x = self.token_embedding(input_ids)
        
        # Add positional embeddings
        position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        x = x + self.position_embedding[:, :input_ids.size(1), :]
        
        x = self.dropout(x)
        
        # Create attention mask if needed
        if attention_mask is not None:
            # Convert from [batch_size, seq_len] to [batch_size, seq_len]
            # where 1 means "not masked" and 0 means "masked"
            attention_mask = attention_mask.bool()
            attention_mask = ~attention_mask  # Invert because transformer expects 1 for masked positions
        
        # Pass through transformer
        x = self.transformer(x, src_key_padding_mask=attention_mask)
        
        # Get logits
        logits = self.output_layer(x)
        
        return logits

    def score_sequence(self, sequence):
        """Calculate language model scores for the sequence.
        
        Args:
            sequence: Tensor of shape (batch_size, sequence_length) containing token IDs
            
        Returns:
            Tensor of shape (batch_size, sequence_length, vocab_size) containing log probabilities
        """
        with torch.no_grad():
            # Get logits from forward pass
            logits = self(sequence)
            # Convert to log probabilities
            log_probs = F.log_softmax(logits, dim=-1)
            return log_probs 