from dataclasses import dataclass, field

@dataclass
class Wav2Vec2Config:
    #Config BASE model
    
    # Feature encoder
    conv_layers: list = field(default_factory=lambda: [(512, 10, 5)] + [(512, 3, 2)] * 5 + [(512, 2, 2)])
    dropout: float = 0.1
    layer_drop: float = 0.05
    
    # Transformer
    d_model: int = 768
    nhead: int = 8
    num_encoder_layers: int = 12
    dim_feedforward: int = 3072
    
    # Quantizer
    num_codebooks: int = 2
    codebook_size: int = 320
    temp: float = 2.0
    min_temp: float = 0.5
    temp_decay: float = 0.999995
    
    # Masking
    mask_prob: float = 0.5
    mask_length: int = 10
    
    # Training
    learning_rate: float = 5e-4
    warmup_steps_pct: float = 0.08  # 8% warmup
    num_updates: int = 400_000  # BASE model updates
    l2_weight: float = 0.1  # L2 penalty for encoder activations
    encoder_grad_scale: float = 0.1  # Scale down encoder gradients
    contrastive_temperature: float = 0.1  # κ in the paper
    diversity_weight: float = 0.1  # α in the paper
    num_negatives: int = 100  # K distractors
    