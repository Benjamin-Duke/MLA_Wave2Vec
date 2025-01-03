import torch
from torch import nn
import os

from dataLibriSpeech import LibriSpeech
from wave2vec import Wav2Vec2
from Modules.config import Wav2Vec2Config
from trainer import Trainer

import torch
from torch.utils.tensorboard import SummaryWriter



# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Set batch size based on device
batch_size = 32 if torch.cuda.is_available() else 2

# Create training and validation datasets
train_dataset = LibriSpeech(split="train-clean-100", target_length=48000)
val_dataset = LibriSpeech(split="dev-clean", target_length=48000)

print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

# Use original BASE configuration
config = Wav2Vec2Config()

# Create model and move to device
model = Wav2Vec2(config)
model = model.to(device)

# Enable multi-GPU if available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# Create checkpoint directory
checkpoint_dir = "checkpoints_test_1"
os.makedirs(checkpoint_dir, exist_ok=True)

# Load checkpoint if it exists
# checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
# if os.path.exists(checkpoint_path):
#     print(f"Loading checkpoint from {checkpoint_path}")
#     trainer.load_checkpoint(checkpoint_path)

# Create trainer
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    config=config,
    device=device,
    is_librispeech=True,
    patience=10,
    log_dir="wav2vec_runs",
    batch_size=batch_size,
    checkpoint_dir=checkpoint_dir
)

# Start training
trainer.train(num_epochs=25)  # Will stop early if no improvement