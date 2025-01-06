import torch
from torch import nn
import os

from src.data.dataLibriSpeech import LibriSpeech
from src.models.wav2vec import Wav2Vec2
from src.config.model_config import Wav2Vec2Config
from src.utils.trainer import Trainer
from torch.utils.tensorboard import SummaryWriter


Start_from_checkpoint = True # Parametre pour lancer le dernier checkpoint

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Set batch size based on device
    batch_size = 16 if torch.cuda.is_available() else 2
    
    # Create training and validation datasets
    train_dataset = LibriSpeech(split="train-clean-100", target_length=48000)
    val_dataset = LibriSpeech(split="dev-clean", target_length=48000)
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    config = Wav2Vec2Config()
    
    model = Wav2Vec2(config)
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    checkpoint_dir = "checkpoints_runs"
    os.makedirs(checkpoint_dir, exist_ok=True)

    if Start_from_checkpoint == True:
    # Load checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            trainer.load_checkpoint(checkpoint_path)
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        device=device,
        is_librispeech=True,
        patience=100,
        log_dir="wav2vec_runs",
        batch_size=batch_size,
        checkpoint_dir=checkpoint_dir
    )
    
    trainer.train(num_epochs=50)  