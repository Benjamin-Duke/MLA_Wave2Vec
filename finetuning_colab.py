import torch
from torch import nn
import os
from google.colab import drive
import matplotlib.pyplot as plt

from dataLibriSpeech import LibriSpeech
from wav2vec_finetuning import Wav2Vec2ForCTC
from Modules.config import Wav2Vec2Config
from finetuning_trainer import FineTuningTrainer

# Mount Google Drive
drive.mount('/content/drive')

# Configuration class for easy parameter management
class FineTuningConfig:
    def __init__(self):
        # Training parameters
        self.batch_size = 4
        self.learning_rate = 2e-5
        self.num_steps = 50000
        self.classifier_steps = 10000
        
        # Paths
        self.pretrained_path = '/content/drive/MyDrive/wav2vec/checkpoints/best_model.pt'  # Update this
        self.log_dir = '/content/drive/MyDrive/wav2vec/finetuning_runs'
        self.checkpoint_dir = '/content/drive/MyDrive/wav2vec/finetuning_checkpoints'
        self.resume_from = None  # Set this if resuming from a checkpoint
        
        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

def setup_fine_tuning(config):
    """Setup the fine-tuning process"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = LibriSpeech(split="train-clean-100", target_length=48000)
    val_dataset = LibriSpeech(split="dev-clean", target_length=48000)
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Load configuration
    model_config = Wav2Vec2Config()
    
    # Create fine-tuning model
    print("\nInitializing model...")
    model = Wav2Vec2ForCTC(model_config)
    
    # Load pre-trained weights
    print(f"\nLoading pre-trained weights from {config.pretrained_path}")
    checkpoint = torch.load(config.pretrained_path, map_location=device)
    # Load only the wav2vec part, not the classifier
    model.wav2vec.load_state_dict(checkpoint['model_state_dict'])
    
    # Enable multi-GPU if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Create trainer
    trainer = FineTuningTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=model_config,
        device=device,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        num_training_steps=config.num_steps,
        classifier_only_steps=config.classifier_steps,
        log_dir=config.log_dir,
        checkpoint_dir=config.checkpoint_dir
    )
    
    # Load checkpoint if continuing training
    if config.resume_from:
        print(f"\nResuming from checkpoint: {config.resume_from}")
        trainer.load_checkpoint(config.resume_from)
    
    return trainer

def plot_training_progress(trainer):
    """Plot training metrics"""
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(trainer.train_losses, label='Train Loss')
    plt.plot(trainer.val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Steps (x100)')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot learning rate
    plt.subplot(1, 2, 2)
    plt.plot(trainer.learning_rates, label='Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Steps (x100)')
    plt.ylabel('Learning Rate')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Example usage in notebook:
"""
# Initialize configuration
config = FineTuningConfig()

# Optionally modify config parameters
config.batch_size = 8  # if you want to change batch size
config.learning_rate = 3e-5  # if you want to change learning rate

# Setup training
trainer = setup_fine_tuning(config)

# Start training
trainer.train()

# Plot results
plot_training_progress(trainer)
""" 