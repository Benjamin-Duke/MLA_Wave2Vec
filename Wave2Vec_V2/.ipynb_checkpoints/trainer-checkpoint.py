import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json

class WarmupLinearSchedule(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super(WarmupLinearSchedule, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            # Linear warmup
            return [base_lr * step / self.warmup_steps for base_lr in self.base_lrs]
        else:
            # Linear decay
            return [base_lr * (self.total_steps - step) / (self.total_steps - self.warmup_steps)
                   for base_lr in self.base_lrs]

class Trainer:
    def __init__(
        self,
        model: Wav2Vec2,
        train_dataset: LibriSpeech,
        val_dataset: LibriSpeech,
        config: Wav2Vec2Config,
        device: torch.device,
        is_librispeech: bool = True,
        patience: int = 10,
        log_dir: str = "runs",
        batch_size: int = 8,
        checkpoint_dir: str = "checkpoints",
        loader_kwargs: dict = None
    ):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.is_librispeech = is_librispeech
        self.patience = patience
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        
        # Create log directory
        self.run_dir = os.path.join(log_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Initialize metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.current_epoch = 0
        
        # Early stopping variables
        self.patience_counter = 0
        self.best_val_loss = float('inf')
        self.early_stop = False
        
        # Librispeech specific regularization
        if is_librispeech:
            self.encoder_grad_scale = 0.1
            self.l2_regularization = True
        else:
            self.encoder_grad_scale = 1.0
            self.l2_regularization = False
            
        # Use provided loader kwargs or default
        if loader_kwargs is None:
            loader_kwargs = {
                'batch_size': batch_size,
                'num_workers': 2 if device.type == 'cuda' else 0,
                'pin_memory': device.type == 'cuda',
            }
            
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            **loader_kwargs
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            **loader_kwargs
        )
        
        # Setup optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
        # Setup learning rate scheduler
        total_steps = 400000 if config.d_model == 768 else 250000  # BASE vs LARGE
        warmup_steps = int(0.08 * total_steps)  # 8% warmup
        self.scheduler = WarmupLinearSchedule(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps
        )
    
    def save_checkpoint(self, is_best=False):
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'config': self.config,
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pt")
        torch.save(checkpoint, latest_path)
        
        # Save best model if needed
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"Saved best model with validation loss: {self.best_val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.learning_rates = checkpoint['learning_rates']
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def save_metrics(self):
        metrics = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates
        }
        with open(os.path.join(self.run_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f)
            
    def plot_metrics(self):
        plt.figure(figsize=(12, 8))
        
        # Plot losses
        plt.subplot(2, 1, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training and Validation Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot learning rate
        plt.subplot(2, 1, 2)
        plt.plot(self.learning_rates, label='Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'training_metrics.png'))
        plt.close()
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Training")):
            audio = batch[0].to(self.device)
            
            # Print shapes for debugging (only first batch)
            if batch_idx == 0:
                print(f"\nInput audio shape: {audio.shape}")
            
            try:
                # Forward pass
                c, q, mask_indices = self.model(audio)
                
                # Print shapes for debugging (only first batch)
                if batch_idx == 0:
                    print(f"Context output shape: {c.shape}")
                    print(f"Quantized output shape: {q.shape}")
                    print(f"Mask indices shape: {mask_indices.shape}")
                    print(f"Number of masked positions: {mask_indices.sum().item()}")
                
                # Compute loss
                loss = self.model.compute_loss(c, q, mask_indices)
                
                if batch_idx == 0:
                    print(f"Initial loss: {loss.item():.4f}")
                
                # Add L2 regularization for Librispeech
                if self.l2_regularization:
                    l2_loss = 0.0
                    for name, param in self.model.feature_encoder.named_parameters():
                        if 'weight' in name:
                            l2_loss += torch.norm(param)
                    loss += 0.01 * l2_loss
                    
                    if batch_idx == 0:
                        print(f"L2 loss: {l2_loss.item():.4f}")
                        print(f"Total loss: {loss.item():.4f}")
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Scale gradients for feature encoder if using Librispeech
                if self.is_librispeech:
                    for param in self.model.feature_encoder.parameters():
                        param.grad *= self.encoder_grad_scale
                
                # Clip gradients
                grad_norm = clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                if batch_idx == 0:
                    print(f"Gradient norm: {grad_norm:.4f}")
                
                self.optimizer.step()
                self.scheduler.step()
                
                # Track learning rate
                self.learning_rates.append(self.scheduler.get_last_lr()[0])
                
                total_loss += loss.item()
                
                # Print batch statistics (only first few batches)
                if batch_idx < 5:
                    print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
                
            except RuntimeError as e:
                print(f"\nError in batch {batch_idx}:")
                print(f"Input shape: {audio.shape}")
                raise e
            
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation")):
                audio = batch[0].to(self.device)
                
                try:
                    # Forward pass with masking enabled (same as training)
                    c, q, mask_indices = self.model(audio, mask=True)
                    
                    # Compute loss
                    loss = self.model.compute_loss(c, q, mask_indices)
                    
                    # Add L2 regularization if using Librispeech (same as training)
                    if self.l2_regularization:
                        l2_loss = 0.0
                        for name, param in self.model.feature_encoder.named_parameters():
                            if 'weight' in name:
                                l2_loss += torch.norm(param)
                        loss += 0.01 * l2_loss
                    
                    total_loss += loss.item()
                    
                    # Print validation statistics (only first batch)
                    if batch_idx == 0:
                        print(f"\nValidation batch statistics:")
                        print(f"Loss: {loss.item():.4f}")
                        print(f"Number of masked positions: {mask_indices.sum().item()}")
                        
                except RuntimeError as e:
                    print(f"\nError in validation batch {batch_idx}:")
                    print(f"Input shape: {audio.shape}")
                    raise e
                
        return total_loss / len(self.val_loader)
        
    def train(self, num_epochs):
        for epoch in range(self.current_epoch, num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            print(f"\nEpoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Check for improvement
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(is_best)
            
            # Save and plot metrics
            self.save_metrics()
            self.plot_metrics()
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
            
            self.current_epoch = epoch + 1
