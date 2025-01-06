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
import random
import numpy as np

from wav2vec_finetuning import Wav2Vec2ForCTC
from dataLibriSpeech import LibriSpeech
from Modules.config import Wav2Vec2Config
from torch.utils.tensorboard import SummaryWriter

class SpecAugmentMask:
    """Implements masking strategy for fine-tuning as described in the paper:
    - Random number of time-step masks with span of 10 steps
    - Random number of channel masks with span of 64 channels
    - Spans may overlap
    - Time masks use the same mask embedding as pre-training
    - Channel masks use zero value
    """
    def __init__(self, mask_emb, max_time_masks=10, max_channel_masks=10):
        self.mask_emb = mask_emb  # The same mask embedding used in pre-training
        self.time_span = 10  # Fixed span of 10 time-steps as per paper
        self.channel_span = 64  # Fixed span of 64 channels as per paper
        self.max_time_masks = max_time_masks
        self.max_channel_masks = max_channel_masks

    def mask_features(self, features):
        """Apply masking to features as described in the paper
        Args:
            features: tensor of shape [batch_size, sequence_length, hidden_dim]
        """
        B, T, C = features.shape
        masked_features = features.clone()

        # For each item in batch
        for batch_idx in range(B):
            # Time masking: random number of masks
            num_time_masks = random.randint(0, self.max_time_masks)
            for _ in range(num_time_masks):
                # Choose random starting point
                start = random.randint(0, T - self.time_span)
                # Replace span with mask embedding
                masked_features[batch_idx, start:start + self.time_span, :] = self.mask_emb

            # Channel masking: random number of masks
            num_channel_masks = random.randint(0, self.max_channel_masks)
            for _ in range(num_channel_masks):
                # Choose random starting channel
                start = random.randint(0, C - self.channel_span)
                # Set span to zero
                masked_features[batch_idx, :, start:start + self.channel_span] = 0

        return masked_features

class TriStateScheduler(optim.lr_scheduler._LRScheduler):
    """Implements the tri-state learning rate schedule from the paper"""
    def __init__(self, optimizer, total_steps, warmup_pct=0.1, constant_pct=0.4, last_epoch=-1):
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_pct)
        self.constant_steps = int(total_steps * constant_pct)
        self.decay_steps = total_steps - self.warmup_steps - self.constant_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1

        if step < self.warmup_steps:  # Warmup phase
            return [base_lr * (step / self.warmup_steps) for base_lr in self.base_lrs]
        elif step < self.warmup_steps + self.constant_steps:  # Constant phase
            return self.base_lrs
        else:  # Decay phase
            decay_step = step - self.warmup_steps - self.constant_steps
            decay_factor = 1.0 - (decay_step / self.decay_steps)
            return [base_lr * decay_factor for base_lr in self.base_lrs]

class FineTuningTrainer:
    def __init__(
        self,
        model: Wav2Vec2ForCTC,
        train_dataset: LibriSpeech,
        val_dataset: LibriSpeech,
        config: Wav2Vec2Config,
        device: torch.device,
        batch_size: int = 4,  # Smaller batch size for fine-tuning
        log_dir: str = "finetuning_runs",
        checkpoint_dir: str = "finetuning_checkpoints",
        learning_rate: float = 2e-5,  # As mentioned in paper for smaller datasets
        num_training_steps: int = 50000,
        classifier_only_steps: int = 10000,  # First 10k updates for classifier only
    ):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.num_training_steps = num_training_steps
        self.classifier_only_steps = classifier_only_steps
        
        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.run_dir = os.path.join(log_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(self.run_dir)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2 if device.type == 'cuda' else 0,
            pin_memory=device.type == 'cuda'
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2 if device.type == 'cuda' else 0,
            pin_memory=device.type == 'cuda'
        )
        
        # Initialize SpecAugment masking
        self.spec_augment = SpecAugmentMask(
            mask_emb=self.model.wav2vec.mask_emb.data,  # Use the same mask embedding as pre-training
            max_time_masks=10,
            max_channel_masks=10
        )
        
        # Setup optimizer with two parameter groups
        self.classifier_params = list(self.model.char_proj.parameters())
        self.transformer_params = [p for n, p in self.model.named_parameters() 
                                 if not n.startswith('wav2vec.feature_encoder.') 
                                 and not n.startswith('char_proj.')]
        
        self.optimizer = optim.Adam([
            {'params': self.classifier_params, 'lr': learning_rate},
            {'params': self.transformer_params, 'lr': 0.0}  # Will be updated after classifier_only_steps
        ])
        
        # Setup learning rate scheduler
        self.scheduler = TriStateScheduler(
            self.optimizer,
            total_steps=num_training_steps,
            warmup_pct=0.1,
            constant_pct=0.4
        )
        
        # Initialize metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.current_step = 0
        self.best_val_loss = float('inf')
        
        # CTC loss
        self.criterion = nn.CTCLoss(blank=26, zero_infinity=True)  # blank token is last (26)
        
    def train_step(self, batch):
        audio, text = batch
        audio = audio.to(self.device)
        text = text.to(self.device)
        
        # Get log probabilities from model
        log_probs = self.model(audio)
        
        # Apply SpecAugment masking during fine-tuning
        if self.current_step >= self.classifier_only_steps:
            log_probs = self.spec_augment.mask_features(log_probs)
        
        # Prepare inputs for CTC loss
        input_lengths = torch.full((audio.shape[0],), log_probs.shape[1], device=self.device)
        target_lengths = torch.tensor([len(t) for t in text], device=self.device)
        
        # Compute CTC loss
        loss = self.criterion(
            log_probs.transpose(0, 1),  # (T, N, C) required by CTCLoss
            text,
            input_lengths,
            target_lengths
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        
        # Update parameters
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                audio, text = batch
                audio = audio.to(self.device)
                text = text.to(self.device)
                
                # Get log probabilities
                log_probs = self.model(audio)
                
                # Prepare inputs for CTC loss
                input_lengths = torch.full((audio.shape[0],), log_probs.shape[1], device=self.device)
                target_lengths = torch.tensor([len(t) for t in text], device=self.device)
                
                # Compute CTC loss
                loss = self.criterion(
                    log_probs.transpose(0, 1),
                    text,
                    input_lengths,
                    target_lengths
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self):
        """Main training loop"""
        self.model.prepare_for_fine_tuning()  # Prepare model for fine-tuning
        
        while self.current_step < self.num_training_steps:
            self.model.train()
            
            # Enable transformer training after classifier_only_steps
            if self.current_step == self.classifier_only_steps:
                print("\nEnabling transformer training...")
                self.optimizer.param_groups[1]['lr'] = self.optimizer.param_groups[0]['lr']
            
            for batch in tqdm(self.train_loader, desc=f"Step {self.current_step}"):
                loss = self.train_step(batch)
                
                # Log metrics
                if self.current_step % 100 == 0:
                    self.train_losses.append(loss)
                    self.learning_rates.append(self.scheduler.get_last_lr()[0])
                    self.writer.add_scalar('Loss/train', loss, self.current_step)
                    self.writer.add_scalar('LearningRate', self.scheduler.get_last_lr()[0], self.current_step)
                
                # Validate periodically
                if self.current_step % 1000 == 0:
                    val_loss = self.validate()
                    self.val_losses.append(val_loss)
                    self.writer.add_scalar('Loss/val', val_loss, self.current_step)
                    
                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(is_best=True)
                    
                    print(f"\nStep {self.current_step}")
                    print(f"Train loss: {loss:.4f}")
                    print(f"Val loss: {val_loss:.4f}")
                    print(f"Learning rate: {self.scheduler.get_last_lr()[0]:.6f}")
                
                self.current_step += 1
                if self.current_step >= self.num_training_steps:
                    break
            
        self.writer.close()
        
    def save_checkpoint(self, is_best=False):
        checkpoint = {
            'step': self.current_step,
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
        self.current_step = checkpoint['step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.learning_rates = checkpoint['learning_rates']
        print(f"Loaded checkpoint from step {self.current_step}") 