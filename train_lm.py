import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformer_lm import TransformerLM, TransformerLMConfig
from lm_dataset import LibriSpeechLMDataset
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import argparse

def train_language_model(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Enable memory efficient loading
    torch.cuda.empty_cache()
    
    # Create dataset with smaller chunks
    dataset = LibriSpeechLMDataset(
        text_path=os.path.join(args.data_dir, 'librispeech-lm-norm.txt'),
        vocab_path=os.path.join(args.data_dir, 'librispeech-vocab.txt'),
        max_length=args.max_length,
        chunk_size=args.chunk_size  # Process text in smaller chunks
    )
    
    # Split into train and validation
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders with pin_memory for faster transfer to GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    # Initialize model with correct vocabulary size
    config = TransformerLMConfig()
    config.vocab_size = dataset.vocab_size
    model = TransformerLM(config).to(device)
    
    # Enable gradient checkpointing to save memory
    model.transformer.enable_checkpoint = True
    
    # Setup optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 is padding
    
    # Create tensorboard writer
    writer = SummaryWriter(args.log_dir)
    
    # Training loop
    best_loss = float('inf')
    grad_accum_steps = args.gradient_accumulation_steps
    
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            
            # Calculate loss (shift logits and labels for next-token prediction)
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = criterion(shift_logits.view(-1, config.vocab_size), shift_labels.view(-1))
            
            # Scale loss for gradient accumulation
            loss = loss / grad_accum_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights every grad_accum_steps batches
            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * grad_accum_steps
            num_batches += 1
            progress_bar.set_postfix({'loss': loss.item() * grad_accum_steps})
            
            # Clear cache periodically
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                
                logits = model(input_ids, attention_mask)
                shift_logits = logits[:, :-1].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                loss = criterion(shift_logits.view(-1, config.vocab_size), shift_labels.view(-1))
                
                val_loss += loss.item()
                val_batches += 1
                
                # Clear cache periodically
                if val_batches % 100 == 0:
                    torch.cuda.empty_cache()
        
        # Log metrics
        avg_train_loss = total_loss / num_batches
        avg_val_loss = val_loss / val_batches
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Learning_rate', scheduler.get_last_lr()[0], epoch)
        
        print(f"Epoch {epoch+1}")
        print(f"Average train loss: {avg_train_loss:.4f}")
        print(f"Average validation loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config,
                'best_loss': best_loss
            }, os.path.join(args.checkpoint_dir, 'best_model.pt'))
        
        # Update learning rate
        scheduler.step()
        
        # Clear cache between epochs
        torch.cuda.empty_cache()
    
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Transformer Language Model')
    
    parser.add_argument('--data_dir', type=str, default='LM_data',
                        help='Directory containing LM data files')
    parser.add_argument('--batch_size', type=int, default=8,  # Reduced batch size
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--max_length', type=int, default=256,  # Reduced sequence length
                        help='Maximum sequence length')
    parser.add_argument('--chunk_size', type=int, default=1000000,  # Process text in chunks
                        help='Number of lines to process at once')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Number of steps to accumulate gradients')
    parser.add_argument('--log_dir', type=str, default='lm_runs',
                        help='Directory for tensorboard logs')
    parser.add_argument('--checkpoint_dir', type=str, default='lm_checkpoints',
                        help='Directory for saving checkpoints')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    train_language_model(args) 