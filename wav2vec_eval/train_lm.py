import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformer_lm import TransformerLM, TransformerLMConfig
import argparse
from tqdm import tqdm

class CharacterLMDataset(Dataset):
    def __init__(self, text_file: str, max_seq_length: int = 512):
        self.max_seq_length = max_seq_length
        self.sequences = []
        with open(text_file, 'r') as f:
            current_sequence = []
            for line in tqdm(f):
                text = line.strip().lower()
                for c in text:
                    if 'a' <= c <= 'z':
                        current_sequence.append(ord(c) - ord('a') + 1)
                    elif c == ' ':
                        current_sequence.append(0)
                    if len(current_sequence) >= max_seq_length:
                        self.sequences.append(current_sequence[:max_seq_length])
                        current_sequence = current_sequence[max_seq_length//2:]
                if current_sequence:
                    current_sequence.append(0)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        target_ids = torch.tensor(sequence[1:], dtype=torch.long)
        return {'input_ids': input_ids, 'labels': target_ids}

def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Training epoch {epoch}")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    return total_loss / len(dataloader)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = CharacterLMDataset(
        text_file=args.text_file,
        max_seq_length=args.max_seq_length
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    config = TransformerLMConfig()
    model = TransformerLM(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    best_loss = float('inf')
    for epoch in range(args.num_epochs):
        avg_loss = train_epoch(model, dataloader, optimizer, device, epoch)
        print(f"Epoch {epoch} average loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': config,
            }, args.output_path)
            print(f"Saved new best model with loss {avg_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Character-Level Language Model')
    parser.add_argument('--text_file', type=str, default='LM_data/librispeech-lm-norm.txt')
    parser.add_argument('--output_path', type=str, default='char_lm.pt')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
   
