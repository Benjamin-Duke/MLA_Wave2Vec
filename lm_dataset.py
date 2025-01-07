from torch.utils.data import Dataset
import torch
from typing import List, Dict
import random

class LibriSpeechLMDataset(Dataset):
    def __init__(self, text_path: str, vocab_path: str, max_length: int = 512):
        # Load vocabulary
        self.word2idx = {}
        print("Loading vocabulary...")
        with open(vocab_path, 'r') as f:
            for idx, word in enumerate(f):
                self.word2idx[word.strip()] = idx
        print(f"Vocabulary size: {len(self.word2idx)}")
        
        # Load and preprocess text
        print("Loading and preprocessing text data...")
        self.sequences = []
        with open(text_path, 'r') as f:
            current_sequence = []
            for line in f:
                words = line.strip().split()
                for word in words:
                    if word in self.word2idx:
                        current_sequence.append(self.word2idx[word])
                        if len(current_sequence) >= max_length:
                            self.sequences.append(current_sequence[:max_length])
                            current_sequence = []
                if current_sequence:
                    self.sequences.append(current_sequence)
                    current_sequence = []
        
        self.max_length = max_length
        print(f"Total sequences: {len(self.sequences)}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Pad sequence if needed
        if len(sequence) < self.max_length:
            sequence = sequence + [0] * (self.max_length - len(sequence))
        
        # Convert to tensors
        input_ids = torch.tensor(sequence, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

    @property
    def vocab_size(self):
        return len(self.word2idx) 