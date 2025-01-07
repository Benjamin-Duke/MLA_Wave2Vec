from torch.utils.data import Dataset
import torch
from typing import List, Dict
import random
import os

class LibriSpeechLMDataset(Dataset):
    def __init__(self, text_path: str, vocab_path: str, max_length: int = 256, chunk_size: int = 1000000):
        # Load vocabulary
        self.word2idx = {}
        print("Loading vocabulary...")
        with open(vocab_path, 'r') as f:
            for idx, word in enumerate(f):
                self.word2idx[word.strip()] = idx
        print(f"Vocabulary size: {len(self.word2idx)}")
        
        # Get file size and calculate number of chunks
        self.text_path = text_path
        self.file_size = os.path.getsize(text_path)
        self.chunk_size = chunk_size
        self.max_length = max_length
        
        # Process first chunk to get an estimate of sequences per chunk
        print("Processing first chunk to estimate dataset size...")
        self.sequences = []
        self._process_chunk(0, chunk_size)
        sequences_per_chunk = len(self.sequences)
        estimated_total = (self.file_size // chunk_size + 1) * sequences_per_chunk
        print(f"Estimated total sequences: {estimated_total}")
        
        # Store chunk information for lazy loading
        self.num_chunks = self.file_size // chunk_size + 1
        self.sequences_per_chunk = sequences_per_chunk
        self.current_chunk = 0
        
    def _process_chunk(self, start_pos: int, size: int):
        """Process a chunk of the text file."""
        self.sequences = []
        with open(self.text_path, 'r') as f:
            f.seek(start_pos)
            current_sequence = []
            
            # Read chunk
            chunk = f.read(size)
            
            # Process complete lines
            lines = chunk.split('\n')
            for line in lines[:-1]:  # Skip last potentially incomplete line
                words = line.strip().split()
                for word in words:
                    if word in self.word2idx:
                        current_sequence.append(self.word2idx[word])
                        if len(current_sequence) >= self.max_length:
                            self.sequences.append(current_sequence[:self.max_length])
                            current_sequence = []
                if current_sequence:
                    if len(current_sequence) >= self.max_length // 2:  # Only keep sequences of reasonable length
                        self.sequences.append(current_sequence)
                    current_sequence = []
    
    def _load_chunk(self, idx: int):
        """Load the chunk containing the requested index."""
        chunk_idx = idx // self.sequences_per_chunk
        if chunk_idx != self.current_chunk:
            start_pos = chunk_idx * self.chunk_size
            self._process_chunk(start_pos, self.chunk_size)
            self.current_chunk = chunk_idx
    
    def __len__(self):
        return self.sequences_per_chunk * self.num_chunks
    
    def __getitem__(self, idx):
        # Load appropriate chunk if necessary
        self._load_chunk(idx)
        
        # Get sequence from current chunk
        chunk_local_idx = idx % self.sequences_per_chunk
        sequence = self.sequences[chunk_local_idx]
        
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