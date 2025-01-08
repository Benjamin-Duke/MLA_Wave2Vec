from torch.utils.data import Dataset
import torch
from typing import List, Dict
import os

class LibriSpeechLMDataset(Dataset):
    def __init__(self, text_path: str, vocab_path: str, max_length: int = 256, chunk_size: int = 1000000):
        self.word2idx = {}
        with open(vocab_path, 'r') as f:
            for idx, word in enumerate(f):
                self.word2idx[word.strip()] = idx
        self.text_path = text_path
        self.file_size = os.path.getsize(text_path)
        self.chunk_size = chunk_size
        self.max_length = max_length
        self.sequences = []
        self._process_chunk(0, chunk_size)
        self.sequences_per_chunk = len(self.sequences)
        with open(text_path, 'r') as f:
            self.total_lines = sum(1 for _ in f)
        sequences_per_line = self.sequences_per_chunk / (chunk_size / 100)
        self.total_sequences = int(self.total_lines * sequences_per_line)
        self.current_chunk = 0
        self.current_chunk_start = 0
    
    def _process_chunk(self, start_pos: int, size: int):
        self.sequences = []
        with open(self.text_path, 'r') as f:
            f.seek(start_pos)
            if start_pos > 0:
                f.readline()
            current_sequence = []
            bytes_read = 0
            while bytes_read < size:
                line = f.readline()
                if not line:
                    break
                bytes_read += len(line.encode('utf-8'))
                words = line.strip().split()
                for word in words:
                    if word in self.word2idx:
                        current_sequence.append(self.word2idx[word])
                        if len(current_sequence) >= self.max_length:
                            self.sequences.append(current_sequence[:self.max_length])
                            current_sequence = []
                if current_sequence:
                    if len(current_sequence) >= self.max_length // 2:
                        self.sequences.append(current_sequence)
                    current_sequence = []
            if not self.sequences and not line:
                f.seek(0)
                self._process_chunk(0, size)
                return
            elif not self.sequences:
                while not self.sequences and line:
                    line = f.readline()
                    if not line:
                        break
                    words = line.strip().split()
                    for word in words:
                        if word in self.word2idx:
                            current_sequence.append(self.word2idx[word])
                            if len(current_sequence) >= self.max_length:
                                self.sequences.append(current_sequence[:self.max_length])
                                current_sequence = []
                    if current_sequence and len(current_sequence) >= self.max_length // 2:
                        self.sequences.append(current_sequence)
                        current_sequence = []
            if not self.sequences:
                raise RuntimeError(f"Could not find any valid sequences in chunk starting at position {start_pos}")
            self.current_chunk_start = start_pos
    
    def _load_chunk(self, idx: int):
        target_chunk = (idx * self.chunk_size) // self.total_sequences
        if target_chunk != self.current_chunk or not self.sequences:
            start_pos = target_chunk * self.chunk_size
            self._process_chunk(start_pos, self.chunk_size)
            self.current_chunk = target_chunk
    
    def __len__(self):
        return self.total_sequences
    
    def __getitem__(self, idx):
        if idx >= self.total_sequences:
            raise IndexError("Index out of range")
        self._load_chunk(idx)
        if not self.sequences:
            raise RuntimeError("No sequences available after loading chunk")
        chunk_local_idx = idx % len(self.sequences)
        sequence = self.sequences[chunk_local_idx]
        if len(sequence) < self.max_length:
            sequence = sequence + [0] * (self.max_length - len(sequence))
        sequence = sequence[:self.max_length]
        input_ids = torch.tensor(sequence, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

    @property
    def vocab_size(self):
        return len(self.word2idx)
