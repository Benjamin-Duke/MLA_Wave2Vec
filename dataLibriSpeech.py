import os
import numpy as np
import torch
import torchaudio

class LibriSpeech(torch.utils.data.Dataset):
    def __init__(self, split="test-clean", target_length=480000, device='cpu'):
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=os.path.expanduser("~/.cache"),
            url=split,
            download=True,
        )
        self.device = device
        self.target_length = target_length
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, item):
        audio, sample_rate, text, _, _, _ = self.dataset[item]
        assert sample_rate == 16000
        audio = audio.flatten().numpy()
        audio_length = len(audio)
        if audio_length < self.target_length:
            padding = np.zeros(self.target_length - audio_length)
            audio = np.concatenate((audio, padding))
        elif audio_length > self.target_length:
            audio = audio[:self.target_length]
        audio = torch.tensor(audio, dtype=torch.float32)
        audio = (audio - audio.mean()) / audio.std()

        text_indices = [ord(c) - ord('a') + 1 for c in text.lower() if 'a' <= c <= 'z']   
        text = torch.tensor(text_indices, dtype=torch.long)
        
        return audio, text
