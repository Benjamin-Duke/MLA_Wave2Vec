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
        
        return audio, text

if __name__ == "__main__":
    dataset = LibriSpeech(split="test-clean", target_length=480000, device='cpu')
    audio, text = dataset[8]
    
    mean = audio.mean().item()
    std = audio.std().item()
    print(f"Mean: {mean}, Std Dev: {std}")
