import torch
from torch.utils.data import Dataset, DataLoader
from torchaudio.datasets import LIBRISPEECH
import pytorch_lightning as pl


class StandardScalerTransform:
    """A transform that standardizes each waveform to have mean=0 and std=1."""
    def __call__(self, waveform):
        mean = waveform.mean()
        std = waveform.std()
        return (waveform - mean) / std


class LargeDataset(Dataset):
    """Large Dataset will download the data from LIBRISPEECH if not already present on dataset_path."""
    def __init__(self, dataset_path, transform=None):
        self.dataset = LIBRISPEECH(dataset_path, url="train-clean-100", download=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform, sample_rate, label, *_ = self.dataset[idx]
        if self.transform:
            waveform = self.transform(waveform)
        return waveform, label


class LargeDataModule(pl.LightningDataModule):
    """Large Data Module will load a portion of the data on the VRAM."""
    def __init__(self, dataset_path, batch_size, num_workers, transform=None):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.transform = transform
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.dataset = LargeDataset(self.dataset_path, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

'''
# Example Usage
dataset_path = "/path/to/dataset"  # The path to the disk where the data is stored
batch_size = 16
num_workers =  x  # Adjust based on your system

# Use the StandardScalerTransform for the transform
transform = StandardScalerTransform()

data_module = LargeDataModule(dataset_path, batch_size, num_workers, transform=transform)
trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=10)

# Define your model before running trainer.fit
# trainer.fit(model, datamodule=data_module)'''
