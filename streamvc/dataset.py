"""
Dataset for StreamVC training.
"""

import os
import random
from pathlib import Path
from typing import List, Tuple

import torch
import librosa
import numpy as np
from torch.utils.data import Dataset

class VoiceConversionDataset(Dataset):
    """
    Dataset for voice conversion training.
    Loads source and target audio pairs for many-to-many VC.
    """
    
    def __init__(
        self,
        data_dirs: List[str],
        segment_length: int = 32000,  # 2 seconds at 16kHz (increased for stable mel extraction)
        sample_rate: int = 16000,
        augment: bool = True
    ):
        super().__init__()
        
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.augment = augment
        
        # Collect all audio files
        self.audio_files = []
        for data_dir in data_dirs:
            data_path = Path(data_dir)
            if data_path.exists():
                files = list(data_path.rglob("*.wav")) + list(data_path.rglob("*.flac"))
                self.audio_files.extend(files)
        
        print(f"Found {len(self.audio_files)} audio files")
        
    def __len__(self):
        return len(self.audio_files)
    
    def load_audio(self, path: Path) -> torch.Tensor:
        """Load and preprocess audio."""
        # Load audio using librosa
        audio, sr = librosa.load(path, sr=self.sample_rate, mono=True)
        
        # Convert to torch tensor
        audio = torch.from_numpy(audio).float()
        
        # Normalize
        audio = audio / (audio.abs().max() + 1e-8)
        
        return audio
    
    def random_crop(self, audio: torch.Tensor) -> torch.Tensor:
        """Randomly crop audio to segment_length."""
        if audio.shape[0] >= self.segment_length:
            start = random.randint(0, audio.shape[0] - self.segment_length)
            audio = audio[start:start + self.segment_length]
        else:
            # Pad if too short
            audio = torch.nn.functional.pad(
                audio, (0, self.segment_length - audio.shape[0])
            )
        
        return audio
    
    def augment_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation."""
        if not self.augment:
            return audio
        
        # Random amplitude scaling
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            audio = audio * scale
        
        # Random noise
        if random.random() < 0.3:
            noise = torch.randn_like(audio) * 0.005
            audio = audio + noise
        
        audio = torch.clamp(audio, -1.0, 1.0)
        
        return audio
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training sample.
        
        Returns:
            source_audio: (segment_length,) source audio
            target_audio: (segment_length,) target audio (different speaker)
        """
        # Load source audio
        source_path = self.audio_files[idx]
        source_audio = self.load_audio(source_path)
        source_audio = self.random_crop(source_audio)
        source_audio = self.augment_audio(source_audio)
        
        # Select random target audio (ensure different file)
        target_idx = idx
        while target_idx == idx and len(self.audio_files) > 1:
            target_idx = random.randint(0, len(self.audio_files) - 1)
        target_path = self.audio_files[target_idx]
        target_audio = self.load_audio(target_path)
        target_audio = self.random_crop(target_audio)
        
        return source_audio, target_audio


def collate_fn(batch):
    """Collate function for dataloader."""
    source_audios, target_audios = zip(*batch)
    
    source_audios = torch.stack(source_audios)
    target_audios = torch.stack(target_audios)
    
    return source_audios, target_audios


if __name__ == "__main__":
    # Test dataset
    dataset = VoiceConversionDataset(
        data_dirs=["./data/libritts/LibriTTS/train-clean-100"],
        segment_length=16000,
        augment=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading
    source, target = dataset[0]
    print(f"Source shape: {source.shape}")
    print(f"Target shape: {target.shape}")
    
    # Test dataloader
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    for batch_idx, (source_batch, target_batch) in enumerate(dataloader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Source: {source_batch.shape}")
        print(f"  Target: {target_batch.shape}")
        
        if batch_idx >= 2:
            break
    
    print("\n✓ Dataset test passed!")
