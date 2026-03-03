"""
Dataset for StreamVC self-reconstruction training.

Key design: self-reconstruction with same-speaker pairs.
  - source_audio:      used for content encoder + F0 extractor + reconstruction target
  - speaker_ref_audio: different utterance from SAME speaker → speaker encoder

This forces the speaker encoder to learn a true speaker-invariant representation
(it can't cheat by encoding content since the reference utterance is different).

At inference, swap the speaker reference to a different speaker for voice conversion.
"""

import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import librosa
import numpy as np
from torch.utils.data import Dataset


def _extract_speaker_id(path: Path) -> str:
    """
    Extract speaker ID from LibriTTS path structure.
    
    LibriTTS layout: .../train-clean-100/{speaker_id}/{chapter_id}/{speaker_id}_{chapter_id}_{utt}.wav
    Fallback: use parent.parent directory name as speaker ID.
    """
    # Try filename-based extraction: {speaker}_{chapter}_{utt}.wav
    stem = path.stem
    parts = stem.split('_')
    if len(parts) >= 2 and parts[0].isdigit():
        return parts[0]
    
    # Fallback: use grandparent directory (speaker dir)
    return path.parent.parent.name


def _group_files_by_speaker(files: List[Path]) -> Dict[str, List[Path]]:
    """Group audio files by speaker ID."""
    speaker_to_files: Dict[str, List[Path]] = defaultdict(list)
    for f in files:
        spk = _extract_speaker_id(f)
        speaker_to_files[spk].append(f)
    return dict(speaker_to_files)


def create_val_split(
    data_dirs: List[str],
    val_ratio: float = 0.05,
    seed: int = 42
) -> Tuple[List[Path], List[Path]]:
    """
    Split all discovered audio files into a deterministic train / val split.
    Split is done per-speaker to avoid speaker leakage between splits.

    Args:
        data_dirs:  Directories to search for .wav / .flac files.
        val_ratio:  Fraction of files for validation (default 5%).
        seed:       Random seed — ensures the same split on every run.

    Returns:
        train_files: List of Path objects for training.
        val_files:   List of Path objects for validation.
    """
    all_files: List[Path] = []
    for data_dir in data_dirs:
        data_path = Path(data_dir)
        if data_path.exists():
            all_files.extend(data_path.rglob("*.wav"))
            all_files.extend(data_path.rglob("*.flac"))

    # Group by speaker so we split per-speaker (no leakage)
    speaker_groups = _group_files_by_speaker(all_files)
    speakers = sorted(speaker_groups.keys())
    
    rng = random.Random(seed)
    rng.shuffle(speakers)
    
    n_val_speakers = max(1, int(len(speakers) * val_ratio))
    val_speakers = set(speakers[:n_val_speakers])
    
    train_files: List[Path] = []
    val_files: List[Path] = []
    for spk in speakers:
        if spk in val_speakers:
            val_files.extend(speaker_groups[spk])
        else:
            train_files.extend(speaker_groups[spk])
    
    train_files = sorted(train_files)
    val_files = sorted(val_files)

    print(f"Dataset split: {len(train_files)} train ({len(speakers) - n_val_speakers} speakers) | "
          f"{len(val_files)} val ({n_val_speakers} speakers)")
    return train_files, val_files


class VoiceConversionDataset(Dataset):
    """
    Dataset for StreamVC self-reconstruction training.
    
    Returns (source_audio, speaker_ref_audio) where both are from the SAME speaker
    but different utterances. The model must reconstruct source_audio given:
      - content features from source_audio
      - speaker embedding from speaker_ref_audio
      - F0 from source_audio
    """

    def __init__(
        self,
        data_dirs: Optional[List[str]] = None,
        segment_length: int = 32000,  # 2 seconds at 16kHz
        sample_rate: int = 16000,
        augment: bool = True,
        file_list: Optional[List[Path]] = None,
    ):
        super().__init__()

        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.augment = augment

        # Collect files
        if file_list is not None:
            self.audio_files = list(file_list)
        else:
            self.audio_files = []
            for data_dir in (data_dirs or []):
                data_path = Path(data_dir)
                if data_path.exists():
                    self.audio_files.extend(data_path.rglob("*.wav"))
                    self.audio_files.extend(data_path.rglob("*.flac"))
            self.audio_files = sorted(self.audio_files)

        # Build speaker → file indices mapping for same-speaker sampling
        self.speaker_to_indices: Dict[str, List[int]] = defaultdict(list)
        self.file_to_speaker: List[str] = []
        for idx, f in enumerate(self.audio_files):
            spk = _extract_speaker_id(f)
            self.speaker_to_indices[spk].append(idx)
            self.file_to_speaker.append(spk)
        
        n_multi = sum(1 for indices in self.speaker_to_indices.values() if len(indices) > 1)
        print(f"Found {len(self.audio_files)} audio files from {len(self.speaker_to_indices)} speakers "
              f"({n_multi} speakers with 2+ utterances)")
        
    def __len__(self):
        return len(self.audio_files)
    
    def load_audio(self, path: Path) -> torch.Tensor:
        """Load and preprocess audio."""
        audio, sr = librosa.load(path, sr=self.sample_rate, mono=True)
        audio = torch.from_numpy(audio).float()
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
        Get a training sample (self-reconstruction with same-speaker ref).
        
        Returns:
            source_audio: (segment_length,) — content source AND reconstruction target
            speaker_ref_audio: (segment_length,) — different utterance from SAME speaker
        """
        # Load source audio (this is both the content source and the reconstruction target)
        source_path = self.audio_files[idx]
        source_audio = self.load_audio(source_path)
        source_audio = self.random_crop(source_audio)
        source_audio = self.augment_audio(source_audio)
        
        # Select a speaker reference: different utterance from the SAME speaker
        spk = self.file_to_speaker[idx]
        same_spk_indices = self.speaker_to_indices[spk]
        
        if len(same_spk_indices) > 1:
            # Pick a different utterance from the same speaker
            ref_idx = idx
            while ref_idx == idx:
                ref_idx = random.choice(same_spk_indices)
        else:
            # Only one utterance for this speaker — use the same file
            # (different random crop provides some variation)
            ref_idx = idx
        
        ref_path = self.audio_files[ref_idx]
        speaker_ref_audio = self.load_audio(ref_path)
        speaker_ref_audio = self.random_crop(speaker_ref_audio)
        # No augmentation on speaker ref — we want clean speaker characteristics
        
        return source_audio, speaker_ref_audio


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
    print(f"Speakers: {len(dataset.speaker_to_indices)}")
    
    # Test loading
    source, speaker_ref = dataset[0]
    print(f"Source shape: {source.shape}")
    print(f"Speaker ref shape: {speaker_ref.shape}")
    
    # Verify same-speaker sampling
    spk0 = dataset.file_to_speaker[0]
    print(f"Speaker of idx 0: {spk0}, utterances: {len(dataset.speaker_to_indices[spk0])}")
    
    # Test dataloader
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    for batch_idx, (source_batch, ref_batch) in enumerate(dataloader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Source: {source_batch.shape}")
        print(f"  Speaker ref: {ref_batch.shape}")
        
        if batch_idx >= 2:
            break
    
    print("\n✓ Dataset test passed!")
