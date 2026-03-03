"""
Audio utilities for mel spectrogram extraction and processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T


class MelSpectrogramExtractor(nn.Module):
    """Extract mel spectrogram from audio."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 320,  # 20ms at 16kHz
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float = 8000.0
    ):
        super().__init__()
        
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0
        )
        
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract mel spectrogram.
        
        Args:
            audio: (B, T) or (B, 1, T) waveform
            
        Returns:
            mel: (B, n_mels, T_mel) mel spectrogram
        """
        if audio.ndim == 3:
            audio = audio.squeeze(1)
        
        # Extract mel spectrogram
        mel = self.mel_transform(audio)
        
        # Log scale
        mel = torch.log(mel + 1e-8)
        
        return mel


def load_audio(
    path: str,
    sample_rate: int = 16000,
    normalize: bool = True
) -> torch.Tensor:
    """
    Load audio file.
    
    Args:
        path: Audio file path
        sample_rate: Target sample rate
        normalize: If True, normalize to [-1, 1]
        
    Returns:
        audio: (T,) audio waveform
    """
    audio, sr = torchaudio.load(path)
    
    # Convert to mono
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    
    # Resample if needed
    if sr != sample_rate:
        resampler = T.Resample(sr, sample_rate)
        audio = resampler(audio)
    
    audio = audio.squeeze(0)
    
    # Normalize
    if normalize:
        audio = audio / (audio.abs().max() + 1e-8)
    
    return audio


def save_audio(
    audio: torch.Tensor,
    path: str,
    sample_rate: int = 16000
):
    """
    Save audio to file.
    
    Args:
        audio: (T,) or (1, T) waveform
        path: Output path
        sample_rate: Sample rate
    """
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    
    torchaudio.save(path, audio.cpu(), sample_rate)


class STFTLoss(nn.Module):
    """Multi-scale STFT loss for training."""
    
    def __init__(
        self,
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[256, 512, 128],
        win_lengths=[1024, 2048, 512]
    ):
        super().__init__()
        
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        
    def stft(self, x, fft_size, hop_size, win_length):
        """Compute STFT."""
        window = torch.hann_window(win_length).to(x.device)
        
        stft = torch.stft(
            x,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_length,
            window=window,
            return_complex=True
        )
        
        return stft
    
    def forward(self, y_pred, y_true):
        """
        Compute multi-scale STFT loss.
        
        Args:
            y_pred: (B, T) predicted audio
            y_true: (B, T) target audio
            
        Returns:
            loss: Scalar STFT loss
        """
        if y_pred.ndim == 3:
            y_pred = y_pred.squeeze(1)
        if y_true.ndim == 3:
            y_true = y_true.squeeze(1)
        
        loss = 0.0
        
        for fft_size, hop_size, win_length in zip(
            self.fft_sizes, self.hop_sizes, self.win_lengths
        ):
            # Compute STFTs
            stft_pred = self.stft(y_pred, fft_size, hop_size, win_length)
            stft_true = self.stft(y_true, fft_size, hop_size, win_length)
            
            # Magnitude
            mag_pred = torch.abs(stft_pred)
            mag_true = torch.abs(stft_true)
            
            # L1 loss on magnitude
            mag_loss = F.l1_loss(mag_pred, mag_true)
            
            # Log magnitude loss
            log_mag_loss = F.l1_loss(
                torch.log(mag_pred + 1e-5),
                torch.log(mag_true + 1e-5)
            )
            
            loss += mag_loss + log_mag_loss
        
        return loss / len(self.fft_sizes)


if __name__ == "__main__":
    import torch.nn.functional as F
    
    # Test mel extraction
    extractor = MelSpectrogramExtractor(sample_rate=16000)
    
    audio = torch.randn(2, 16000)  # 1 second
    mel = extractor(audio)
    
    print(f"Audio shape: {audio.shape}")
    print(f"Mel shape: {mel.shape}")
    
    # Test STFT loss
    stft_loss = STFTLoss()
    
    y_pred = torch.randn(2, 16000)
    y_true = torch.randn(2, 16000)
    
    loss = stft_loss(y_pred, y_true)
    print(f"STFT loss: {loss.item():.4f}")
    
    print("✓ Audio utilities test passed!")
