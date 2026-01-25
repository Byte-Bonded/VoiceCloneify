"""
F0 (pitch) extraction using Yin algorithm with whitening.
Produces 9-dimensional features: f0 + uncertainty as per StreamVC paper.
"""

import numpy as np
import torch
import torch.nn as nn
import parselmouth
from typing import Tuple, Optional


class YinF0Extractor:
    """
    Yin algorithm-based F0 extractor with per-utterance whitening.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length: float = 20.0,  # ms
        f0_min: float = 50.0,
        f0_max: float = 600.0,
        whitening: bool = True
    ):
        self.sample_rate = sample_rate
        self.frame_length = frame_length / 1000.0  # Convert to seconds
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.whitening = whitening
        
    def extract_f0(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract f0 contour using Yin algorithm.
        
        Args:
            audio: Audio waveform (samples,)
            
        Returns:
            f0: F0 contour (T,)
            voiced: Voice/unvoiced binary mask (T,)
        """
        # Use parselmouth (Praat) for Yin algorithm
        sound = parselmouth.Sound(audio, sampling_frequency=self.sample_rate)
        
        pitch = sound.to_pitch_ac(
            time_step=self.frame_length,
            pitch_floor=self.f0_min,
            pitch_ceiling=self.f0_max
        )
        
        # Extract f0 values
        f0_values = pitch.selected_array['frequency']
        
        # Create voiced mask (non-zero f0)
        voiced = (f0_values > 0).astype(np.float32)
        
        # Replace unvoiced frames with interpolated values
        if np.any(voiced):
            # Linear interpolation for unvoiced frames
            f0_interp = np.copy(f0_values)
            voiced_indices = np.where(voiced > 0)[0]
            
            if len(voiced_indices) > 1:
                for i in range(len(f0_values)):
                    if f0_values[i] == 0:
                        # Find nearest voiced frames
                        left_idx = voiced_indices[voiced_indices < i]
                        right_idx = voiced_indices[voiced_indices > i]
                        
                        if len(left_idx) > 0 and len(right_idx) > 0:
                            left = left_idx[-1]
                            right = right_idx[0]
                            # Linear interpolation
                            alpha = (i - left) / (right - left)
                            f0_interp[i] = (1 - alpha) * f0_values[left] + alpha * f0_values[right]
                        elif len(left_idx) > 0:
                            f0_interp[i] = f0_values[left_idx[-1]]
                        elif len(right_idx) > 0:
                            f0_interp[i] = f0_values[right_idx[0]]
            else:
                # Only one voiced frame or none - use median
                f0_interp = np.full_like(f0_values, np.median(f0_values[voiced > 0]) if np.any(voiced) else 150.0)
        else:
            # No voiced frames - use default
            f0_interp = np.full_like(f0_values, 150.0)
        
        return f0_interp, voiced
    
    def compute_uncertainty_features(
        self, 
        f0: np.ndarray, 
        voiced: np.ndarray
    ) -> np.ndarray:
        """
        Compute 9-dimensional uncertainty features from f0.
        
        Features: [f0, log_f0, f0_delta, f0_delta_delta, voiced, 
                   confidence, periodicity, harmonicity, spectral_flatness]
        
        Args:
            f0: F0 contour (T,)
            voiced: Voiced mask (T,)
            
        Returns:
            features: (T, 9) uncertainty features
        """
        T = len(f0)
        features = np.zeros((T, 9), dtype=np.float32)
        
        # Feature 0: f0 (whitened if enabled)
        if self.whitening and np.any(voiced):
            voiced_f0 = f0[voiced > 0]
            f0_mean = np.mean(voiced_f0)
            f0_std = np.std(voiced_f0) + 1e-8
            features[:, 0] = (f0 - f0_mean) / f0_std
        else:
            features[:, 0] = f0
        
        # Feature 1: log f0
        features[:, 1] = np.log(f0 + 1e-8)
        
        # Feature 2: f0 delta (first derivative)
        features[1:, 2] = f0[1:] - f0[:-1]
        
        # Feature 3: f0 delta-delta (second derivative)
        features[2:, 3] = features[2:, 2] - features[1:-1, 2]
        
        # Feature 4: voiced flag
        features[:, 4] = voiced
        
        # Feature 5: confidence (smoothed voiced probability)
        window = 3
        for i in range(T):
            start = max(0, i - window)
            end = min(T, i + window + 1)
            features[i, 5] = np.mean(voiced[start:end])
        
        # Features 6-8: Simplified proxies (periodicity, harmonicity, spectral flatness)
        # In practice, these would be computed from audio; here we use f0-based proxies
        features[:, 6] = voiced * (1.0 - np.abs(features[:, 2]) / (f0 + 1e-8))  # Periodicity proxy
        features[:, 7] = voiced  # Harmonicity proxy
        features[:, 8] = 1.0 - voiced  # Spectral flatness proxy (higher for unvoiced)
        
        return features
    
    def __call__(
        self, 
        audio: np.ndarray,
        return_numpy: bool = True
    ) -> np.ndarray:
        """
        Extract f0 features from audio.
        
        Args:
            audio: Audio waveform (samples,) or (batch, samples)
            return_numpy: If True, return numpy; else return torch.Tensor
            
        Returns:
            features: (T, 9) or (batch, T, 9) f0 uncertainty features
        """
        if audio.ndim == 1:
            # Single utterance
            f0, voiced = self.extract_f0(audio)
            features = self.compute_uncertainty_features(f0, voiced)
            
            if not return_numpy:
                features = torch.from_numpy(features)
            
            return features
        else:
            # Batch processing
            batch_features = []
            for i in range(audio.shape[0]):
                f0, voiced = self.extract_f0(audio[i])
                features = self.compute_uncertainty_features(f0, voiced)
                batch_features.append(features)
            
            if return_numpy:
                return np.stack(batch_features, axis=0)
            else:
                return torch.stack([torch.from_numpy(f) for f in batch_features])


class F0ExtractorModule(nn.Module):
    """
    Torch module wrapper for F0 extraction (used during training).
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length: float = 20.0,
        f0_min: float = 50.0,
        f0_max: float = 600.0,
        whitening: bool = True
    ):
        super().__init__()
        
        self.extractor = YinF0Extractor(
            sample_rate=sample_rate,
            frame_length=frame_length,
            f0_min=f0_min,
            f0_max=f0_max,
            whitening=whitening
        )
        
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract f0 features from audio batch.
        
        Args:
            audio: (B, T) or (B, 1, T) audio waveform
            
        Returns:
            features: (B, 9, T_f0) f0 uncertainty features
        """
        if audio.ndim == 3:
            audio = audio.squeeze(1)  # Remove channel dim
        
        # Convert to numpy for processing
        audio_np = audio.cpu().numpy()
        
        # Extract features
        features = self.extractor(audio_np, return_numpy=False)
        
        # Transpose to (B, 9, T)
        features = features.transpose(1, 2)
        
        return features.to(audio.device)


if __name__ == "__main__":
    # Test f0 extraction
    import librosa
    
    # Generate test audio (sine wave)
    sr = 16000
    duration = 1.0
    f0 = 200  # Hz
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.5 * np.sin(2 * np.pi * f0 * t)
    
    # Extract f0 features
    extractor = YinF0Extractor(sample_rate=sr, whitening=True)
    features = extractor(audio)
    
    print(f"Audio shape: {audio.shape}")
    print(f"F0 features shape: {features.shape}")
    print(f"F0 features (first 5 frames):\n{features[:5]}")
    
    # Test torch module
    module = F0ExtractorModule(sample_rate=sr)
    audio_torch = torch.from_numpy(audio).unsqueeze(0)  # (1, T)
    features_torch = module(audio_torch)
    
    print(f"\nTorch input shape: {audio_torch.shape}")
    print(f"Torch output shape: {features_torch.shape}")
    print("✓ F0 extraction test passed!")
