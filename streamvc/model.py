"""
StreamVC Model Implementation
Following the StreamVC paper architecture with causal streaming design.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CausalConv1d(nn.Module):
    """Causal 1D convolution for streaming (no future context)."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 dilation: int = 1, groups: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation, groups=groups
        )
        
    def forward(self, x):
        x = self.conv(x)
        # Remove future context
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x


class ResidualBlock(nn.Module):
    """Residual block with causal convolutions."""
    
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        
    def forward(self, x):
        residual = x
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.gelu(x + residual)


class ContentEncoder(nn.Module):
    """
    Content encoder that predicts HuBERT soft labels.
    Operates at 50 Hz (20ms frames) to produce discrete content units.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 8,
        kernel_size: int = 5,
        num_classes: int = 100,
        dropout: float = 0.1,
        hop_length: int = 320  # Downsample from 16kHz to 50Hz (320 samples = 20ms)
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.hop_length = hop_length
        
        # Input projection
        self.input_proj = CausalConv1d(in_channels, hidden_dim, kernel_size=7)
        
        # Residual blocks with increasing dilation
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, kernel_size, dilation=2**i)
            for i in range(num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Temporal downsampling to 50Hz (strided conv)
        self.downsample = nn.Conv1d(
            hidden_dim, hidden_dim,
            kernel_size=self.hop_length * 2,
            stride=self.hop_length,
            padding=self.hop_length // 2
        )
        
        # Output projection to num_classes (soft labels)
        self.output_proj = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input audio (B, 1, T) where T is audio samples
            
        Returns:
            logits: (B, num_classes, T') - soft label predictions at 50Hz
            features: (B, hidden_dim, T') - content features at 50Hz
        """
        # Input projection
        x = self.input_proj(x)
        
        # Apply residual blocks
        for block in self.blocks:
            x = block(x)
            x = self.dropout(x)
        
        # Downsample to 50Hz
        x = self.downsample(x)
        
        # Content features
        features = x
        
        # Soft label logits
        logits = self.output_proj(x)
        
        return logits, features


class LearnableAttentionPooling(nn.Module):
    """Learnable attention pooling to aggregate frame-level speaker features."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        
        self.query = nn.Parameter(torch.randn(1, output_dim))
        self.key_proj = nn.Linear(input_dim, output_dim)
        self.value_proj = nn.Linear(input_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D, T) frame-level features
            
        Returns:
            pooled: (B, output_dim) global speaker embedding
        """
        B, D, T = x.shape
        
        # Transpose to (B, T, D)
        x = x.transpose(1, 2)
        
        # Project to keys and values
        keys = self.key_proj(x)  # (B, T, output_dim)
        values = self.value_proj(x)  # (B, T, output_dim)
        
        # Compute attention scores
        query = self.query.expand(B, -1, -1)  # (B, 1, output_dim)
        scores = torch.bmm(query, keys.transpose(1, 2))  # (B, 1, T)
        attention_weights = F.softmax(scores / (self.query.size(-1) ** 0.5), dim=-1)
        
        # Apply attention
        pooled = torch.bmm(attention_weights, values).squeeze(1)  # (B, output_dim)
        
        return pooled


class SpeakerEncoder(nn.Module):
    """
    Speaker encoder with learnable attention pooling.
    Produces a global speaker embedding from frame-level features.
    """
    
    def __init__(
        self,
        in_channels: int = 80,  # Mel spectrogram
        hidden_dim: int = 256,
        num_layers: int = 4,
        embedding_dim: int = 256,
        kernel_size: int = 5
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=7, padding=3)
        
        # Residual blocks (non-causal for speaker encoding)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2),
                nn.GroupNorm(8, hidden_dim),
                nn.GELU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2),
                nn.GroupNorm(8, hidden_dim),
            )
            for _ in range(num_layers)
        ])
        
        # Learnable attention pooling
        self.pooling = LearnableAttentionPooling(hidden_dim, embedding_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Mel spectrogram (B, n_mels, T)
            
        Returns:
            embedding: (B, embedding_dim) global speaker embedding
        """
        x = self.input_proj(x)
        
        for block in self.blocks:
            residual = x
            x = block(x)
            x = F.gelu(x + residual)
        
        # Pool to global embedding
        embedding = self.pooling(x)
        
        return embedding


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation for speaker conditioning."""
    
    def __init__(self, num_features: int, cond_dim: int):
        super().__init__()
        
        self.scale_transform = nn.Linear(cond_dim, num_features)
        self.shift_transform = nn.Linear(cond_dim, num_features)
        
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) features
            cond: (B, cond_dim) conditioning vector
            
        Returns:
            modulated: (B, C, T)
        """
        scale = self.scale_transform(cond).unsqueeze(-1)  # (B, C, 1)
        shift = self.shift_transform(cond).unsqueeze(-1)  # (B, C, 1)
        
        return x * (1 + scale) + shift


class StreamDecoder(nn.Module):
    """
    Streaming decoder with FiLM conditioning.
    Generates audio from content units + speaker embedding + f0 features.
    """
    
    def __init__(
        self,
        in_channels: int = 64,  # From content encoder
        hidden_dim: int = 40,
        num_layers: int = 12,
        kernel_size: int = 5,
        output_channels: int = 1,
        speaker_dim: int = 256,
        f0_dim: int = 9,
        lookahead_frames: int = 2,
        hop_length: int = 320  # Upsample from 50Hz to 16kHz
    ):
        super().__init__()
        
        self.lookahead_frames = lookahead_frames
        self.hop_length = hop_length
        
        # Input projection (content + f0 features)
        self.input_proj = CausalConv1d(
            in_channels + f0_dim, hidden_dim, kernel_size=7
        )
        
        # Residual blocks with FiLM conditioning
        self.blocks = nn.ModuleList()
        self.film_layers = nn.ModuleList()
        
        for i in range(num_layers):
            self.blocks.append(
                ResidualBlock(hidden_dim, kernel_size, dilation=2**(i % 4))
            )
            self.film_layers.append(FiLMLayer(hidden_dim, speaker_dim))
        
        # Temporal upsampling back to audio rate (transposed conv)
        self.upsample = nn.ConvTranspose1d(
            hidden_dim, hidden_dim,
            kernel_size=self.hop_length * 2,
            stride=self.hop_length,
            padding=self.hop_length // 2
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, output_channels, kernel_size=1),
            nn.Tanh()
        )
        
    def forward(
        self,
        content_features: torch.Tensor,
        speaker_embedding: torch.Tensor,
        f0_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            content_features: (B, in_channels, T)
            speaker_embedding: (B, speaker_dim)
            f0_features: (B, f0_dim, T)
            
        Returns:
            audio: (B, 1, T_audio) reconstructed waveform
        """
        # Align temporal dimensions
        min_len = min(content_features.size(2), f0_features.size(2))
        content_features = content_features[:, :, :min_len]
        f0_features = f0_features[:, :, :min_len]
        
        # Concatenate content and f0
        x = torch.cat([content_features, f0_features], dim=1)
        
        # Input projection
        x = self.input_proj(x)
        
        # Apply residual blocks with FiLM conditioning
        for block, film in zip(self.blocks, self.film_layers):
            x = block(x)
            x = film(x, speaker_embedding)
        
        # Upsample to audio rate
        x = self.upsample(x)
        
        # Output projection
        audio = self.output_proj(x)
        
        return audio


class StreamVC(nn.Module):
    """
    Complete StreamVC model for streaming voice conversion.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        # Extract config
        content_cfg = config['model']['content_encoder']
        speaker_cfg = config['model']['speaker_encoder']
        decoder_cfg = config['model']['decoder']
        
        # Build components
        self.content_encoder = ContentEncoder(
            in_channels=content_cfg['in_channels'],
            hidden_dim=content_cfg['hidden_dim'],
            num_layers=content_cfg['num_layers'],
            kernel_size=content_cfg['kernel_size'],
            num_classes=content_cfg['num_classes'],
            dropout=content_cfg['dropout']
        )
        
        self.speaker_encoder = SpeakerEncoder(
            in_channels=speaker_cfg['in_channels'],
            hidden_dim=speaker_cfg['hidden_dim'],
            num_layers=speaker_cfg['num_layers'],
            embedding_dim=speaker_cfg['embedding_dim'],
            kernel_size=5
        )
        
        self.decoder = StreamDecoder(
            in_channels=decoder_cfg['in_channels'],
            hidden_dim=decoder_cfg['hidden_dim'],
            num_layers=decoder_cfg['num_layers'],
            kernel_size=decoder_cfg['kernel_size'],
            output_channels=decoder_cfg['output_channels'],
            speaker_dim=speaker_cfg['embedding_dim'],
            f0_dim=config['model']['f0_extractor']['uncertainty_features'],
            lookahead_frames=decoder_cfg['lookahead_frames']
        )
        
    def forward(
        self,
        source_audio: torch.Tensor,
        target_mel: torch.Tensor,
        f0_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            source_audio: (B, 1, T) source waveform
            target_mel: (B, n_mels, T_mel) target speaker mel
            f0_features: (B, 9, T_f0) f0 + uncertainty features
            
        Returns:
            output_audio: (B, 1, T_out) converted waveform
            content_logits: (B, num_classes, T_content) for content loss
        """
        # Extract content (with stop gradient to decoder)
        content_logits, content_features = self.content_encoder(source_audio)
        
        # Stop gradient from decoder to content encoder
        content_features_stopped = content_features.detach()
        
        # Extract speaker embedding
        speaker_embedding = self.speaker_encoder(target_mel)
        
        # Decode
        output_audio = self.decoder(
            content_features_stopped,
            speaker_embedding,
            f0_features
        )
        
        return output_audio, content_logits


if __name__ == "__main__":
    # Test model
    import yaml
    
    with open("../configs/streamvc_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    model = StreamVC(config)
    
    # Test shapes
    B, T = 2, 16000  # 1 second at 16kHz
    source_audio = torch.randn(B, 1, T)
    target_mel = torch.randn(B, 80, T // 320)  # 50 Hz
    f0_features = torch.randn(B, 9, T // 320)
    
    output_audio, content_logits = model(source_audio, target_mel, f0_features)
    
    print(f"Input audio: {source_audio.shape}")
    print(f"Target mel: {target_mel.shape}")
    print(f"F0 features: {f0_features.shape}")
    print(f"Output audio: {output_audio.shape}")
    print(f"Content logits: {content_logits.shape}")
    print("✓ Model test passed!")
