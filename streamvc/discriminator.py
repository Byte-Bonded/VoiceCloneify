"""
Multi-scale discriminator for adversarial training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DiscriminatorBlock(nn.Module):
    """Discriminator block with strided convolutions."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 15,
        stride: int = 1,
        groups: int = 1
    ):
        super().__init__()
        
        padding = (kernel_size - 1) // 2
        
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups
        )
        self.norm = nn.GroupNorm(min(32, out_channels // 4), out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = F.leaky_relu(x, 0.2)
        return x


class ScaleDiscriminator(nn.Module):
    """Single-scale discriminator."""
    
    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 4
    ):
        super().__init__()
        
        channels = [1] + [hidden_dim * (2 ** i) for i in range(num_layers)]
        
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(
                DiscriminatorBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    stride=4 if i < num_layers - 1 else 1
                )
            )
        
        # Output layer
        self.output = nn.Conv1d(channels[-1], 1, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (B, 1, T) audio
            
        Returns:
            logits: (B, 1, T') discriminator logits
            features: List of intermediate features
        """
        features = []
        
        for block in self.blocks:
            x = block(x)
            features.append(x)
        
        logits = self.output(x)
        
        return logits, features


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator operating on different temporal resolutions.
    """
    
    def __init__(
        self,
        scales: List[int] = [1, 2, 4],
        hidden_dim: int = 64,
        num_layers: int = 4
    ):
        super().__init__()
        
        self.scales = scales
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(hidden_dim, num_layers)
            for _ in scales
        ])
        
        # Downsampling layers
        self.downsamplers = nn.ModuleList()
        for scale in scales[1:]:
            self.downsamplers.append(
                nn.AvgPool1d(kernel_size=scale * 2, stride=scale, padding=scale // 2)
            )
        
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (B, 1, T) audio
            
        Returns:
            logits_list: List of discriminator logits at each scale
            features_list: List of feature lists at each scale
        """
        logits_list = []
        features_list = []
        
        # First scale (original resolution)
        logits, features = self.discriminators[0](x)
        logits_list.append(logits)
        features_list.append(features)
        
        # Other scales (downsampled)
        for downsampler, discriminator in zip(self.downsamplers, self.discriminators[1:]):
            x_down = downsampler(x)
            logits, features = discriminator(x_down)
            logits_list.append(logits)
            features_list.append(features)
        
        return logits_list, features_list


def discriminator_loss(
    disc_real_outputs: List[torch.Tensor],
    disc_fake_outputs: List[torch.Tensor]
) -> torch.Tensor:
    """
    Discriminator loss (hinge loss).
    
    Args:
        disc_real_outputs: List of discriminator outputs for real audio
        disc_fake_outputs: List of discriminator outputs for fake audio
        
    Returns:
        loss: Scalar discriminator loss
    """
    loss = 0.0
    
    for dr, df in zip(disc_real_outputs, disc_fake_outputs):
        # Hinge loss
        loss_real = torch.mean(F.relu(1.0 - dr))
        loss_fake = torch.mean(F.relu(1.0 + df))
        loss += loss_real + loss_fake
    
    return loss / len(disc_real_outputs)


def generator_adversarial_loss(
    disc_fake_outputs: List[torch.Tensor]
) -> torch.Tensor:
    """
    Generator adversarial loss.
    
    Args:
        disc_fake_outputs: List of discriminator outputs for fake audio
        
    Returns:
        loss: Scalar generator loss
    """
    loss = 0.0
    
    for df in disc_fake_outputs:
        loss += torch.mean((1.0 - df) ** 2)  # MSE with target=1
    
    return loss / len(disc_fake_outputs)


def feature_matching_loss(
    disc_real_features: List[List[torch.Tensor]],
    disc_fake_features: List[List[torch.Tensor]]
) -> torch.Tensor:
    """
    Feature matching loss between real and fake discriminator features.
    
    Args:
        disc_real_features: List of feature lists from discriminator (real)
        disc_fake_features: List of feature lists from discriminator (fake)
        
    Returns:
        loss: Scalar feature matching loss
    """
    loss = 0.0
    count = 0
    
    for real_feats, fake_feats in zip(disc_real_features, disc_fake_features):
        for real_feat, fake_feat in zip(real_feats, fake_feats):
            loss += F.l1_loss(fake_feat, real_feat)
            count += 1
    
    return loss / count


if __name__ == "__main__":
    # Test discriminator
    discriminator = MultiScaleDiscriminator(
        scales=[1, 2, 4],
        hidden_dim=64,
        num_layers=4
    )
    
    # Test inputs
    B, T = 2, 16000
    audio_real = torch.randn(B, 1, T)
    audio_fake = torch.randn(B, 1, T)
    
    # Forward pass
    logits_real, features_real = discriminator(audio_real)
    logits_fake, features_fake = discriminator(audio_fake)
    
    print(f"Number of scales: {len(logits_real)}")
    print(f"Logits shapes: {[l.shape for l in logits_real]}")
    print(f"Features per scale: {[len(f) for f in features_real]}")
    
    # Test losses
    d_loss = discriminator_loss(logits_real, logits_fake)
    g_loss = generator_adversarial_loss(logits_fake)
    fm_loss = feature_matching_loss(features_real, features_fake)
    
    print(f"\nDiscriminator loss: {d_loss.item():.4f}")
    print(f"Generator adversarial loss: {g_loss.item():.4f}")
    print(f"Feature matching loss: {fm_loss.item():.4f}")
    
    print("✓ Discriminator test passed!")
