"""
Training script for StreamVC model.
"""

import argparse
import os
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm

from model import StreamVC
from discriminator import (
    MultiScaleDiscriminator,
    discriminator_loss,
    generator_adversarial_loss,
    feature_matching_loss
)
from dataset import VoiceConversionDataset, collate_fn
from audio_utils import MelSpectrogramExtractor, STFTLoss
from f0_extractor import F0ExtractorModule


class StreamVCTrainer:
    """Trainer for StreamVC model."""
    
    def __init__(self, config_path: str, resume_path: str = None):
        """Initialize trainer."""
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Build models
        print("Building models...")
        self.model = StreamVC(self.config).to(self.device)
        self.discriminator = MultiScaleDiscriminator(
            scales=self.config['training']['discriminator']['scales'],
            hidden_dim=self.config['training']['discriminator']['hidden_dim'],
            num_layers=self.config['training']['discriminator']['num_layers']
        ).to(self.device)
        
        # Audio processors
        self.mel_extractor = MelSpectrogramExtractor(
            sample_rate=self.config['audio']['sample_rate'],
            n_fft=self.config['audio']['n_fft'],
            hop_length=self.config['audio']['hop_length'],
            n_mels=self.config['audio']['n_mels']
        ).to(self.device)
        
        self.f0_extractor = F0ExtractorModule(
            sample_rate=self.config['audio']['sample_rate'],
            frame_length=self.config['model']['f0_extractor']['frame_length'],
            f0_min=self.config['model']['f0_extractor']['f0_min'],
            f0_max=self.config['model']['f0_extractor']['f0_max'],
            whitening=self.config['model']['f0_extractor']['whitening']
        )
        
        # Loss functions
        self.stft_loss = STFTLoss().to(self.device)
        
        # Load HuBERT centroids
        print("Loading HuBERT centroids...")
        with open(self.config['hubert']['kmeans_path'], 'rb') as f:
            centroids_data = pickle.load(f)
            self.hubert_centroids = torch.from_numpy(
                centroids_data['centroids']
            ).to(self.device)
        
        # Projection layer to map content features to HuBERT dimension
        hubert_dim = self.hubert_centroids.shape[1]  # 768
        content_dim = self.config['model']['content_encoder']['hidden_dim']  # 64
        self.content_projection = nn.Linear(content_dim, hubert_dim).to(self.device)
        
        # Optimizers
        self.optimizer_g = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.content_projection.parameters()),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        self.optimizer_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Schedulers
        self.scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_g,
            T_max=self.config['training']['num_epochs']
        )
        
        self.scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_d,
            T_max=self.config['training']['num_epochs']
        )
        
        # Create dataset
        print("Creating dataset...")
        self.train_dataset = VoiceConversionDataset(
            data_dirs=self.config['data']['train_datasets'],
            segment_length=int(
                self.config['audio']['sample_rate'] * 1.0  # 1 second segments
            ),
            sample_rate=self.config['audio']['sample_rate'],
            augment=True
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            pin_memory=self.config['data']['pin_memory'],
            collate_fn=collate_fn,
            drop_last=True
        )
        
        # Logging
        log_dir = Path(self.config['logging']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Checkpointing
        self.checkpoint_dir = Path(self.config['checkpointing']['save_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.global_step = 0
        self.epoch = 0
        
        # Resume from checkpoint
        if resume_path:
            self.load_checkpoint(resume_path)
    
    def compute_hubert_targets(self, content_features: torch.Tensor) -> torch.Tensor:
        """
        Compute HuBERT pseudo-labels from content features.
        
        Args:
            content_features: (B, hidden_dim, T)
            
        Returns:
            targets: (B, T) class indices
        """
        B, D, T = content_features.shape
        
        # Reshape for projection
        features = content_features.transpose(1, 2).reshape(-1, D)  # (B*T, D)
        
        # Project to HuBERT dimension
        features = self.content_projection(features)  # (B*T, 768)
        
        # Compute distances to centroids
        distances = torch.cdist(features, self.hubert_centroids)  # (B*T, num_classes)
        
        # Get nearest centroid
        targets = torch.argmin(distances, dim=1)  # (B*T,)
        
        # Reshape back
        targets = targets.view(B, T)
        
        return targets
    
    def train_step(self, source_audio: torch.Tensor, target_audio: torch.Tensor):
        """Single training step."""
        
        B = source_audio.size(0)
        
        # Add channel dimension
        source_audio = source_audio.unsqueeze(1)  # (B, 1, T)
        target_audio = target_audio.unsqueeze(1)  # (B, 1, T)
        
        # Extract features
        target_mel = self.mel_extractor(target_audio)
        f0_features = self.f0_extractor(source_audio)
        
        # Ensure temporal alignment
        min_len = min(target_mel.size(2), f0_features.size(2))
        target_mel = target_mel[:, :, :min_len]
        f0_features = f0_features[:, :, :min_len]
        
        #########################
        # Train Generator
        #########################
        
        self.optimizer_g.zero_grad()
        
        # Forward pass
        output_audio, content_logits = self.model(
            source_audio, target_mel, f0_features
        )
        
        # Ensure same length for losses
        min_audio_len = min(output_audio.size(2), target_audio.size(2))
        output_audio = output_audio[:, :, :min_audio_len]
        target_audio_crop = target_audio[:, :, :min_audio_len]
        
        # Content prediction loss
        # Get HuBERT targets (pseudo-labels)
        with torch.no_grad():
            _, content_features = self.model.content_encoder(source_audio)
            hubert_targets = self.compute_hubert_targets(content_features)
        
        # Crop content_logits to match targets
        min_content_len = min(content_logits.size(2), hubert_targets.size(1))
        content_logits = content_logits[:, :, :min_content_len]
        hubert_targets = hubert_targets[:, :min_content_len]
        
        content_loss = F.cross_entropy(
            content_logits.transpose(1, 2).reshape(-1, content_logits.size(1)),
            hubert_targets.reshape(-1)
        )
        
        # Reconstruction loss (L1)
        recon_loss = F.l1_loss(output_audio, target_audio_crop)
        
        # STFT loss
        stft_loss = self.stft_loss(output_audio, target_audio_crop)
        
        # Discriminator outputs for fake audio
        disc_fake_logits, disc_fake_features = self.discriminator(output_audio)
        
        # Discriminator outputs for real audio (for feature matching)
        with torch.no_grad():
            disc_real_logits, disc_real_features = self.discriminator(target_audio_crop)
        
        # Adversarial loss
        adv_loss = generator_adversarial_loss(disc_fake_logits)
        
        # Feature matching loss
        fm_loss = feature_matching_loss(disc_real_features, disc_fake_features)
        
        # Total generator loss
        loss_g = (
            self.config['training']['losses']['reconstruction_weight'] * recon_loss +
            self.config['training']['losses']['content_prediction_weight'] * content_loss +
            self.config['training']['losses']['stft_weight'] * stft_loss +
            self.config['training']['losses']['adversarial_weight'] * adv_loss +
            self.config['training']['losses']['feature_matching_weight'] * fm_loss
        )
        
        loss_g.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config['training']['gradient']['clip_norm']
        )
        
        self.optimizer_g.step()
        
        #########################
        # Train Discriminator
        #########################
        
        self.optimizer_d.zero_grad()
        
        # Real audio
        disc_real_logits, _ = self.discriminator(target_audio_crop)
        
        # Fake audio (detached)
        with torch.no_grad():
            output_audio_det, _ = self.model(source_audio, target_mel, f0_features)
            output_audio_det = output_audio_det[:, :, :min_audio_len]
        
        disc_fake_logits, _ = self.discriminator(output_audio_det)
        
        # Discriminator loss
        loss_d = discriminator_loss(disc_real_logits, disc_fake_logits)
        
        loss_d.backward()
        self.optimizer_d.step()
        
        # Return losses for logging
        return {
            'loss_g': loss_g.item(),
            'loss_d': loss_d.item(),
            'recon_loss': recon_loss.item(),
            'content_loss': content_loss.item(),
            'stft_loss': stft_loss.item(),
            'adv_loss': adv_loss.item(),
            'fm_loss': fm_loss.item()
        }
    
    def train_epoch(self):
        """Train one epoch."""
        
        self.model.train()
        self.discriminator.train()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, (source_audio, target_audio) in enumerate(pbar):
            source_audio = source_audio.to(self.device)
            target_audio = target_audio.to(self.device)
            
            # Training step
            losses = self.train_step(source_audio, target_audio)
            
            # Update progress bar
            pbar.set_postfix({
                'g': f"{losses['loss_g']:.4f}",
                'd': f"{losses['loss_d']:.4f}",
                'recon': f"{losses['recon_loss']:.4f}"
            })
            
            # Logging
            if self.global_step % self.config['logging']['log_interval'] == 0:
                for k, v in losses.items():
                    self.writer.add_scalar(f'train/{k}', v, self.global_step)
            
            # Checkpointing
            if self.global_step % self.config['checkpointing']['save_interval'] == 0:
                self.save_checkpoint()
            
            self.global_step += 1
        
        # Step schedulers
        self.scheduler_g.step()
        self.scheduler_d.step()
        
        self.epoch += 1
    
    def train(self):
        """Full training loop."""
        
        print("Starting training...")
        print(f"Total epochs: {self.config['training']['num_epochs']}")
        print(f"Batch size: {self.config['training']['batch_size']}")
        print(f"Steps per epoch: {len(self.train_loader)}")
        
        for epoch in range(self.epoch, self.config['training']['num_epochs']):
            self.train_epoch()
            self.save_checkpoint(name=f"checkpoint_epoch_{epoch}.pt")
        
        print("Training complete!")
    
    def save_checkpoint(self, name: str = None):
        """Save checkpoint."""
        
        if name is None:
            name = f"checkpoint_step_{self.global_step}.pt"
        
        checkpoint_path = self.checkpoint_dir / name
        
        torch.save({
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model': self.model.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizer_g': self.optimizer_g.state_dict(),
            'optimizer_d': self.optimizer_d.state_dict(),
            'scheduler_g': self.scheduler_g.state_dict(),
            'scheduler_d': self.scheduler_d.state_dict(),
            'config': self.config
        }, checkpoint_path)
        
        print(f"Saved checkpoint: {checkpoint_path}")
        
        # Keep only last N checkpoints
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
        if len(checkpoints) > self.config['checkpointing']['keep_last_n']:
            for old_ckpt in checkpoints[:-self.config['checkpointing']['keep_last_n']]:
                old_ckpt.unlink()
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        
        print(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d'])
        self.scheduler_g.load_state_dict(checkpoint['scheduler_g'])
        self.scheduler_d.load_state_dict(checkpoint['scheduler_d'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        print(f"Resumed from epoch {self.epoch}, step {self.global_step}")


def main():
    parser = argparse.ArgumentParser(description="Train StreamVC model")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/streamvc_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = StreamVCTrainer(
        config_path=args.config,
        resume_path=args.resume
    )
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
