"""
Training script for StreamVC model.

Supports single-GPU and multi-GPU (DDP) training via torchrun.

Usage:
    # Single GPU
    python streamvc/train.py --config configs/streamvc_config.yaml

    # Multi-GPU (2 GPUs) via torchrun
    torchrun --nproc_per_node=2 streamvc/train.py \
        --config configs/streamvc_config.yaml --ddp

    # Resume training
    torchrun --nproc_per_node=2 streamvc/train.py \
        --config configs/streamvc_config.yaml --ddp \
        --resume checkpoints/streamvc/checkpoint_step_50000.pt
"""

import argparse
import os
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
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
    """
    Trainer for StreamVC model.
    
    Supports single-GPU and multi-GPU DDP training.
    When --ddp is enabled, the script must be launched with torchrun.
    """
    
    def __init__(self, config_path: str, resume_path: str = None, use_ddp: bool = False):
        """Initialize trainer."""
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # ---- DDP setup ----------------------------------------------------------
        self.use_ddp = use_ddp
        if self.use_ddp:
            dist.init_process_group(backend='nccl')
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.is_main = (self.rank == 0)  # Only rank 0 logs / saves
        
        if self.is_main:
            print(f"Using device: {self.device}  |  DDP: {self.use_ddp}  |  World size: {self.world_size}")
        
        # ---- Build models -------------------------------------------------------
        if self.is_main:
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
        if self.is_main:
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
        
        # ---- Wrap with DDP ------------------------------------------------------
        if self.use_ddp:
            self.model = DDP(self.model, device_ids=[self.local_rank],
                             output_device=self.local_rank, find_unused_parameters=False)
            self.discriminator = DDP(self.discriminator, device_ids=[self.local_rank],
                                     output_device=self.local_rank, find_unused_parameters=False)
            self.content_projection = DDP(self.content_projection, device_ids=[self.local_rank],
                                          output_device=self.local_rank)
        
        # ---- Optimizers ---------------------------------------------------------
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
        
        # ---- Datasets with train/val split --------------------------------------
        if self.is_main:
            print("Creating dataset...")
        
        full_dataset = VoiceConversionDataset(
            data_dirs=self.config['data']['train_datasets'],
            segment_length=int(
                self.config['audio']['sample_rate'] * 1.0  # 1 second segments
            ),
            sample_rate=self.config['audio']['sample_rate'],
            augment=True
        )
        
        # Split into train / validation
        val_ratio = self.config['data'].get('val_split', 0.05)
        val_size = int(len(full_dataset) * val_ratio)
        train_size = len(full_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        if self.is_main:
            print(f"  Train samples: {train_size}")
            print(f"  Val samples  : {val_size}")
            if train_size == 0:
                print("  \u26a0 WARNING: Training dataset is EMPTY! Check data.train_datasets paths in config.")
                print(f"    Expected data in: {self.config['data']['train_datasets']}")
        
        # DataLoaders — use DistributedSampler under DDP
        if self.use_ddp:
            self.train_sampler = DistributedSampler(
                self.train_dataset, num_replicas=self.world_size,
                rank=self.rank, shuffle=True, drop_last=True
            )
            shuffle = False  # sampler handles shuffling
        else:
            self.train_sampler = None
            shuffle = True
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=shuffle,
            sampler=self.train_sampler,
            num_workers=self.config['data']['num_workers'],
            pin_memory=self.config['data']['pin_memory'],
            collate_fn=collate_fn,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=max(1, self.config['data']['num_workers'] // 2),
            pin_memory=self.config['data']['pin_memory'],
            collate_fn=collate_fn,
            drop_last=False
        )
        
        # ---- Logging & checkpointing (rank 0 only) -----------------------------
        if self.is_main:
            log_dir = Path(self.config['logging']['log_dir'])
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None
        
        self.checkpoint_dir = Path(self.config['checkpointing']['save_dir'])
        if self.is_main:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Resume from checkpoint
        if resume_path:
            self.load_checkpoint(resume_path)
    
    @property
    def _model(self):
        """Access the underlying model (unwrap DDP if needed)."""
        return self.model.module if self.use_ddp else self.model
    
    @property
    def _discriminator(self):
        """Access the underlying discriminator (unwrap DDP if needed)."""
        return self.discriminator.module if self.use_ddp else self.discriminator
    
    @property
    def _content_projection(self):
        """Access the underlying content_projection (unwrap DDP if needed)."""
        return self.content_projection.module if self.use_ddp else self.content_projection
    
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
        
        # Project to HuBERT dimension (use unwrapped projection)
        proj = self._content_projection
        features = proj(features)  # (B*T, 768)
        
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
            _, content_features = self._model.content_encoder(source_audio)
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
        
        # Gradient clipping (both model and content_projection)
        torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.content_projection.parameters()),
            self.config['training']['gradient']['clip_norm']
        )
        
        self.optimizer_g.step()
        
        #########################
        # Train Discriminator
        #########################
        
        self.optimizer_d.zero_grad()
        
        # Real audio
        disc_real_logits, _ = self.discriminator(target_audio_crop)
        
        # Fake audio (reuse generator output, detached)
        output_audio_det = output_audio.detach()
        
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
        
        # Update sampler epoch for proper shuffling in DDP
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.epoch)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}",
                     disable=not self.is_main)
        
        for batch_idx, (source_audio, target_audio) in enumerate(pbar):
            source_audio = source_audio.to(self.device)
            target_audio = target_audio.to(self.device)
            
            # Training step
            losses = self.train_step(source_audio, target_audio)
            
            # Update progress bar (rank 0 only)
            if self.is_main:
                pbar.set_postfix({
                    'g': f"{losses['loss_g']:.4f}",
                    'd': f"{losses['loss_d']:.4f}",
                    'recon': f"{losses['recon_loss']:.4f}"
                })
            
            # Logging (rank 0 only)
            if self.is_main and self.global_step % self.config['logging']['log_interval'] == 0:
                for k, v in losses.items():
                    self.writer.add_scalar(f'train/{k}', v, self.global_step)
                self.writer.add_scalar('train/lr_g',
                                       self.optimizer_g.param_groups[0]['lr'],
                                       self.global_step)
            
            # Checkpointing (rank 0 only)
            if self.is_main and self.global_step % self.config['checkpointing']['save_interval'] == 0:
                self.save_checkpoint()
            
            self.global_step += 1
        
        # Step schedulers
        self.scheduler_g.step()
        self.scheduler_d.step()
        
        self.epoch += 1
    
    @torch.no_grad()
    def validate(self):
        """
        Run validation and return average losses.
        Only executed on rank 0 (main process).
        """
        if not self.is_main:
            return {}
        
        self.model.eval()
        self.discriminator.eval()
        
        val_losses = {}
        num_batches = 0
        
        for source_audio, target_audio in tqdm(self.val_loader, desc="Validation",
                                                 leave=False, disable=not self.is_main):
            source_audio = source_audio.to(self.device)
            target_audio = target_audio.to(self.device)
            
            B = source_audio.size(0)
            source_audio = source_audio.unsqueeze(1)
            target_audio = target_audio.unsqueeze(1)
            
            target_mel = self.mel_extractor(target_audio)
            f0_features = self.f0_extractor(source_audio)
            
            min_len = min(target_mel.size(2), f0_features.size(2))
            target_mel = target_mel[:, :, :min_len]
            f0_features = f0_features[:, :, :min_len]
            
            output_audio, content_logits = self.model(
                source_audio, target_mel, f0_features
            )
            
            min_audio_len = min(output_audio.size(2), target_audio.size(2))
            output_audio = output_audio[:, :, :min_audio_len]
            target_audio_crop = target_audio[:, :, :min_audio_len]
            
            recon_loss = F.l1_loss(output_audio, target_audio_crop).item()
            stft_loss = self.stft_loss(output_audio, target_audio_crop).item()
            
            batch_losses = {
                'val_recon_loss': recon_loss,
                'val_stft_loss': stft_loss,
                'val_total_loss': recon_loss + stft_loss,
            }
            
            for k, v in batch_losses.items():
                val_losses[k] = val_losses.get(k, 0.0) + v
            num_batches += 1
        
        if num_batches > 0:
            for k in val_losses:
                val_losses[k] /= num_batches
        
        # Log validation metrics
        if self.writer is not None:
            for k, v in val_losses.items():
                self.writer.add_scalar(f'val/{k}', v, self.global_step)
        
        return val_losses
    
    def train(self):
        """Full training loop."""
        
        if self.is_main:
            print("Starting training...")
            print(f"  Total epochs     : {self.config['training']['num_epochs']}")
            print(f"  Batch size/GPU   : {self.config['training']['batch_size']}")
            print(f"  World size       : {self.world_size}")
            print(f"  Effective batch  : {self.config['training']['batch_size'] * self.world_size}")
            print(f"  Steps per epoch  : {len(self.train_loader)}")
            print(f"  Val batches      : {len(self.val_loader)}")
        
        for epoch in range(self.epoch, self.config['training']['num_epochs']):
            self.train_epoch()
            
            # Validation every epoch
            val_losses = self.validate()
            
            if self.is_main:
                val_total = val_losses.get('val_total_loss', float('inf'))
                print(f"  Epoch {epoch} val loss: {val_total:.4f}  "
                      f"(best: {self.best_val_loss:.4f})")
                
                self.save_checkpoint(name=f"checkpoint_epoch_{epoch}.pt")
                
                # Save best model
                if val_total < self.best_val_loss:
                    self.best_val_loss = val_total
                    self.save_checkpoint(name="best.pt")
                    print(f"  ★ New best model saved (val_loss={val_total:.4f})")
            
            # Synchronize before next epoch
            if self.use_ddp:
                dist.barrier()
        
        if self.is_main:
            print("Training complete!")
        
        # Cleanup DDP
        if self.use_ddp:
            dist.destroy_process_group()
    
    def save_checkpoint(self, name: str = None):
        """Save checkpoint (rank 0 only)."""
        
        if not self.is_main:
            return
        
        if name is None:
            name = f"checkpoint_step_{self.global_step}.pt"
        
        checkpoint_path = self.checkpoint_dir / name
        
        torch.save({
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'model': self._model.state_dict(),
            'discriminator': self._discriminator.state_dict(),
            'content_projection': self._content_projection.state_dict(),
            'optimizer_g': self.optimizer_g.state_dict(),
            'optimizer_d': self.optimizer_d.state_dict(),
            'scheduler_g': self.scheduler_g.state_dict(),
            'scheduler_d': self.scheduler_d.state_dict(),
            'config': self.config
        }, checkpoint_path)
        
        print(f"Saved checkpoint: {checkpoint_path}")
        
        # Keep only last N step checkpoints (don't delete epoch/best checkpoints)
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
        if len(checkpoints) > self.config['checkpointing']['keep_last_n']:
            for old_ckpt in checkpoints[:-self.config['checkpointing']['keep_last_n']]:
                old_ckpt.unlink()
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        
        if self.is_main:
            print(f"Loading checkpoint: {path}")
        
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self._model.load_state_dict(checkpoint['model'])
        self._discriminator.load_state_dict(checkpoint['discriminator'])
        
        # Backward-compatible: content_projection may not exist in old checkpoints
        if 'content_projection' in checkpoint:
            self._content_projection.load_state_dict(checkpoint['content_projection'])
        
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d'])
        self.scheduler_g.load_state_dict(checkpoint['scheduler_g'])
        self.scheduler_d.load_state_dict(checkpoint['scheduler_d'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if self.is_main:
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
    parser.add_argument(
        "--ddp",
        action="store_true",
        default=False,
        help="Enable DistributedDataParallel (launch with torchrun)"
    )
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = StreamVCTrainer(
        config_path=args.config,
        resume_path=args.resume,
        use_ddp=args.ddp
    )
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
