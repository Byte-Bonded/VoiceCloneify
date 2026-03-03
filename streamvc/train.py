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

import numpy as np
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
import librosa

from model import StreamVC
from discriminator import (
    MultiScaleDiscriminator,
    discriminator_loss,
    generator_adversarial_loss,
    feature_matching_loss,
    r1_gradient_penalty,
    compute_disc_accuracy
)
from dataset import VoiceConversionDataset, collate_fn, create_val_split
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
            num_layers=self.config['training']['discriminator']['num_layers'],
            use_spectral_norm=self.config['training']['discriminator'].get('use_spectral_norm', True)
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
        
        # Load HuBERT model for computing ground-truth content targets
        if self.is_main:
            print("Loading HuBERT model for content targets...")
        from transformers import HubertModel
        self.hubert_model = HubertModel.from_pretrained(
            self.config['hubert']['model_name']
        ).to(self.device)
        self.hubert_model.eval()
        for p in self.hubert_model.parameters():
            p.requires_grad = False
        self.hubert_layer = self.config['hubert'].get('layer', 7)
        if self.is_main:
            print(f"  HuBERT model loaded (layer {self.hubert_layer}, frozen)")
        
        # ---- Wrap with DDP ------------------------------------------------------
        if self.use_ddp:
            self.model = DDP(self.model, device_ids=[self.local_rank],
                             output_device=self.local_rank, find_unused_parameters=False)
            self.discriminator = DDP(self.discriminator, device_ids=[self.local_rank],
                                     output_device=self.local_rank, find_unused_parameters=False)
        
        # ---- Optimizers ---------------------------------------------------------
        base_lr = self.config['training']['learning_rate']
        disc_lr_mult = self.config['training'].get('disc_lr_multiplier', 2.0)
        
        self.optimizer_g = torch.optim.AdamW(
            list(self.model.parameters()),
            lr=base_lr,
            weight_decay=self.config['training']['weight_decay']
        )
        
        self.optimizer_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=base_lr * disc_lr_mult,  # Higher LR keeps D competitive
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Schedulers — D uses a slower decay to stay competitive
        self.scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_g,
            T_max=self.config['training']['num_epochs'],
            eta_min=base_lr * 0.01  # Decay to 1% of base
        )
        
        self.scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_d,
            T_max=self.config['training']['num_epochs'],
            eta_min=base_lr * disc_lr_mult * 0.1  # D decays less aggressively (10%)
        )
        
        # Discriminator training config
        self.disc_warmup_steps = self.config['training'].get('disc_warmup_steps', 5000)
        self.r1_penalty_weight = self.config['training'].get('r1_penalty_weight', 10.0)
        self.r1_interval = self.config['training'].get('r1_interval', 16)  # Lazy R1
        self.disc_grad_clip = self.config['training'].get('disc_grad_clip', 1.0)
        self.disc_steps_per_g = self.config['training'].get('disc_steps_per_g', 1)
        
        # ---- Datasets with deterministic train/val split -----------------------
        if self.is_main:
            print("Creating dataset...")
        
        segment_length = int(
            self.config['audio']['sample_rate'] *
            self.config.get('data', {}).get('segment_duration_s', 2.0)
        )
        val_ratio = self.config.get('data', {}).get('val_ratio', 0.05)

        train_files, val_files = create_val_split(
            self.config['data']['train_datasets'],
            val_ratio=val_ratio,
            seed=42
        )

        self.train_dataset = VoiceConversionDataset(
            file_list=train_files,
            segment_length=segment_length,
            sample_rate=self.config['audio']['sample_rate'],
            augment=True
        )
        self.val_dataset = VoiceConversionDataset(
            file_list=val_files,
            segment_length=segment_length,
            sample_rate=self.config['audio']['sample_rate'],
            augment=False   # No augmentation during validation
        )

        if self.is_main:
            print(f"Dataset split: {len(train_files)} train | {len(val_files)} val")
            if len(train_files) == 0:
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
    
    @torch.no_grad()
    def compute_hubert_targets(self, source_audio: torch.Tensor) -> torch.Tensor:
        """
        Compute ground-truth HuBERT pseudo-labels from source audio.
        
        Uses the frozen HuBERT model to extract features from layer N,
        then finds the nearest centroid for each frame.
        
        Args:
            source_audio: (B, 1, T) raw waveform at 16kHz
            
        Returns:
            targets: (B, T') class indices at 50Hz
        """
        # HuBERT expects (B, T)
        wav = source_audio.squeeze(1)  # (B, T)
        
        outputs = self.hubert_model(wav, output_hidden_states=True)
        # hidden_states[0] = CNN output, hidden_states[i] = transformer layer i
        hidden = outputs.hidden_states[self.hubert_layer]  # (B, T', 768)
        
        B, T_prime, D = hidden.shape
        features_flat = hidden.reshape(-1, D)  # (B*T', 768)
        
        # Find nearest centroid
        distances = torch.cdist(features_flat, self.hubert_centroids)  # (B*T', num_classes)
        targets = torch.argmin(distances, dim=1).view(B, T_prime)  # (B, T')
        
        return targets
    
    def train_step(self, source_audio: torch.Tensor, speaker_ref_audio: torch.Tensor):
        """
        Single training step — SELF-RECONSTRUCTION paradigm.
        
        source_audio:      content source + F0 source + reconstruction TARGET
        speaker_ref_audio: different utterance from SAME speaker → speaker embedding
        
        The model must reconstruct source_audio from:
          - content features of source_audio
          - speaker embedding from speaker_ref_audio
          - F0 from source_audio
        """
        
        B = source_audio.size(0)
        
        # Add channel dimension
        source_audio = source_audio.unsqueeze(1)      # (B, 1, T)
        speaker_ref_audio = speaker_ref_audio.unsqueeze(1)  # (B, 1, T)
        
        # Extract features — speaker mel from the REFERENCE audio (same speaker, diff utterance)
        speaker_ref_mel = self.mel_extractor(speaker_ref_audio)
        f0_features = self.f0_extractor(source_audio)
        
        # Ensure temporal alignment for mel/f0
        min_len = min(speaker_ref_mel.size(2), f0_features.size(2))
        speaker_ref_mel = speaker_ref_mel[:, :, :min_len]
        f0_features = f0_features[:, :, :min_len]
        
        # Reconstruction target is source_audio itself
        recon_target = source_audio
        
        # Check if we're in discriminator warmup phase
        in_warmup = self.global_step < self.disc_warmup_steps
        
        #########################
        # Train Discriminator FIRST
        #########################
        
        with torch.no_grad():
            output_audio_for_d, _ = self.model(
                source_audio, speaker_ref_mel, f0_features
            )
            min_audio_len = min(output_audio_for_d.size(2), recon_target.size(2))
            output_audio_for_d = output_audio_for_d[:, :, :min_audio_len]
            recon_target_crop = recon_target[:, :, :min_audio_len]
        
        self.optimizer_d.zero_grad()
        
        # Real audio (the source audio we want to reconstruct)
        disc_real_logits, _ = self.discriminator(recon_target_crop)
        
        # Fake audio (detached from generator)
        disc_fake_logits, _ = self.discriminator(output_audio_for_d)
        
        # Hinge discriminator loss
        loss_d = discriminator_loss(disc_real_logits, disc_fake_logits)
        
        # R1 gradient penalty (lazy — every r1_interval steps)
        r1_loss = torch.tensor(0.0, device=self.device)
        if self.global_step % self.r1_interval == 0:
            r1_loss = r1_gradient_penalty(self.discriminator, recon_target_crop)
            loss_d = loss_d + self.r1_penalty_weight * r1_loss
        
        loss_d.backward()
        
        # Gradient clipping for discriminator
        torch.nn.utils.clip_grad_norm_(
            self.discriminator.parameters(),
            self.disc_grad_clip
        )
        
        self.optimizer_d.step()
        
        # Compute discriminator health metrics
        with torch.no_grad():
            disc_acc = compute_disc_accuracy(disc_real_logits, disc_fake_logits)
        
        #########################
        # Train Generator (skip during warmup to let D stabilize)
        #########################
        
        if in_warmup:
            return {
                'loss_g': 0.0,
                'loss_d': loss_d.item(),
                'recon_loss': 0.0,
                'content_loss': 0.0,
                'stft_loss': 0.0,
                'adv_loss': 0.0,
                'fm_loss': 0.0,
                'r1_penalty': r1_loss.item(),
                'disc_acc_real': disc_acc['disc_acc_real'],
                'disc_acc_fake': disc_acc['disc_acc_fake'],
                'disc_acc': disc_acc['disc_acc'],
                'warmup': 1.0,
            }
        
        self.optimizer_g.zero_grad()
        
        # Fresh forward pass for generator (needs gradients)
        output_audio, content_logits = self.model(
            source_audio, speaker_ref_mel, f0_features
        )
        
        # Ensure same length for losses — compare against source_audio (self-reconstruction!)
        min_audio_len = min(output_audio.size(2), recon_target.size(2))
        output_audio = output_audio[:, :, :min_audio_len]
        recon_target_crop = recon_target[:, :, :min_audio_len]
        
        # Content prediction loss
        hubert_targets = self.compute_hubert_targets(source_audio)
        
        min_content_len = min(content_logits.size(2), hubert_targets.size(1))
        content_logits = content_logits[:, :, :min_content_len]
        hubert_targets = hubert_targets[:, :min_content_len]
        
        content_loss = F.cross_entropy(
            content_logits.transpose(1, 2).reshape(-1, content_logits.size(1)),
            hubert_targets.reshape(-1)
        )
        
        # Reconstruction loss (L1) — compare against SOURCE audio (self-reconstruction)
        recon_loss = F.l1_loss(output_audio, recon_target_crop)
        
        # STFT loss — also against source audio
        stft_loss = self.stft_loss(output_audio, recon_target_crop)
        
        # Discriminator outputs for fake audio
        disc_fake_logits_g, disc_fake_features = self.discriminator(output_audio)
        
        # Discriminator outputs for real audio (for feature matching)
        with torch.no_grad():
            disc_real_logits_g, disc_real_features = self.discriminator(recon_target_crop)
        
        # Adversarial loss — scale by D health
        adv_loss = generator_adversarial_loss(disc_fake_logits_g)
        
        # Feature matching loss
        fm_loss = feature_matching_loss(disc_real_features, disc_fake_features)
        
        # Adaptive adversarial weight based on D health
        adv_weight = self.config['training']['losses']['adversarial_weight']
        fm_weight = self.config['training']['losses']['feature_matching_weight']
        
        if disc_acc['disc_acc'] < 0.55:
            adv_scale = 0.1
        elif disc_acc['disc_acc'] < 0.65:
            adv_scale = 0.5
        else:
            adv_scale = 1.0
        
        # Total generator loss
        loss_g = (
            self.config['training']['losses']['reconstruction_weight'] * recon_loss +
            self.config['training']['losses']['content_prediction_weight'] * content_loss +
            self.config['training']['losses']['stft_weight'] * stft_loss +
            adv_weight * adv_scale * adv_loss +
            fm_weight * adv_scale * fm_loss
        )
        
        loss_g.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config['training']['gradient']['clip_norm']
        )
        
        self.optimizer_g.step()
        
        return {
            'loss_g': loss_g.item(),
            'loss_d': loss_d.item(),
            'recon_loss': recon_loss.item(),
            'content_loss': content_loss.item(),
            'stft_loss': stft_loss.item(),
            'adv_loss': adv_loss.item(),
            'fm_loss': fm_loss.item(),
            'r1_penalty': r1_loss.item(),
            'disc_acc_real': disc_acc['disc_acc_real'],
            'disc_acc_fake': disc_acc['disc_acc_fake'],
            'disc_acc': disc_acc['disc_acc'],
            'adv_scale': adv_scale,
            'warmup': 0.0,
        }
    
    @torch.no_grad()
    def validate_epoch(self) -> dict:
        """
        Run one validation pass — self-reconstruction evaluation.

        Computes per-epoch:
          val_recon_loss   : L1 waveform reconstruction (output vs source)
          val_stft_loss    : Multi-scale STFT loss
          val_content_loss : HuBERT pseudo-label cross-entropy
          val_mcd_db       : Mel Cepstral Distortion (dB, C0 excluded)
          val_loss         : Composite loss used for best-checkpoint tracking
        """
        self.model.eval()
        self.discriminator.eval()

        total_recon   = 0.0
        total_stft    = 0.0
        total_content = 0.0
        total_mcd     = 0.0
        n_batches     = 0

        pbar = tqdm(self.val_loader, desc=f"Val   {self.epoch}", leave=False)
        for source_audio, speaker_ref_audio in pbar:
            source_audio = source_audio.to(self.device)
            speaker_ref_audio = speaker_ref_audio.to(self.device)

            source_audio = source_audio.unsqueeze(1)        # (B, 1, T)
            speaker_ref_audio = speaker_ref_audio.unsqueeze(1)

            # Speaker mel from REFERENCE audio (same speaker, diff utterance)
            speaker_ref_mel = self.mel_extractor(speaker_ref_audio)
            f0_features = self.f0_extractor(source_audio)

            min_len = min(speaker_ref_mel.size(2), f0_features.size(2))
            speaker_ref_mel = speaker_ref_mel[:, :, :min_len]
            f0_features = f0_features[:, :, :min_len]

            output_audio, content_logits = self.model(source_audio, speaker_ref_mel, f0_features)

            # Reconstruction target = source_audio
            recon_target = source_audio
            min_audio_len    = min(output_audio.size(2), recon_target.size(2))
            output_audio_crop = output_audio[:, :, :min_audio_len]
            recon_target_crop = recon_target[:, :, :min_audio_len]

            # Reconstruction losses (against source audio)
            recon_loss = F.l1_loss(output_audio_crop, recon_target_crop)
            stft_loss  = self.stft_loss(output_audio_crop, recon_target_crop)

            # Content loss — ground-truth HuBERT targets from source
            hubert_targets = self.compute_hubert_targets(source_audio)
            min_cl = min(content_logits.size(2), hubert_targets.size(1))
            content_loss = F.cross_entropy(
                content_logits[:, :, :min_cl].transpose(1, 2).reshape(-1, content_logits.size(1)),
                hubert_targets[:, :min_cl].reshape(-1)
            )

            # MCD — proper MFCC-based Mel Cepstral Distortion (dB)
            # C0 excluded (energy coefficient) for standard MCD measurement
            mcd_batch_val = 0.0
            out_np = output_audio_crop.squeeze(1).cpu().numpy()
            tgt_np = recon_target_crop.squeeze(1).cpu().numpy()
            n_valid_mcd = 0
            for b_idx in range(out_np.shape[0]):
                try:
                    mfcc_out = librosa.feature.mfcc(y=out_np[b_idx], sr=16000, n_mfcc=13,
                                                     hop_length=256, n_fft=1024)
                    mfcc_tgt = librosa.feature.mfcc(y=tgt_np[b_idx], sr=16000, n_mfcc=13,
                                                     hop_length=256, n_fft=1024)
                    min_f = min(mfcc_out.shape[1], mfcc_tgt.shape[1])
                    diff = mfcc_out[1:, :min_f] - mfcc_tgt[1:, :min_f]  # Exclude C0
                    mcd_sample = (10.0 / np.log(10.0)) * np.sqrt(2.0) * np.mean(
                        np.sqrt(np.sum(diff ** 2, axis=0))
                    )
                    mcd_batch_val += mcd_sample
                    n_valid_mcd += 1
                except Exception:
                    pass
            if n_valid_mcd > 0:
                mcd_batch_val /= n_valid_mcd

            total_recon   += recon_loss.item()
            total_stft    += stft_loss.item()
            total_content += content_loss.item()
            total_mcd     += mcd_batch_val
            n_batches     += 1

        val_metrics = {
            'val_recon_loss':   total_recon   / max(n_batches, 1),
            'val_stft_loss':    total_stft    / max(n_batches, 1),
            'val_content_loss': total_content / max(n_batches, 1),
            'val_mcd_db':       total_mcd     / max(n_batches, 1),
        }
        val_metrics['val_loss'] = (
            val_metrics['val_recon_loss'] +
            self.config['training']['losses']['stft_weight'] * val_metrics['val_stft_loss']
        )

        # Log to TensorBoard (rank 0 only)
        if self.writer is not None:
            for k, v in val_metrics.items():
                self.writer.add_scalar(f'val/{k}', v, self.epoch)

        if self.is_main:
            print(
                f"  [Val Epoch {self.epoch}]  "
                f"recon={val_metrics['val_recon_loss']:.4f}  "
                f"stft={val_metrics['val_stft_loss']:.4f}  "
                f"content={val_metrics['val_content_loss']:.4f}  "
                f"MCD={val_metrics['val_mcd_db']:.2f} dB  "
                f"val_loss={val_metrics['val_loss']:.4f}"
            )

        # Save best checkpoint based on composite val loss
        if val_metrics['val_loss'] < self.best_val_loss:
            self.best_val_loss = val_metrics['val_loss']
            self.save_checkpoint(name="best_model.pt")
            if self.is_main:
                print(f"  ✓ New best val loss: {self.best_val_loss:.4f}  → saved best_model.pt")

        return val_metrics

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
                    'recon': f"{losses['recon_loss']:.4f}",
                    'Dacc': f"{losses.get('disc_acc', 0):.2f}",
                })
            
            # Logging (rank 0 only)
            if self.is_main and self.global_step % self.config['logging']['log_interval'] == 0:
                for k, v in losses.items():
                    self.writer.add_scalar(f'train/{k}', v, self.global_step)
                self.writer.add_scalar('train/lr_g',
                                       self.optimizer_g.param_groups[0]['lr'],
                                       self.global_step)
                self.writer.add_scalar('train/lr_d',
                                       self.optimizer_d.param_groups[0]['lr'],
                                       self.global_step)
            
            # Checkpointing (rank 0 only)
            if self.is_main and self.global_step % self.config['checkpointing']['save_interval'] == 0:
                self.save_checkpoint()
            
            self.global_step += 1
        
        # Step schedulers
        self.scheduler_g.step()
        self.scheduler_d.step()
        
        self.epoch += 1
    
    def train(self):
        """Full training loop with per-epoch validation, early stopping, and DDP support."""
        
        val_freq = self.config.get('training', {}).get('val_every_n_epochs', 1)
        patience = self.config.get('training', {}).get('early_stopping_patience', 15)
        epochs_without_improvement = 0
        
        if self.is_main:
            print("Starting training...")
            print(f"  Total epochs     : {self.config['training']['num_epochs']}")
            print(f"  Batch size/GPU   : {self.config['training']['batch_size']}")
            print(f"  World size       : {self.world_size}")
            print(f"  Effective batch  : {self.config['training']['batch_size'] * self.world_size}")
            print(f"  Steps per epoch  : {len(self.train_loader)}")
            print(f"  Val batches      : {len(self.val_loader)}")
            print(f"  Validate every   : {val_freq} epoch(s)")
            print(f"  Early stopping   : patience={patience} epochs")
            print(f"  G learning rate  : {self.optimizer_g.param_groups[0]['lr']:.1e}")
            print(f"  D learning rate  : {self.optimizer_d.param_groups[0]['lr']:.1e}")
            print(f"  D warmup steps   : {self.disc_warmup_steps}")
            print(f"  R1 penalty       : {self.r1_penalty_weight} (every {self.r1_interval} steps)")
            print(f"  D grad clip      : {self.disc_grad_clip}")
        
        for epoch in range(self.epoch, self.config['training']['num_epochs']):
            self.train_epoch()
            
            # Validation
            if (epoch + 1) % val_freq == 0:
                prev_best = self.best_val_loss
                self.validate_epoch()
                
                # Early stopping check
                if self.best_val_loss < prev_best:
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if self.is_main:
                        print(f"  No improvement for {epochs_without_improvement}/{patience} epochs")
                    if epochs_without_improvement >= patience:
                        if self.is_main:
                            print(f"\n  ⏹ Early stopping triggered after {patience} epochs without improvement.")
                            print(f"  Best val_loss: {self.best_val_loss:.4f}")
                        break
            
            if self.is_main:
                self.save_checkpoint(name=f"checkpoint_epoch_{epoch}.pt")
                
                # Prune old epoch checkpoints — keep only last N
                keep_n = self.config['checkpointing']['keep_last_n']
                epoch_ckpts = sorted(
                    self.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
                    key=lambda p: int(p.stem.split('_')[-1])
                )
                for old in epoch_ckpts[:-keep_n]:
                    old.unlink()
            
            # Synchronize before next epoch
            if self.use_ddp:
                dist.barrier()
        
        if self.is_main:
            print(f"Training complete! Best val loss: {self.best_val_loss:.4f}")
        
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
            'optimizer_g': self.optimizer_g.state_dict(),
            'optimizer_d': self.optimizer_d.state_dict(),
            'scheduler_g': self.scheduler_g.state_dict(),
            'scheduler_d': self.scheduler_d.state_dict(),
            'config': self.config
        }, checkpoint_path)
        
        print(f"Saved checkpoint: {checkpoint_path}")
        
        # Keep only last N step checkpoints (sort numerically by step number)
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_step_*.pt"),
            key=lambda p: int(p.stem.split('_')[-1])
        )
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
        
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d'])
        self.scheduler_g.load_state_dict(checkpoint['scheduler_g'])
        self.scheduler_d.load_state_dict(checkpoint['scheduler_d'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if self.is_main:
            print(f"Resumed from epoch {self.epoch}, step {self.global_step}, best_val_loss={self.best_val_loss:.4f}")


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
