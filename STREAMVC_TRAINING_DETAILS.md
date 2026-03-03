# StreamVC Training Details - Deep Dive

## Table of Contents
1. [Overview](#overview)
2. [Architecture Components](#architecture-components)
3. [GAN Training Process](#gan-training-process)
4. [Loss Functions Explained](#loss-functions-explained)
5. [Training Loop Details](#training-loop-details)
6. [Mathematical Formulations](#mathematical-formulations)
7. [Pipeline Flow](#pipeline-flow)
8. [Training Configuration](#training-configuration)

---

## Overview

**StreamVC** is a streaming voice conversion model that transforms speech from one speaker to sound like a target speaker while preserving the linguistic content. It uses a **GAN (Generative Adversarial Network)** architecture with multiple specialized components.

### Key Innovation
- **Streaming-capable**: 60ms architectural latency (2-frame lookahead at 50Hz)
- **Disentangled representations**: Separates content, speaker identity, and prosody
- **Many-to-many**: Can convert any source speaker to any target speaker

---

## Architecture Components

### 1. Content Encoder (Linguistic Information)

**Purpose**: Extract "what is being said" independent of "who is saying it"

**Architecture**:
```
Input: Raw audio waveform (B, 1, 32000) @ 16kHz

Layer 1: Causal Conv1d (1 → 64 channels, kernel=7)
├─ Ensures causality (no future information)
└─ Projects mono audio to feature space

Layers 2-9: 8 Residual Blocks
├─ Dilation pattern: 1, 2, 4, 8, 16, 32, 64, 128
├─ Each block:
│   ├─ CausalConv1d(64 → 64, kernel=5, dilation=d)
│   ├─ LayerNorm + GELU
│   ├─ CausalConv1d(64 → 64, kernel=5, dilation=d)
│   ├─ LayerNorm
│   └─ Skip connection + GELU
└─ Receptive field: 2^8 * 5 = 1280 samples (80ms)

Layer 10: Temporal Downsampling
├─ Strided Conv1d (stride=320)
├─ Downsamples from 16kHz to 50Hz
└─ Output: (B, 64, 100) for 2-second audio

Layer 11: Content Prediction Head
├─ Conv1d(64 → 100, kernel=1)
└─ Predicts HuBERT cluster (soft labels)

Output:
├─ content_features: (B, 64, T) @ 50Hz
└─ content_logits: (B, 100, T) - discrete content units
```

**Why This Design?**
- **Causal convolutions**: Enable streaming (no future peeking)
- **Exponential dilation**: Large receptive field without many parameters
- **50Hz frame rate**: Matches typical phoneme duration (~20ms)
- **HuBERT supervision**: Forces content features to represent phonetic units

---

### 2. Speaker Encoder (Identity Information)

**Purpose**: Extract "who is speaking" independent of "what is being said"

**Architecture**:
```
Input: Mel spectrogram from target speaker (B, 80, T_mel)

Layer 1: Conv1d(80 → 256, kernel=7)
├─ Projects mel features to hidden space
└─ ReLU activation

Layers 2-5: 4 Residual Blocks
├─ Each block:
│   ├─ Conv1d(256 → 256, kernel=5)
│   ├─ BatchNorm + ReLU
│   ├─ Conv1d(256 → 256, kernel=5)
│   ├─ BatchNorm
│   └─ Skip connection + ReLU
└─ No dilation (all frames equally important)

Layer 6: Attention Pooling (Learnable)
├─ Query: Learnable parameter (1, 256)
├─ Keys/Values: Hidden features (B, 256, T)
├─ Attention weights: softmax(Query · Keys)
├─ Weighted sum across time
└─ Focuses on speaker-characteristic regions

Layer 7: Output Projection
├─ Linear(256 → 256)
└─ Final speaker embedding

Output:
└─ speaker_embedding: (B, 256) - global speaker vector
```

**Why This Design?**
- **Mel spectrogram input**: Contains timbral/spectral information
- **Attention pooling**: Learns which frames are most speaker-discriminative
- **Global embedding**: Single vector represents entire speaker identity
- **Time-invariant**: Same speaker → same embedding regardless of content

---

### 3. F0 (Pitch) Extractor (Prosody Information)

**Purpose**: Extract "how it's said" - pitch contour and intonation

**Algorithm**: Yin (via Parselmouth/Praat)

**Process**:
```
Input: Raw audio waveform (B, 1, 32000)

Step 1: Yin F0 Extraction
├─ Frame length: 20ms (320 samples @ 16kHz)
├─ Hop size: 20ms → 50Hz frame rate
├─ F0 range: 50-600 Hz
└─ Output: f0_contour (B, 100) for 2-sec audio

Step 2: Voiced/Unvoiced Detection
├─ voiced_mask = (f0 > 0)
└─ Distinguishes speech from silence/consonants

Step 3: Interpolation
├─ Fill unvoiced frames with linear interpolation
├─ Prevents discontinuities
└─ Smooth f0 contour

Step 4: Uncertainty Features (9 dimensions)
├─ f0_value (1D): Log-scaled fundamental frequency
├─ f0_confidence (1D): Yin algorithm confidence
├─ f0_delta (1D): Frame-to-frame change
├─ f0_delta_delta (1D): Acceleration
├─ voiced_prob (1D): Probability of voicing
└─ 4 additional uncertainty features
    ├─ Local variance
    ├─ Spectral flatness
    ├─ Harmonic-to-noise ratio
    └─ Zero-crossing rate

Step 5: Whitening (Per-utterance normalization)
├─ mean = f0_features.mean(dim=1, keepdim=True)
├─ std = f0_features.std(dim=1, keepdim=True)
├─ f0_normalized = (f0_features - mean) / (std + 1e-8)
└─ Removes speaker-specific pitch range

Output:
└─ f0_features: (B, 9, 100) @ 50Hz
```

**Why 9 Dimensions?**
- **f0_value**: Core pitch information
- **Confidence**: Reliability of extraction
- **Deltas**: Prosodic dynamics (intonation patterns)
- **Uncertainty**: Helps model handle extraction errors
- **Whitening**: Removes speaker-specific pitch range, allows content transfer

---

### 4. Decoder (Audio Synthesis)

**Purpose**: Combine content + speaker + F0 to generate converted audio

**Architecture**:
```
Inputs:
├─ content_features: (B, 64, 100)
├─ f0_features: (B, 9, 100)
└─ speaker_embedding: (B, 256)

Layer 1: Feature Concatenation
├─ cat(content_features, f0_features) along channel dim
└─ Combined: (B, 73, 100)

Layer 2: Input Projection
├─ CausalConv1d(73 → 40, kernel=7)
└─ Project to decoder hidden dim

Layers 3-14: 12 Residual Blocks with FiLM Conditioning
├─ For each block:
│   ├─ Residual Block:
│   │   ├─ CausalConv1d(40 → 40, dilation=2^(i%4))
│   │   ├─ LayerNorm + GELU
│   │   ├─ CausalConv1d(40 → 40, dilation=2^(i%4))
│   │   ├─ LayerNorm
│   │   └─ Skip connection
│   │
│   └─ FiLM Layer (Feature-wise Linear Modulation):
│       ├─ scale = Linear(speaker_embedding → 40)
│       ├─ shift = Linear(speaker_embedding → 40)
│       └─ output = (input * (1 + scale)) + shift
│
└─ FiLM injects speaker identity into each layer

Layer 15: Temporal Upsampling
├─ ConvTranspose1d(stride=320)
├─ Upsamples from 50Hz to 16kHz
├─ (B, 40, 100) → (B, 40, 32000)
└─ Back to audio sample rate

Layer 16: Output Projection
├─ Conv1d(40 → 40, kernel=1) + GELU
├─ Conv1d(40 → 1, kernel=1)
└─ Tanh activation (clips to [-1, 1])

Output:
└─ audio_waveform: (B, 1, 32000) @ 16kHz
```

**FiLM (Feature-wise Linear Modulation) Explained**:
```python
# Traditional approach:
x = f(content, speaker)  # Speaker affects entire computation

# FiLM approach:
scale = W_scale(speaker_embedding)  # (B, 40)
shift = W_shift(speaker_embedding)  # (B, 40)
x = x * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)

# Effect: Speaker modulates each feature channel independently
# - scale: Amplifies/attenuates features (e.g., formants)
# - shift: Adds/removes features (e.g., breathiness)
```

**Why FiLM?**
- **Disentanglement**: Speaker doesn't mix with content prematurely
- **Controllability**: Each speaker dimension affects specific acoustic features
- **Efficiency**: Conditional generation without concatenation overhead

---

### 5. Discriminator (Realism Evaluator)

**Purpose**: Distinguish real audio from generated audio (adversarial training)

**Architecture**: Multi-Scale Discriminator (MSD)
```
3 Parallel Discriminators at different scales:
├─ Scale 1: Original audio (16kHz)
├─ Scale 2: 2x downsampled (8kHz)
└─ Scale 4: 4x downsampled (4kHz)

Each Scale Discriminator:
├─ Input: (B, 1, T)
│
├─ Layer 1: Conv1d(1 → 64, kernel=15, stride=1)
│   └─ LeakyReLU(0.2)
│
├─ Layer 2: Conv1d(64 → 128, kernel=41, stride=4, groups=4)
│   └─ LeakyReLU(0.2)
│
├─ Layer 3: Conv1d(128 → 256, kernel=41, stride=4, groups=16)
│   └─ LeakyReLU(0.2)
│
├─ Layer 4: Conv1d(256 → 512, kernel=41, stride=4, groups=64)
│   └─ LeakyReLU(0.2)
│
├─ Layer 5: Conv1d(512 → 1024, kernel=41, stride=4, groups=256)
│   └─ LeakyReLU(0.2)
│
├─ Layer 6: Conv1d(1024 → 1024, kernel=5, stride=1)
│   └─ LeakyReLU(0.2)
│
└─ Output: Conv1d(1024 → 1, kernel=3, stride=1)
    └─ Real/Fake score per time frame

Final Score: Average across all scales and time frames
```

**Why Multi-Scale?**
- **Scale 1 (16kHz)**: Detects high-frequency artifacts (breathiness, consonants)
- **Scale 2 (8kHz)**: Detects mid-frequency issues (formants, vowel quality)
- **Scale 4 (4kHz)**: Detects low-frequency issues (pitch artifacts, prosody)
- **Combined**: Comprehensive realism evaluation across frequency spectrum

**Grouped Convolutions**:
- Reduces parameters while maintaining expressive power
- Each group focuses on different frequency bands

---

## GAN Training Process

### The Adversarial Game

**Generator (G)**: StreamVC model trying to fool the discriminator
**Discriminator (D)**: Classifier trying to detect fake audio

**Training Loop** (each step):
```
1. Generator Forward Pass:
   real_audio (target) ──┐
   source_audio ──────────┤
                          ├─→ [Generator] ──→ fake_audio
   target_speaker ────────┘

2. Discriminator Training:
   real_audio ──→ [D] ──→ score_real (should be ~1)
   fake_audio ──→ [D] ──→ score_fake (should be ~0)
   
   Loss_D = hinge_loss(score_real, 1) + hinge_loss(score_fake, 0)
   Update D parameters

3. Generator Training:
   fake_audio ──→ [D] ──→ score_fake (want it to be ~1)
   
   Loss_G = -score_fake + reconstruction_loss + content_loss + ...
   Update G parameters

4. Repeat
```

**Nash Equilibrium** (ideal state):
```
Early training:
├─ G: Produces noise → D: 100% accurate (spots all fakes)
├─ G: Learns basic audio structure → D: 90% accurate
├─ G: Produces speech-like audio → D: 70% accurate
└─ G: High-quality audio → D: 50% accurate (can't tell difference)
    └─ Training complete!
```

---

## Loss Functions Explained

### 1. Reconstruction Loss (L1 Distance)

**Purpose**: Ensure generated audio matches target waveform point-by-point

**Formula**:
```
L_recon = (1/N) * Σ |y_pred[i] - y_true[i]|
```

**Why L1 instead of L2?**
- L1 is more robust to outliers
- L2 tends to blur sharp transitions (plosives, fricatives)
- L1 preserves temporal details better

**Weight**: 1.0

---

### 2. Content Prediction Loss (Cross-Entropy)

**Purpose**: Force content encoder to learn discrete phonetic units

**Process**:
```
1. Extract content features: (B, 64, T)

2. Project to HuBERT space:
   content_proj = Linear_768(content_features)
   → (B, 768, T)

3. Find nearest HuBERT centroid:
   distances = cdist(content_proj, hubert_centroids)
   pseudo_labels = argmin(distances)
   → (B, T) with values in [0, 99]

4. Compute loss:
   L_content = CrossEntropy(content_logits, pseudo_labels)
```

**Why HuBERT?**
- HuBERT is pretrained on massive speech data
- Its clusters represent phonetic units (phonemes, sub-phonemes)
- Provides strong supervision for content disentanglement

**Weight**: 1.0

---

### 3. STFT Loss (Multi-scale Spectral)

**Purpose**: Match frequency-domain characteristics at multiple resolutions

**Multi-scale STFT**:
```
For FFT_sizes = [512, 1024, 2048]:
    hop_length = FFT_size / 4
    win_length = FFT_size
    
    # Compute STFT
    STFT_real = torch.stft(y_true, n_fft=FFT_size, ...)
    STFT_fake = torch.stft(y_pred, n_fft=FFT_size, ...)
    
    # Magnitude loss
    mag_real = |STFT_real|
    mag_fake = |STFT_fake|
    L_mag = ||mag_fake - mag_real||_1
    
    # Log-magnitude loss (spectral convergence)
    log_mag_real = log(mag_real + 1e-5)
    log_mag_fake = log(mag_fake + 1e-5)
    L_sc = ||log_mag_fake - log_mag_real||_F / ||log_mag_real||_F
    
    Total_per_scale = L_mag + L_sc

L_stft = Σ Total_per_scale for all FFT sizes
```

**Why Multiple Scales?**
- **512 FFT**: High time resolution → Captures rapid transients (consonants)
- **1024 FFT**: Balanced → General spectral envelope
- **2048 FFT**: High frequency resolution → Captures fine harmonic structure

**Why Log-magnitude?**
- Human perception is logarithmic
- Emphasizes low-energy components (breathiness, room tone)
- Prevents model from ignoring quiet sounds

**Weight**: 45.0 (highest weight - spectral quality is crucial!)

---

### 4. Adversarial Loss (Fool the Discriminator)

**Purpose**: Make generated audio indistinguishable from real audio

**Formula**:
```
# Generator wants D(fake) = 1
scores_fake = Discriminator(y_pred)  # Multi-scale outputs
L_adv = -mean(scores_fake)  # Negative for gradient ascent

# Alternative (non-saturating loss):
L_adv = -mean(log(D(y_pred)))
```

**Why Negative Sign?**
- Discriminator outputs high scores for real audio
- Generator wants to maximize discriminator scores on fake audio
- Maximizing = minimizing negative

**Weight**: 1.0

---

### 5. Feature Matching Loss

**Purpose**: Match intermediate discriminator features (perceptual similarity)

**Process**:
```
# Discriminator has multiple layers
# Extract features from each layer

features_real = []
features_fake = []

for layer in Discriminator.layers:
    features_real.append(layer(y_true))
    features_fake.append(layer(y_pred))

# Match features at each layer
L_fm = Σ ||features_fake[i] - features_real[i]||_1
```

**Why This Helps?**
- Direct adversarial loss is hard to optimize (unstable)
- Intermediate features capture perceptual qualities
- Easier to match than fooling discriminator completely
- Stabilizes GAN training

**Weight**: 10.0

---

### Total Loss Summary

```python
# Generator total loss
L_G = (1.0 * L_recon +          # Waveform similarity
       1.0 * L_content +         # Phonetic content
       45.0 * L_stft +           # Spectral quality (most important!)
       1.0 * L_adv +             # Realism
       10.0 * L_fm)              # Perceptual similarity

# Discriminator total loss
L_D = L_D_real + L_D_fake        # Real vs fake classification
```

---

## Training Loop Details

### Single Training Step

```python
def train_step(source_audio, target_audio):
    """
    source_audio: (B, 1, 32000) - Random speaker
    target_audio: (B, 1, 32000) - Random different speaker
    """
    
    # ==================== FEATURE EXTRACTION ====================
    # Extract mel spectrogram from target (for speaker encoder)
    target_mel = mel_extractor(target_audio)
    # → (B, 80, ~100)
    
    # Extract F0 from source (for decoder conditioning)
    f0_features = f0_extractor(source_audio)
    # → (B, 9, ~100)
    
    # Align temporal dimensions (due to boundary effects)
    min_len = min(target_mel.size(2), f0_features.size(2))
    target_mel = target_mel[:, :, :min_len]
    f0_features = f0_features[:, :, :min_len]
    
    
    # ==================== GENERATOR FORWARD ====================
    output_audio, content_logits = model(
        source_audio, target_mel, f0_features
    )
    # output_audio: (B, 1, ~32000)
    # content_logits: (B, 100, ~100)
    
    # Align output with target (due to upsampling artifacts)
    min_audio_len = min(output_audio.size(2), target_audio.size(2))
    output_audio = output_audio[:, :, :min_audio_len]
    target_audio_crop = target_audio[:, :, :min_audio_len]
    
    
    # ==================== COMPUTE LOSSES ====================
    # 1. Content prediction loss
    with torch.no_grad():
        # Get HuBERT pseudo-labels
        content_features = model.content_encoder.get_features(source_audio)
        hubert_targets = compute_hubert_targets(content_features)
    
    loss_content = F.cross_entropy(
        content_logits.transpose(1, 2),  # (B, T, 100)
        hubert_targets                    # (B, T)
    )
    
    # 2. Reconstruction loss
    loss_recon = F.l1_loss(output_audio, target_audio_crop)
    
    # 3. STFT loss
    loss_stft = stft_loss(output_audio, target_audio_crop)
    
    # 4. Discriminator forward (for adversarial + feature matching)
    # Real audio
    disc_real_outputs, disc_real_features = discriminator(
        target_audio_crop, return_features=True
    )
    
    # Fake audio
    disc_fake_outputs, disc_fake_features = discriminator(
        output_audio, return_features=True
    )
    
    # 5. Adversarial loss (generator wants D(fake) = 1)
    loss_adv = 0
    for scale_output in disc_fake_outputs:
        loss_adv += torch.mean((scale_output - 1) ** 2)  # MSE with target=1
    
    # 6. Feature matching loss
    loss_fm = 0
    for real_feat, fake_feat in zip(disc_real_features, disc_fake_features):
        loss_fm += F.l1_loss(fake_feat, real_feat)
    
    
    # ==================== GENERATOR UPDATE ====================
    optimizer_g.zero_grad()
    
    loss_g = (1.0 * loss_recon +
              1.0 * loss_content +
              45.0 * loss_stft +
              1.0 * loss_adv +
              10.0 * loss_fm)
    
    loss_g.backward()
    torch.nn.utils.clip_grad_norm_(generator_params, max_norm=1.0)
    optimizer_g.step()
    
    
    # ==================== DISCRIMINATOR UPDATE ====================
    optimizer_d.zero_grad()
    
    # Real audio (detach to stop gradient flow to generator)
    disc_real = discriminator(target_audio_crop.detach())
    loss_d_real = 0
    for scale_output in disc_real:
        # Hinge loss: max(0, 1 - D(real))
        loss_d_real += torch.mean(torch.relu(1.0 - scale_output))
    
    # Fake audio (detach to stop gradient flow to generator)
    disc_fake = discriminator(output_audio.detach())
    loss_d_fake = 0
    for scale_output in disc_fake:
        # Hinge loss: max(0, 1 + D(fake))
        loss_d_fake += torch.mean(torch.relu(1.0 + scale_output))
    
    loss_d = loss_d_real + loss_d_fake
    
    loss_d.backward()
    torch.nn.utils.clip_grad_norm_(discriminator_params, max_norm=1.0)
    optimizer_d.step()
    
    
    return {
        'loss_g': loss_g.item(),
        'loss_d': loss_d.item(),
        'loss_recon': loss_recon.item(),
        'loss_content': loss_content.item(),
        'loss_stft': loss_stft.item(),
        'loss_adv': loss_adv.item(),
        'loss_fm': loss_fm.item()
    }
```

---

### Gradient Clipping

**Why?**
- GAN training is notoriously unstable
- Large gradients can cause mode collapse or divergence
- Clipping prevents exploding gradients

**Implementation**:
```python
torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)

# This normalizes gradient vector to have L2 norm ≤ 1.0
# If ||grad|| > 1.0:
#     grad = grad / ||grad||
```

---

### Learning Rate Scheduling

**Cosine Annealing with Warmup**:
```python
def get_lr(step, total_steps=200_000, warmup_steps=1000):
    if step < warmup_steps:
        # Linear warmup
        return base_lr * (step / warmup_steps)
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return base_lr * 0.5 * (1 + cos(π * progress))

# Schedule:
# Steps 0-1000:     LR ramps 0 → 1e-4
# Steps 1000-200k:  LR decays 1e-4 → 0 (cosine curve)
```

**Why Warmup?**
- Initially, generator produces random noise
- Discriminator would dominate without warmup
- Gradual warmup allows both to stabilize

---

## Mathematical Formulations

### Content Encoder Forward

```
Input: x ∈ ℝ^(B×1×T), T=32000

h₀ = CausalConv(x)  # (B, 64, T)

For i = 1 to 8:
    h_i = ResBlock_i(h_{i-1})
    h_i = Dropout(h_i, p=0.1)

h_down = StrideConv(h_8, stride=320)  # (B, 64, T/320)

c = h_down  # Content features
logits = Conv1x1(c)  # (B, 100, T/320)

Output: c, logits
```

### Speaker Encoder Forward

```
Input: m ∈ ℝ^(B×80×T_mel), mel spectrogram

h₀ = Conv(m)  # (B, 256, T_mel)

For i = 1 to 4:
    h_i = ResBlock_i(h_{i-1})

# Attention pooling
q = LearnableQuery  # (1, 256)
k = h_4             # (B, 256, T_mel)
v = h_4             # (B, 256, T_mel)

α = softmax(q · k^T / √d)  # (B, T_mel)
s = Σ_t α_t · v_t          # (B, 256)

Output: s (speaker embedding)
```

### Decoder Forward with FiLM

```
Input:
├─ c: content features (B, 64, T_c)
├─ f: F0 features (B, 9, T_c)
└─ s: speaker embedding (B, 256)

h₀ = CausalConv(concat(c, f))  # (B, 40, T_c)

For i = 1 to 12:
    # Residual block
    h_i = ResBlock_i(h_{i-1})
    
    # FiLM conditioning
    γ_i = FC_scale(s)   # (B, 40)
    β_i = FC_shift(s)   # (B, 40)
    h_i = (h_i ⊙ (1 + γ_i)) + β_i

h_up = TransposeConv(h_12, stride=320)  # (B, 40, T)

y = Tanh(Conv1x1(h_up))  # (B, 1, T)

Output: y (audio waveform)
```

### Discriminator (Single Scale)

```
Input: x ∈ ℝ^(B×1×T)

h₁ = LeakyReLU(Conv(x, 1→64))
h₂ = LeakyReLU(Conv(h₁, 64→128, stride=4))
h₃ = LeakyReLU(Conv(h₂, 128→256, stride=4))
h₄ = LeakyReLU(Conv(h₃, 256→512, stride=4))
h₅ = LeakyReLU(Conv(h₄, 512→1024, stride=4))
h₆ = LeakyReLU(Conv(h₅, 1024→1024))
score = Conv(h₆, 1024→1)  # (B, 1, T')

Output: score, [h₁, h₂, h₃, h₄, h₅, h₆] (for feature matching)
```

### Hinge Loss (Discriminator)

```
For real samples:
L_D_real = 𝔼[max(0, 1 - D(x_real))]

For fake samples:
L_D_fake = 𝔼[max(0, 1 + D(x_fake))]

Total:
L_D = L_D_real + L_D_fake

# Optimal discriminator outputs:
# D(x_real) = 1
# D(x_fake) = -1
```

### Feature Matching Loss

```
Let φ_i(x) = features from i-th layer of discriminator

L_FM = Σ_i (1/N_i) ||φ_i(x_real) - φ_i(x_fake)||₁

Where N_i = dimensionality of layer i features
```

---

## Pipeline Flow

### Complete VoiceCloneify Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: English Speech                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  MODULE 1: Automatic Speech Recognition (ASR)               │
│  ├─ Model: Distil-Whisper (distil-small.en)                 │
│  ├─ Streaming: 30ms chunks with 300ms context               │
│  └─ Output: English transcription                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  MODULE 2: Machine Translation (MT)                          │
│  ├─ Model: NLLB-200-distilled-600M                          │
│  ├─ Languages: eng_Latn → fra_Latn                          │
│  └─ Output: French translation                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  MODULE 3: Text-to-Speech (TTS)                              │
│  ├─ Model: SpeechT5 + HiFiGAN vocoder                        │
│  ├─ Speaker: Default embedding (generic voice)               │
│  └─ Output: French speech (generic voice)                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  MODULE 4: Voice Conversion (StreamVC) ← TRAINING NOW        │
│  ├─ Inputs:                                                  │
│  │   ├─ Source: French TTS audio (generic voice)            │
│  │   └─ Target: Reference audio (desired voice)             │
│  ├─ Process:                                                 │
│  │   ├─ Extract content (what is said)                      │
│  │   ├─ Extract speaker (target voice characteristics)      │
│  │   ├─ Extract F0 (pitch/prosody)                          │
│  │   └─ Synthesize: content + target_speaker + F0           │
│  └─ Output: French speech (target speaker's voice)          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│             OUTPUT: French Speech (Voice Cloned)             │
└─────────────────────────────────────────────────────────────┘
```

### Latency Budget

```
Component          | Latency    | Type
-------------------|------------|------------------
ASR (Distil-Whis)  | ~50ms      | Streaming
MT (NLLB-200)      | ~200ms     | Sentence-level
TTS (SpeechT5)     | ~300ms     | Sentence-level
StreamVC           | 60ms       | Streaming (2-frame)
-------------------|------------|------------------
Total              | ~610ms     | Near real-time
```

**Note**: ASR and StreamVC are truly streaming. MT and TTS require full sentences.

---

## Training Configuration

### Current Setup

```yaml
Hardware:
  GPU: NVIDIA RTX 4060 Laptop (8GB VRAM)
  CUDA: 12.8
  Precision: FP32 (FP16 for HuBERT extraction)

Data:
  Datasets: LibriTTS + LJSpeech
  Total Files: 33,544 audio files
  Size: ~10GB (wav/flac)
  Speakers: ~500 unique speakers
  Segment Length: 2 seconds (32,000 samples)

Model:
  Content Encoder: 8 layers, hidden_dim=64
  Speaker Encoder: 4 layers, embedding_dim=256
  Decoder: 12 layers, hidden_dim=40
  Discriminator: 3 scales, 4 layers each
  Total Parameters: ~15M

Training:
  Batch Size: 16
  Epochs: 200
  Steps per Epoch: 2,096
  Total Steps: 419,200
  Optimizer: AdamW
  Learning Rate: 1e-4
  Weight Decay: 1e-6
  Gradient Clipping: 1.0
  Scheduler: Cosine (warmup=1000 steps)

Losses:
  Reconstruction: 1.0
  Content: 1.0
  STFT: 45.0
  Adversarial: 1.0
  Feature Matching: 10.0

Checkpointing:
  Save Interval: 5,000 steps
  Keep Last: 5 checkpoints
  Directory: ./checkpoints/streamvc/

Logging:
  TensorBoard: Enabled
  Log Interval: 100 steps
  Directory: ./logs/streamvc/
```

### Training Timeline

```
Estimated Timeline (RTX 4060):
├─ Per Step: ~500ms
├─ Per Epoch: ~18 minutes
├─ Total: ~60 hours (2.5 days)
└─ With interruptions: 3-4 days

Progress Indicators:
├─ Loss_G should decrease: 130 → 20-30
├─ Loss_D should stabilize: 0.8 → 0.5-0.7
├─ Loss_Recon should decrease: 0.11 → 0.02-0.05
└─ Discriminator accuracy: ~50% at convergence
```

---

## Advanced Topics

### Stop Gradient to Content Encoder

**Implementation**:
```python
content_features_stopped = content_features.detach()
output = decoder(content_features_stopped, speaker_emb, f0)
```

**Why?**
- Prevents decoder reconstruction loss from affecting content encoder
- Content encoder only learns from HuBERT supervision
- Ensures content encoder learns speaker-invariant features
- Critical for disentanglement!

### Causal Convolutions

**Standard Conv1d**:
```
h[t] = Σ_{k=-K/2}^{K/2} w[k] · x[t+k]
# Uses future samples (t+1, t+2, ..., t+K/2)
```

**Causal Conv1d**:
```
h[t] = Σ_{k=0}^{K-1} w[k] · x[t-k]
# Only uses past samples (t, t-1, ..., t-K+1)
```

**Implementation**:
```python
padding = (kernel_size - 1) * dilation
output = F.conv1d(input, weight, padding=padding)
output = output[:, :, :-padding]  # Trim future frames
```

### Lookahead Frames

**Concept**:
- StreamVC allows 2-frame lookahead (40ms)
- Improves quality while maintaining low latency
- Trade-off: latency vs quality

**Implementation**:
```python
# In decoder, before output projection
if lookahead_frames > 0:
    # Zero-pad future frames
    padding = (0, lookahead_frames)
    x = F.pad(x, padding)
    
    # Process with lookahead context
    x = output_projection(x)
    
    # Trim to original length
    x = x[:, :, :-lookahead_frames]
```

---

## Troubleshooting & Tips

### Common Issues During Training

**1. Discriminator Dominates (Loss_D → 0, Loss_G → ∞)**
```
Solution:
- Reduce discriminator learning rate
- Increase feature matching weight
- Train generator 2x more than discriminator
```

**2. Mode Collapse (Loss_G oscillates wildly)**
```
Solution:
- Increase gradient clipping (0.5 instead of 1.0)
- Add noise to discriminator inputs
- Use spectral normalization in discriminator
```

**3. Checkerboard Artifacts in Audio**
```
Cause: Transposed convolution in upsampling
Solution:
- Use resize + conv instead of transposed conv
- Ensure kernel_size divisible by stride
```

**4. Poor Content Disentanglement**
```
Solution:
- Increase content loss weight
- Use more HuBERT clusters (200 instead of 100)
- Ensure stop gradient is working
```

### Monitoring Training Health

**Healthy Training**:
```
Epoch 0:   Loss_G=130, Loss_D=0.87, Recon=0.11
Epoch 10:  Loss_G=80,  Loss_D=0.65, Recon=0.08
Epoch 50:  Loss_G=45,  Loss_D=0.52, Recon=0.05
Epoch 100: Loss_G=30,  Loss_D=0.48, Recon=0.03
Epoch 200: Loss_G=25,  Loss_D=0.45, Recon=0.02
```

**Unhealthy Training**:
```
# Mode collapse
Epoch 20: Loss_G=500, Loss_D=0.01 ← Discriminator too strong

# Generator collapse
Epoch 20: Loss_G=5, Loss_D=1.5 ← Generator fooling too easily

# Oscillation
Epoch 20: Loss_G=200 → 50 → 180 → 60 ← Unstable
```

---

## References

**StreamVC Paper**:
- Title: "StreamVC: Real-Time Low-Latency Voice Conversion"
- Key Ideas: Causal architecture, FiLM conditioning, HuBERT content

**Related Work**:
- HuBERT: Self-supervised speech representation learning
- FiLM: Feature-wise Linear Modulation for conditional generation
- Multi-Scale Discriminators: From HiFi-GAN vocoder
- Yin Algorithm: Robust pitch estimation

---

## Appendix: Code Snippets

### Creating the Model

```python
import yaml
from streamvc.model import StreamVC

# Load config
with open('configs/streamvc_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize model
model = StreamVC(config).cuda()

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")  # ~15M
```

### Inference (After Training)

```python
import torch
from streamvc.model import StreamVC

# Load trained model
model = StreamVC.load_from_checkpoint('checkpoints/best.pt')
model.eval()

# Load audio
source_audio = load_audio('french_tts.wav')  # (1, 32000)
target_audio = load_audio('target_speaker.wav')  # (1, 32000)

# Extract target speaker embedding
with torch.no_grad():
    target_mel = mel_extractor(target_audio)
    speaker_emb = model.speaker_encoder(target_mel)

# Convert
with torch.no_grad():
    f0_features = f0_extractor(source_audio)
    converted_audio = model.convert(
        source_audio, speaker_emb, f0_features
    )

# Save
save_audio('output_cloned.wav', converted_audio)
```

### Visualizing Training

```bash
# Launch TensorBoard
tensorboard --logdir ./logs/streamvc --port 6006

# View in browser
http://localhost:6006

# Metrics to watch:
# - losses/generator (should decrease)
# - losses/discriminator (should stabilize ~0.5)
# - losses/reconstruction (should decrease)
# - losses/stft (should decrease)
```

---

**Document Version**: 1.0  
**Last Updated**: January 17, 2026  
**Status**: Training in progress (Epoch 0, ~3% complete)
