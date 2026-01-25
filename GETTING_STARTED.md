# Getting Started with VoiceCloneify

This guide walks you through setting up and using VoiceCloneify step-by-step.

---

## Prerequisites

- **Python:** 3.9, 3.10, or 3.11
- **GPU:** NVIDIA GPU with CUDA (recommended for training, optional for inference)
- **Storage:** At least 25GB free space (for datasets + models)
- **RAM:** 16GB+ recommended

---

## Step-by-Step Setup

### 1. Install Dependencies

```bash
# Option A: Using setup script (recommended)
chmod +x setup.sh
./setup.sh

# Option B: Manual installation
pip install -r requirements.txt

# Option C: Install core packages only
pip install torch torchaudio transformers faster-whisper librosa soundfile
```

### 2. Verify Installation

```bash
python test_installation.py
```

You should see:
```
✓ All tests passed! Installation is working correctly.
```

---

## Quick Start: Testing Without Training

You can test the pipeline immediately using pretrained models (no StreamVC training required):

### Test Individual Components

```bash
# 1. Test ASR (requires an English audio file)
echo "Hello, this is a test" | say -o test_input.aiff  # macOS
# or record your own: test_input.wav
python asr/streaming_asr.py --audio test_input.wav

# 2. Test MT (translation)
python mt/translator.py --text "Hello, how are you?"

# 3. Test TTS (synthesis)
python tts/text_to_speech.py \
    --text "Bonjour, comment allez-vous?" \
    --output test_output.wav
```

### Run Full Pipeline (ASR → MT → TTS)

```bash
# Without voice conversion (works immediately)
python run_pipeline.py \
    --input test_input.wav \
    --output test_output.wav
```

**Expected output:**
- Input: English speech
- Output: French speech (neutral voice from TTS)

---

## Training StreamVC for Voice Cloning

To add voice cloning capability, you need to train the StreamVC model.

### Step 1: Download Datasets

```bash
# Download all datasets (~19GB)
python data/download_datasets.py --data-dir ./data --datasets all

# Or minimal setup (LibriTTS + LJSpeech ~9GB)
python data/download_datasets.py --data-dir ./data --datasets libritts ljspeech
```

**Datasets:**
- **LibriTTS:** Multi-speaker TTS, HuBERT features (~6GB)
- **LJSpeech:** Single-speaker TTS (~2.6GB)
- **VCTK:** Voice conversion training (~11GB)

### Step 2: Extract HuBERT Centroids

```bash
# This creates pseudo-labels for StreamVC content encoder
python data/extract_hubert_centroids.py \
    --audio-dir ./data/libritts/LibriTTS/train-clean-100 \
    --output-path ./data/hubert_centroids_100.pkl \
    --n-clusters 100 \
    --max-samples 10000
```

**Note:** Use `--max-samples 10000` for quick testing, remove for full quality.

### Step 3: Train StreamVC

```bash
# Start training
python streamvc/train.py --config ./configs/streamvc_config.yaml

# Monitor progress
tensorboard --logdir ./logs/streamvc
```

**Training time:**
- **Single GPU (V100/A100):** 2-5 days
- **Reduce batch size if OOM:** Edit `configs/streamvc_config.yaml`, set `batch_size: 16` or `8`

### Step 4: Use Trained Model

```bash
# Run pipeline with voice conversion
python run_pipeline.py \
    --input english_speech.wav \
    --target-speaker target_voice_reference.wav \
    --output cloned_french_output.wav \
    --streamvc-checkpoint ./checkpoints/streamvc/best.pt
```

---

## Configuration

### Pipeline Settings

Edit `configs/pipeline_config.yaml`:

```yaml
pipeline:
  language_pair:
    source: "eng_Latn"  # English
    target: "fra_Latn"  # French
  sample_rate: 16000

asr:
  model_name: "distil-whisper/distil-small.en"
  device: "cuda"  # Change to "cpu" if no GPU

mt:
  model_name: "facebook/nllb-200-distilled-600M"
  
tts:
  model_name: "microsoft/speecht5_tts"
  default_speaker: "awb"  # Neutral male voice
```

### StreamVC Training Settings

Edit `configs/streamvc_config.yaml`:

```yaml
training:
  batch_size: 32  # Reduce if OOM
  num_epochs: 200
  learning_rate: 1.0e-4

data:
  train_datasets:
    - libritts/LibriTTS/train-clean-100
    - vctk/VCTK-Corpus-0.92/wav48_silence_trimmed
```

---

## Usage Examples

### Example 1: English Speech → French Speech (No Cloning)

```bash
# Input: your_speech.wav (English)
# Output: french_output.wav (French, neutral voice)

python run_pipeline.py \
    --input your_speech.wav \
    --output french_output.wav
```

### Example 2: English Speech → French Speech (With Cloning)

```bash
# Input: your_speech.wav (English)
# Target: celebrity_voice.wav (reference for voice cloning)
# Output: french_cloned_output.wav (French, in celebrity's voice)

python run_pipeline.py \
    --input your_speech.wav \
    --target-speaker celebrity_voice.wav \
    --output french_cloned_output.wav \
    --streamvc-checkpoint ./checkpoints/streamvc/best.pt
```

### Example 3: Training From Scratch

```bash
# 1. Download data
python data/download_datasets.py --data-dir ./data --datasets all

# 2. Extract centroids
python data/extract_hubert_centroids.py \
    --audio-dir ./data/libritts/LibriTTS/train-clean-100 \
    --output-path ./data/hubert_centroids_100.pkl

# 3. Train (takes days)
python streamvc/train.py --config ./configs/streamvc_config.yaml

# 4. Use trained model
python run_pipeline.py \
    --input input.wav \
    --target-speaker target.wav \
    --output output.wav \
    --streamvc-checkpoint ./checkpoints/streamvc/checkpoint_epoch_199.pt
```

---

## Troubleshooting

### Issue: Out of Memory during training

**Solution:**
```yaml
# Edit configs/streamvc_config.yaml
training:
  batch_size: 16  # or 8
```

### Issue: Slow inference

**Solution:**
- Use GPU: `--device cuda`
- Use smaller ASR model: Set `model_name: "distil-whisper/distil-small.en"` (already default)

### Issue: HuBERT centroids not found

**Solution:**
```bash
python data/extract_hubert_centroids.py \
    --audio-dir ./data/libritts/LibriTTS/train-clean-100 \
    --output-path ./data/hubert_centroids_100.pkl
```

### Issue: Missing datasets

**Solution:**
```bash
python data/download_datasets.py --data-dir ./data --datasets all
```

---

## What's Next?

- **Explore configs:** Customize languages, models, and hyperparameters
- **Train StreamVC:** For high-quality voice cloning
- **Fine-tune models:** Adapt ASR/MT/TTS to your domain
- **Deploy:** Create a web interface or API (see `deploy/`)

---

## Getting Help

- **Check README.md** for full documentation
- **Run test_installation.py** to diagnose setup issues
- **Review configs/** for all configuration options

---

## Quick Reference

| Task | Command |
|------|---------|
| Install | `./setup.sh` or `pip install -r requirements.txt` |
| Test | `python test_installation.py` |
| ASR | `python asr/streaming_asr.py --audio input.wav` |
| MT | `python mt/translator.py --text "Hello"` |
| TTS | `python tts/text_to_speech.py --text "Bonjour" --output out.wav` |
| Pipeline | `python run_pipeline.py --input in.wav --output out.wav` |
| Train | `python streamvc/train.py --config configs/streamvc_config.yaml` |

---

**Happy Voice Cloning! 🎙️→🌍→🗣️**
