# VoiceCloneify 🎙️→🌍→🗣️

Real-time voice cloning with translation: **English → French** with target speaker's voice.

**Pipeline:** ASR → MT → TTS → StreamVC

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
cd VoiceCloneify

# Run setup script
chmod +x setup.sh
./setup.sh

# Or manually install
pip install torch torchaudio transformers speechbrain faster-whisper librosa soundfile gradio pyyaml omegaconf praat-parselmouth scikit-learn tqdm
```

### 2. Download Datasets

```bash
# Download required datasets (~19GB total)
python data/download_datasets.py --data-dir ./data --datasets all

# Or selectively (for prototype: LJSpeech + LibriTTS = ~9GB)
python data/download_datasets.py --data-dir ./data --datasets libritts ljspeech
```

### 3. Extract HuBERT Centroids

```bash
# Required for StreamVC training
python data/extract_hubert_centroids.py \
    --audio-dir ./data/libritts/LibriTTS/train-clean-100 \
    --output-path ./data/hubert_centroids_100.pkl \
    --max-samples 10000
```

---

## 🎯 Usage

### Run Complete Pipeline

```bash
# With voice conversion (requires trained StreamVC)
python run_pipeline.py \
    --input your_english_audio.wav \
    --target-speaker target_speaker_reference.wav \
    --output translated_cloned_output.wav \
    --streamvc-checkpoint ./checkpoints/streamvc/best.pt

# Without voice conversion (TTS only - works immediately)
python run_pipeline.py \
    --input your_english_audio.wav \
    --output translated_output.wav
```

### Train StreamVC

```bash
# Start training
python streamvc/train.py --config ./configs/streamvc_config.yaml

# Monitor progress
tensorboard --logdir ./logs/streamvc

# Resume from checkpoint
python streamvc/train.py \
    --config ./configs/streamvc_config.yaml \
    --resume ./checkpoints/streamvc/checkpoint_step_50000.pt
```

### Test Individual Components

```bash
# ASR (Speech → Text)
python asr/streaming_asr.py --audio input.wav

# MT (English → French)
python mt/translator.py --text "Hello, how are you?"

# TTS (French Text → Speech)
python tts/text_to_speech.py --text "Bonjour" --output output.wav
```

---

## 📁 Project Structure

```
VoiceCloneify/
├── asr/streaming_asr.py      # Distil-Whisper ASR
├── mt/translator.py           # NLLB-200 MT (EN→FR)
├── tts/text_to_speech.py     # SpeechT5 TTS
├── streamvc/                  # Voice Conversion
│   ├── model.py               # StreamVC architecture
│   ├── train.py               # Training script
│   ├── f0_extractor.py        # F0 + whitening
│   └── ...
├── data/                      # Data scripts
├── configs/                   # Configuration files
├── run_pipeline.py            # Main pipeline
└── setup.sh                   # Setup script
```

---

## 📊 Models Used

| Component | Model | Size | Purpose |
|-----------|-------|------|---------|
| ASR | Distil-Whisper small.en | 166M | Speech → Text (EN) |
| MT | NLLB-200-600M | 600M | Translation (EN→FR) |
| TTS | SpeechT5 + HiFi-GAN | 250MB | Text → Speech (FR) |
| StreamVC | Custom | ~50M | Voice Conversion |

**Total:** ~1.5GB pretrained models + datasets

---

## 🎓 Training Details

### StreamVC Training

**Prerequisites:**
- Downloaded datasets (LibriTTS + VCTK)
- HuBERT centroids extracted
- GPU (recommended)

**Training command:**
```bash
python streamvc/train.py --config ./configs/streamvc_config.yaml
```

**Config adjustments** (in `configs/streamvc_config.yaml`):
- `batch_size`: 32 (reduce to 16 or 8 if OOM)
- `num_epochs`: 200
- `learning_rate`: 1e-4

**Expected training time:** 2-5 days on single GPU (V100/A100)

---

## 🔧 Configuration

Edit `configs/pipeline_config.yaml`:

```yaml
pipeline:
  language_pair:
    source: "eng_Latn"
    target: "fra_Latn"
  sample_rate: 16000

asr:
  model_name: "distil-whisper/distil-small.en"
  
mt:
  model_name: "facebook/nllb-200-distilled-600M"
  
tts:
  model_name: "microsoft/speecht5_tts"
```

---

## 🐛 Troubleshooting

**Out of Memory:**
```bash
# Reduce batch size in streamvc_config.yaml
batch_size: 16  # or 8

# Use CPU (slower)
python run_pipeline.py --input audio.wav --device cpu
```

**Missing Centroids:**
```bash
python data/extract_hubert_centroids.py \
    --audio-dir ./data/libritts/LibriTTS/train-clean-100 \
    --output-path ./data/hubert_centroids_100.pkl
```

---

## 📚 References

- [StreamVC Paper](https://arxiv.org/abs/2401.11053)
- [Distil-Whisper](https://huggingface.co/distil-whisper)
- [NLLB-200](https://ai.meta.com/research/no-language-left-behind/)
- [SpeechT5](https://arxiv.org/abs/2110.07205)

---

## 📝 License

MIT License

---

## ⭐ Star this repo if you find it useful! 
