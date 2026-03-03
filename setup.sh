#!/bin/bash
# Quick setup script for VoiceCloneify

echo "================================================"
echo "  VoiceCloneify Setup Script"
echo "================================================"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Install dependencies
echo ""
echo "[1/4] Installing Python dependencies..."
pip install -e . || pip install torch torchaudio transformers speechbrain faster-whisper \
    librosa soundfile gradio pyyaml omegaconf praat-parselmouth scikit-learn tqdm

# Create necessary directories
echo ""
echo "[2/4] Creating directories..."
mkdir -p data checkpoints logs outputs

# Download lightweight datasets (optional)
echo ""
echo "[3/4] Dataset download (optional)..."
read -p "Download datasets now? This will download ~19GB (LibriTTS + LJSpeech + VCTK). (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    python data/download_datasets.py --data-dir ./data --datasets all
fi

# Extract HuBERT centroids (optional)
echo ""
echo "[4/4] HuBERT centroids extraction (optional)..."
read -p "Extract HuBERT centroids now? Requires LibriTTS dataset. (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    if [ -d "./data/libritts/LibriTTS/train-clean-100" ]; then
        python data/extract_hubert_centroids.py \
            --audio-dir ./data/libritts/LibriTTS/train-clean-100 \
            --output-path ./data/hubert_centroids_100.pkl \
            --max-samples 10000
    else
        echo "Error: LibriTTS not found. Please download datasets first."
    fi
fi

echo ""
echo "================================================"
echo "  Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Test ASR: python asr/streaming_asr.py --audio your_audio.wav"
echo "  2. Test MT:  python mt/translator.py --text 'Hello world'"
echo "  3. Test TTS: python tts/text_to_speech.py --text 'Bonjour' --output out.wav"
echo "  4. Train StreamVC: python streamvc/train.py --config configs/streamvc_config.yaml"
echo "  5. Run pipeline: python run_pipeline.py --input input.wav --output output.wav"
echo ""
