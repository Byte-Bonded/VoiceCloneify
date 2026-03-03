#!/bin/bash
set -e
VENV="/home/pranaav/Work/projects/SEM6/VoiceCloneify/.venv"
LOG="/tmp/setup_env.log"
source "$VENV/bin/activate"

echo "=== ENV SETUP $(date) ===" | tee "$LOG"
echo "Python: $(python --version)" | tee -a "$LOG"

# 1. Check what NVIDIA packages are already installed
echo "--- Installed NVIDIA packages ---" | tee -a "$LOG"
pip list 2>/dev/null | grep -i nvidia | tee -a "$LOG"

# 2. Install remaining NVIDIA deps + torch in one shot
echo "--- Installing torch + remaining deps ---" | tee -a "$LOG"
pip install --timeout 600 --retries 5 \
  torch==2.6.0+cu124 \
  torchaudio==2.6.0+cu124 \
  torchvision==0.21.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124 2>&1 | tee -a "$LOG"

echo "--- Torch install done, rc=$? ---" | tee -a "$LOG"

# 3. Install other project dependencies
echo "--- Installing other deps ---" | tee -a "$LOG"
pip install --timeout 300 --retries 3 \
  transformers accelerate librosa soundfile pyyaml tqdm tensorboard \
  scipy scikit-learn pandas matplotlib omegaconf praat-parselmouth \
  speechbrain faster-whisper gradio datasets huggingface_hub \
  pesq pystoi resemblyzer webrtcvad 2>&1 | tee -a "$LOG"

echo "--- Other deps done, rc=$? ---" | tee -a "$LOG"

# 4. Verify installation
echo "=== VERIFICATION ===" | tee -a "$LOG"
python -c "
import torch
print(f'torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
import torchaudio; print(f'torchaudio {torchaudio.__version__}')
import transformers; print(f'transformers {transformers.__version__}')
import librosa; print(f'librosa {librosa.__version__}')
import yaml; print('pyyaml OK')
import tensorboard; print('tensorboard OK')

# Test model import
import sys; sys.path.insert(0, 'streamvc')
from model import StreamVC
import yaml as y
with open('configs/streamvc_config.yaml') as f:
    cfg = y.safe_load(f)
m = StreamVC(cfg)
params = sum(p.numel() for p in m.parameters())
print(f'StreamVC params: {params:,}')

# Quick forward pass
B, T = 2, 32000
src = torch.randn(B, 1, T)
mel = torch.randn(B, 80, 100)
f0 = torch.randn(B, 9, 100)
out, logits = m(src, mel, f0)
print(f'Forward pass OK: out={out.shape}, logits={logits.shape}')
print('ALL_CHECKS_PASSED')
" 2>&1 | tee -a "$LOG"

echo "=== SETUP COMPLETE $(date) ===" | tee -a "$LOG"
