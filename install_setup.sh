#!/bin/bash
# ==============================================================================
# VoiceCloneify — Environment Setup Script (SLURM Cluster)
# ==============================================================================
# This script creates a conda environment with all dependencies and prepares
# the directory structure for training and evaluation.
#
# Usage:
#   chmod +x install_setup.sh
#   ./install_setup.sh
#
# Cluster: ASAI Compute (SLURM workq partition, RTX 6000 Ada, CUDA 12.x)
# ==============================================================================

set -euo pipefail

# ---- Configuration ----------------------------------------------------------
ENV_NAME="voicecloneify"
PYTHON_VERSION="3.10"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAMBA_BIN="/dist_home/common-apps/mamba/bin/mamba"
CONDA_BIN="/dist_home/common-apps/mamba/bin/conda"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}================================================================${NC}"
echo -e "${CYAN}  VoiceCloneify — Environment Setup${NC}"
echo -e "${CYAN}================================================================${NC}"
echo ""
echo "  Repo directory : ${REPO_DIR}"
echo "  Environment    : ${ENV_NAME}"
echo "  Python version : ${PYTHON_VERSION}"
echo ""

# ---- Step 0: Detect package manager ----------------------------------------
echo -e "${YELLOW}[0/7] Detecting package manager...${NC}"

if command -v mamba &>/dev/null; then
    PKG_MGR="mamba"
elif [[ -x "${MAMBA_BIN}" ]]; then
    PKG_MGR="${MAMBA_BIN}"
elif command -v conda &>/dev/null; then
    PKG_MGR="conda"
elif [[ -x "${CONDA_BIN}" ]]; then
    PKG_MGR="${CONDA_BIN}"
else
    echo -e "${RED}ERROR: Neither mamba nor conda found.${NC}"
    echo "  Please install mamba/conda or adjust MAMBA_BIN in this script."
    exit 1
fi
echo -e "  Using: ${GREEN}${PKG_MGR}${NC}"

# ---- Step 1: Load CUDA modules ---------------------------------------------
echo ""
echo -e "${YELLOW}[1/7] Loading CUDA modules...${NC}"

if command -v module &>/dev/null; then
    module purge 2>/dev/null || true
    # Load CUDA 12.3 (closest available to driver CUDA 12.4)
    if module avail cuda/12.3 2>&1 | grep -q "cuda/12.3"; then
        module load cuda/12.3
        echo -e "  ${GREEN}✓${NC} Loaded cuda/12.3"
    elif module avail cuda/12.5 2>&1 | grep -q "cuda/12.5"; then
        module load cuda/12.5
        echo -e "  ${GREEN}✓${NC} Loaded cuda/12.5"
    elif module avail cuda/12.2 2>&1 | grep -q "cuda/12.2"; then
        module load cuda/12.2
        echo -e "  ${GREEN}✓${NC} Loaded cuda/12.2"
    else
        echo -e "  ${YELLOW}⚠${NC} No CUDA 12.x module found. Continuing with system CUDA."
    fi

    # Load cuDNN
    if module avail cudnn/12/9.8 2>&1 | grep -q "cudnn/12/9.8"; then
        module load cudnn/12/9.8
        echo -e "  ${GREEN}✓${NC} Loaded cudnn/12/9.8"
    elif module avail cudnn/12/9.3 2>&1 | grep -q "cudnn/12/9.3"; then
        module load cudnn/12/9.3
        echo -e "  ${GREEN}✓${NC} Loaded cudnn/12/9.3"
    fi
else
    echo -e "  ${YELLOW}⚠${NC} 'module' command not available. Skipping module loads."
fi

echo -e "  CUDA: $(nvcc --version 2>/dev/null | grep 'release' | awk '{print $6}' || echo 'not found')"

# ---- Step 2: Create conda environment --------------------------------------
echo ""
echo -e "${YELLOW}[2/7] Creating conda environment '${ENV_NAME}'...${NC}"

# Check if env already exists
if ${PKG_MGR} env list 2>/dev/null | grep -qw "${ENV_NAME}"; then
    echo -e "  ${YELLOW}⚠${NC} Environment '${ENV_NAME}' already exists."
    read -p "  Recreate it? This will delete the existing env. (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "  Removing old environment..."
        ${PKG_MGR} env remove -n "${ENV_NAME}" -y
    else
        echo "  Keeping existing environment. Will update packages."
    fi
fi

# Create env if it doesn't exist
if ! ${PKG_MGR} env list 2>/dev/null | grep -qw "${ENV_NAME}"; then
    ${PKG_MGR} create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
    echo -e "  ${GREEN}✓${NC} Created environment '${ENV_NAME}'"
else
    echo -e "  ${GREEN}✓${NC} Environment '${ENV_NAME}' ready"
fi

# ---- Step 3: Activate environment & install packages ------------------------
echo ""
echo -e "${YELLOW}[3/7] Installing packages...${NC}"

# We need to source conda/mamba init for activate to work in a script
CONDA_BASE=$($PKG_MGR info --base 2>/dev/null || echo "/dist_home/common-apps/mamba")
source "${CONDA_BASE}/etc/profile.d/conda.sh"
if [[ -f "${CONDA_BASE}/etc/profile.d/mamba.sh" ]]; then
    source "${CONDA_BASE}/etc/profile.d/mamba.sh"
fi

conda activate "${ENV_NAME}" 2>/dev/null || mamba activate "${ENV_NAME}" 2>/dev/null || {
    echo -e "${RED}ERROR: Could not activate environment. Try manually:${NC}"
    echo "  conda activate ${ENV_NAME}"
    exit 1
}

echo -e "  Python: $(python --version)"
echo -e "  pip:    $(pip --version | awk '{print $2}')"
echo ""

# 3a. Install PyTorch with CUDA support (cu121 wheels are compatible with CUDA 12.x drivers)
echo "  [3a] Installing PyTorch + torchaudio (CUDA 12.1 wheels)..."
pip install --upgrade pip setuptools wheel
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA works
python -c "
import torch
print(f'  PyTorch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'  CUDA version: {torch.version.cuda}')
" || echo -e "  ${YELLOW}⚠${NC} PyTorch CUDA check failed (may work on compute nodes)"

# 3b. Install project and all dependencies
echo ""
echo "  [3b] Installing VoiceCloneify and dependencies..."
cd "${REPO_DIR}"
pip install -e ".[dev]" 2>/dev/null || pip install -e . 2>/dev/null || {
    echo "  Falling back to requirements.txt..."
    pip install -r requirements.txt
}

# 3c. Install additional packages that may not be in requirements.txt
echo ""
echo "  [3c] Installing evaluation & utility packages..."
pip install jiwer sacrebleu pesq resemblyzer scikit-learn matplotlib seaborn

echo -e "  ${GREEN}✓${NC} All packages installed"

# ---- Step 4: Create directory structure -------------------------------------
echo ""
echo -e "${YELLOW}[4/7] Creating directory structure...${NC}"

cd "${REPO_DIR}"

# Training outputs
mkdir -p checkpoints/streamvc
mkdir -p logs/streamvc

# Evaluation outputs  
mkdir -p outputs/eval/asr
mkdir -p outputs/eval/mt
mkdir -p outputs/eval/tts
mkdir -p outputs/eval/streamvc
mkdir -p outputs/eval/reports

# Pipeline inference outputs
mkdir -p outputs/pipeline

# Data directories
mkdir -p data/libritts
mkdir -p data/ljspeech
mkdir -p data/vctk
mkdir -p data/speaker_references
mkdir -p data/test_sets

# SLURM logs
mkdir -p logs/slurm

echo -e "  ${GREEN}✓${NC} Directory structure created:"
echo "      checkpoints/streamvc/    — model checkpoints"
echo "      logs/streamvc/           — TensorBoard logs"
echo "      logs/slurm/              — SLURM job logs"
echo "      outputs/eval/            — evaluation results"
echo "      outputs/pipeline/        — pipeline inference outputs"
echo "      data/                    — datasets & references"

# ---- Step 5: Quick verification ---------------------------------------------
echo ""
echo -e "${YELLOW}[5/7] Running quick verification...${NC}"

python "${REPO_DIR}/quick_test.py" && echo -e "  ${GREEN}✓${NC} Core imports OK" || {
    echo -e "  ${YELLOW}⚠${NC} Some imports failed — check output above"
}

# ---- Step 6: Download datasets (optional) -----------------------------------
echo ""
echo -e "${YELLOW}[6/7] Dataset download (optional)...${NC}"
echo "  Available datasets:"
echo "    - LibriTTS train-clean-100  (~6 GB)  — Required for StreamVC training"
echo "    - LJSpeech 1.1              (~2.6 GB) — Optional single-speaker TTS data"
echo "    - VCTK 0.92                 (~11 GB) — Optional multi-speaker VC data"
echo ""
read -p "  Download datasets now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "  Downloading all datasets..."
    python "${REPO_DIR}/data/download_datasets.py" --data-dir "${REPO_DIR}/data" --datasets all
fi

# ---- Step 7: Extract HuBERT centroids (optional) ----------------------------
echo ""
echo -e "${YELLOW}[7/7] HuBERT centroids extraction (optional)...${NC}"

if [[ -f "${REPO_DIR}/data/hubert_centroids_100.pkl" ]]; then
    echo -e "  ${GREEN}✓${NC} HuBERT centroids already exist at data/hubert_centroids_100.pkl"
else
    if [[ -d "${REPO_DIR}/data/libritts/LibriTTS/train-clean-100" ]]; then
        read -p "  Extract HuBERT centroids? Requires GPU (~30 min). (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python "${REPO_DIR}/data/extract_hubert_centroids.py" \
                --audio-dir "${REPO_DIR}/data/libritts/LibriTTS/train-clean-100" \
                --output-path "${REPO_DIR}/data/hubert_centroids_100.pkl" \
                --max-samples 10000
        fi
    else
        echo -e "  ${YELLOW}⚠${NC} LibriTTS not found. Download datasets first, then run:"
        echo "      python data/extract_hubert_centroids.py \\"
        echo "          --audio-dir data/libritts/LibriTTS/train-clean-100 \\"
        echo "          --output-path data/hubert_centroids_100.pkl \\"
        echo "          --max-samples 10000"
    fi
fi

# ---- Done -------------------------------------------------------------------
echo ""
echo -e "${CYAN}================================================================${NC}"
echo -e "${CYAN}  Setup Complete!${NC}"
echo -e "${CYAN}================================================================${NC}"
echo ""
echo "  To activate the environment:"
echo "    conda activate ${ENV_NAME}"
echo ""
echo "  Quick commands:"
echo "    # Test pipeline (no StreamVC checkpoint needed):"
echo "    python run_pipeline.py --input audio.wav --output output.wav --device cpu"
echo ""
echo "    # Train StreamVC (single GPU):"
echo "    python streamvc/train.py --config configs/streamvc_config.yaml"
echo ""
echo "    # Train StreamVC via SLURM (2 GPUs):"
echo "    sbatch slurm_train.sbatch"
echo ""
echo "    # Evaluate pipeline:"
echo "    python evaluate_pipeline.py --auto-testset --num-samples 20 --device cuda"
echo ""
echo "    # Run evaluation via SLURM:"
echo "    sbatch slurm_eval.sbatch"
echo ""
echo -e "${CYAN}================================================================${NC}"
