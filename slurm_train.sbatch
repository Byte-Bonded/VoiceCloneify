#!/bin/bash
# ==============================================================================
# SLURM Job Script — StreamVC Training (2-GPU DDP)
# ==============================================================================
# Train from scratch:
#   sbatch slurm_train.sbatch
#
# Resume from a checkpoint (only after you have one):
#   sbatch slurm_train.sbatch --export=RESUME_CKPT=checkpoints/streamvc/checkpoint_epoch_5.pt
#
# Monitor with: squeue -u $USER
# Cancel with:  scancel <job_id>
# View logs:    tail -f logs/slurm/train_<job_id>.out
# ==============================================================================

#SBATCH --job-name=streamvc-train
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm/train_%j.out
#SBATCH --error=logs/slurm/train_%j.err

# ==============================================================================
# Environment Setup
# ==============================================================================

set -euo pipefail

echo "================================================================"
echo "  StreamVC Training — SLURM Job ${SLURM_JOB_ID}"
echo "================================================================"
echo "  Date       : $(date)"
echo "  Node       : ${SLURM_NODELIST}"
echo "  GPUs       : ${SLURM_GPUS_ON_NODE:-2}"
echo "  CPUs       : ${SLURM_CPUS_PER_TASK}"
echo "  Partition  : ${SLURM_JOB_PARTITION}"
echo "================================================================"

# ---- Paths ------------------------------------------------------------------
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${REPO_DIR}/configs/streamvc_config.yaml"
ENV_NAME="voicecloneify"

# Resume checkpoint (set via --export=RESUME_CKPT=... or leave empty)
RESUME_CKPT="${RESUME_CKPT:-}"

# ---- Load modules -----------------------------------------------------------
echo ""
echo "[1/4] Loading modules..."
module purge 2>/dev/null || true
module load cuda/12.3 2>/dev/null || module load cuda/12.5 2>/dev/null || module load cuda/12.2 2>/dev/null || true
module load cudnn/12/9.8 2>/dev/null || module load cudnn/12/9.3 2>/dev/null || true
echo "  CUDA: $(nvcc --version 2>/dev/null | grep release | awk '{print $6}' || echo 'system')"

# ---- Activate conda environment --------------------------------------------
echo ""
echo "[2/4] Activating environment..."
CONDA_BASE="/dist_home/common-apps/mamba"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
if [[ -f "${CONDA_BASE}/etc/profile.d/mamba.sh" ]]; then
    source "${CONDA_BASE}/etc/profile.d/mamba.sh"
fi
conda activate "${ENV_NAME}" 2>/dev/null || mamba activate "${ENV_NAME}" 2>/dev/null || {
    echo "ERROR: Could not activate '${ENV_NAME}'. Run install_setup.sh first."
    exit 1
}
echo "  Python : $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"

# ---- Environment variables --------------------------------------------------
echo ""
echo "[3/4] Setting environment..."

# NCCL configuration for multi-GPU
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_P2P_DISABLE=0

# Performance
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TORCH_NCCL_BLOCKING_WAIT=1

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "  Detected ${NUM_GPUS} GPU(s):"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Create output directories
mkdir -p "${REPO_DIR}/logs/slurm"
mkdir -p "${REPO_DIR}/checkpoints/streamvc"
mkdir -p "${REPO_DIR}/logs/streamvc"

# ---- Launch training --------------------------------------------------------
echo ""
echo "[4/4] Launching training..."
echo "  Config : ${CONFIG}"
echo "  Resume : ${RESUME_CKPT:-none}"
echo "  GPUs   : ${NUM_GPUS}"
echo "  Mode   : DDP (torchrun)"
echo ""

cd "${REPO_DIR}"

# Build resume flag
RESUME_FLAG=""
if [[ -n "${RESUME_CKPT}" ]]; then
    RESUME_FLAG="--resume ${REPO_DIR}/${RESUME_CKPT}"
    echo "  Resuming from: ${RESUME_CKPT}"
fi

# Launch with torchrun for DDP
# Note: train.py uses bare imports (from model import ...) so we launch
# with --module to run it as streamvc.train from the repo root.
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NUM_GPUS}" \
    streamvc/train.py \
        --config "${CONFIG}" \
        --ddp \
        ${RESUME_FLAG}

# ---- Done -------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  Training complete — Job ${SLURM_JOB_ID}"
echo "  $(date)"
echo "================================================================"
echo "  Checkpoints: ${REPO_DIR}/checkpoints/streamvc/"
echo "  TensorBoard: tensorboard --logdir ${REPO_DIR}/logs/streamvc"
echo "================================================================"
