#!/bin/bash
# ==============================================================================
# SLURM Job Script — Pipeline Evaluation (1 GPU)
# ==============================================================================
# Submit with:  sbatch slurm_eval.sbatch
#
# Override settings via --export:
#   sbatch slurm_eval.sbatch --export=NUM_SAMPLES=100,STREAMVC_CKPT=checkpoints/streamvc/best.pt
# ==============================================================================

#SBATCH --job-name=voiceclone-eval
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/eval_%j.out
#SBATCH --error=logs/slurm/eval_%j.err

# ==============================================================================
# Configuration (override via --export)
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="voicecloneify"

# Configurable parameters (can be overridden with --export=)
NUM_SAMPLES="${NUM_SAMPLES:-50}"
STREAMVC_CKPT="${STREAMVC_CKPT:-}"
TARGET_SPEAKER="${TARGET_SPEAKER:-}"
TEST_DIR="${TEST_DIR:-}"
LIBRITTS_DIR="${LIBRITTS_DIR:-data/libritts/LibriTTS/train-clean-100}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/eval}"

echo "================================================================"
echo "  VoiceCloneify Evaluation — SLURM Job ${SLURM_JOB_ID}"
echo "================================================================"
echo "  Date     : $(date)"
echo "  Node     : ${SLURM_NODELIST}"
echo "  GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "================================================================"

# ---- Load modules -----------------------------------------------------------
module purge 2>/dev/null || true
module load cuda/12.3 2>/dev/null || module load cuda/12.5 2>/dev/null || true
module load cudnn/12/9.8 2>/dev/null || true

# ---- Activate environment ---------------------------------------------------
CONDA_BASE="/dist_home/common-apps/mamba"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
[[ -f "${CONDA_BASE}/etc/profile.d/mamba.sh" ]] && source "${CONDA_BASE}/etc/profile.d/mamba.sh"
conda activate "${ENV_NAME}" 2>/dev/null || mamba activate "${ENV_NAME}" 2>/dev/null || {
    echo "ERROR: Could not activate '${ENV_NAME}'. Run install_setup.sh first."
    exit 1
}

echo "  Python  : $(python --version)"
echo "  PyTorch : $(python -c 'import torch; print(torch.__version__)')"
echo ""

# ---- Create output directories ----------------------------------------------
cd "${REPO_DIR}"
mkdir -p "${OUTPUT_DIR}/reports" logs/slurm

# ---- Build evaluation command -----------------------------------------------
EVAL_CMD="python evaluate_pipeline.py"

if [[ -n "${TEST_DIR}" ]]; then
    EVAL_CMD+=" --test-dir ${TEST_DIR}"
else
    EVAL_CMD+=" --auto-testset --num-samples ${NUM_SAMPLES} --libritts-dir ${LIBRITTS_DIR}"
fi

if [[ -n "${STREAMVC_CKPT}" && -f "${STREAMVC_CKPT}" ]]; then
    EVAL_CMD+=" --streamvc-checkpoint ${STREAMVC_CKPT}"
fi

if [[ -n "${TARGET_SPEAKER}" && -f "${TARGET_SPEAKER}" ]]; then
    EVAL_CMD+=" --target-speaker ${TARGET_SPEAKER}"
fi

EVAL_CMD+=" --output-dir ${OUTPUT_DIR} --device cuda"

echo "Running: ${EVAL_CMD}"
echo ""

# ---- Run evaluation ---------------------------------------------------------
eval "${EVAL_CMD}"

# ---- Done -------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  Evaluation complete — Job ${SLURM_JOB_ID}"
echo "  $(date)"
echo "================================================================"
echo "  Report : ${OUTPUT_DIR}/reports/metrics_report.json"
echo "  Audios : ${OUTPUT_DIR}/"
echo "================================================================"
