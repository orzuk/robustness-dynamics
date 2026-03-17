#!/bin/bash
# ============================================================================
# One-time environment setup for the ATLAS robustness-dynamics pipeline.
#
# Creates a Python virtual environment with all dependencies for:
#   - Meira's thesis code (ESM-1v, BioPython, etc.)
#   - Direction 7 analysis scripts (scipy, matplotlib, seaborn, etc.)
#   - ThermoMPNN (pytorch-lightning, omegaconf, etc.)
#
# Usage:
#   # Run on an interactive GPU node (to detect CUDA properly):
#   srun --gpus=1 --mem=8G --time=01:00:00 --pty bash
#   bash /path/to/robustness-dynamics/scripts/slurm/0_setup_env.sh
#
# ============================================================================

set -euo pipefail

# Load shared paths (0_setup_env.sh is always run interactively, never via sbatch)
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

echo "============================================"
echo "Environment Setup"
echo "Date:       $(date)"
echo "Node:       $(hostname)"
echo "Venv:       ${VENV_DIR}"
echo "ThermoMPNN: ${THERMOMPNN_DIR}"
echo "============================================"

# --- Step 1: Detect Python ---
if command -v module &>/dev/null; then
    echo "Checking available Python modules..."
    module avail python 2>&1 | head -10 || true
    module avail anaconda 2>&1 | head -10 || true
    module avail miniconda 2>&1 | head -10 || true
    echo ""

    module load python/3.10 2>/dev/null \
        || module load python/3.9 2>/dev/null \
        || module load python 2>/dev/null \
        || echo "No python module found, using system python"
fi

PYTHON=$(command -v python3 || command -v python)
echo "Using Python: ${PYTHON}"
${PYTHON} --version

# --- Step 2: Create venv ---
if [[ -d "${VENV_DIR}" ]]; then
    echo "Venv already exists at ${VENV_DIR}"
    echo "To recreate, delete it first: rm -rf ${VENV_DIR}"
else
    echo "Creating venv at ${VENV_DIR}..."
    mkdir -p "$(dirname "${VENV_DIR}")"
    ${PYTHON} -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
pip install --upgrade pip

# --- Step 3: Detect CUDA version ---
echo ""
echo "Detecting CUDA..."
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
    CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[\d.]+' || echo "")
    echo "CUDA driver version: ${CUDA_VERSION}"
else
    CUDA_VERSION=""
    echo "WARNING: nvidia-smi not found. Run this on a GPU node!"
    echo "  srun --gpus=1 --mem=8G --time=01:00:00 --pty bash"
fi

# --- Step 4: Install PyTorch ---
echo ""
echo "Installing PyTorch..."

CUDA_MAJOR=$(echo "${CUDA_VERSION}" | cut -d. -f1)
CUDA_MINOR=$(echo "${CUDA_VERSION}" | cut -d. -f2)

if [[ "${CUDA_MAJOR}" == "12" ]]; then
    TORCH_URL="https://download.pytorch.org/whl/cu121"
elif [[ "${CUDA_MAJOR}" == "11" && "${CUDA_MINOR}" -ge "8" ]]; then
    TORCH_URL="https://download.pytorch.org/whl/cu118"
elif [[ "${CUDA_MAJOR}" == "11" ]]; then
    TORCH_URL="https://download.pytorch.org/whl/cu117"
else
    echo "WARNING: Unrecognized CUDA version '${CUDA_VERSION}', defaulting to cu118"
    TORCH_URL="https://download.pytorch.org/whl/cu118"
fi

echo "Using PyTorch index: ${TORCH_URL}"
pip install torch torchvision torchaudio --index-url "${TORCH_URL}"

# --- Step 5: Install all other dependencies ---
echo ""
echo "Installing dependencies..."

pip install \
    pytorch-lightning \
    omegaconf \
    biopython \
    fair-esm \
    numpy \
    pandas \
    scipy \
    matplotlib \
    seaborn \
    scikit-learn \
    tqdm \
    wandb \
    joblib \
    torchmetrics

# --- Step 6: Verify ---
echo ""
echo "============================================"
echo "Verification"
echo "============================================"

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "import esm; print('ESM: OK')"
python -c "import pytorch_lightning; print(f'PyTorch Lightning {pytorch_lightning.__version__}: OK')"
python -c "import omegaconf; print('OmegaConf: OK')"
python -c "import Bio; print('BioPython: OK')"
python -c "import scipy; print(f'SciPy {scipy.__version__}: OK')"
python -c "import matplotlib; print('Matplotlib: OK')"
python -c "import seaborn; print('Seaborn: OK')"

# --- Step 7: Check ThermoMPNN weights ---
echo ""
if [[ -d "${THERMOMPNN_DIR}" ]]; then
    echo "ThermoMPNN directory found: ${THERMOMPNN_DIR}"
    if [[ -f "${THERMOMPNN_DIR}/models/thermoMPNN_default.pt" ]]; then
        echo "  Checkpoint: OK"
    else
        echo "  WARNING: models/thermoMPNN_default.pt not found"
    fi
    if [[ -f "${THERMOMPNN_DIR}/vanilla_model_weights/v_48_020.pt" ]]; then
        echo "  ProteinMPNN weights: OK"
    else
        echo "  WARNING: vanilla_model_weights/v_48_020.pt not found"
    fi
else
    echo "ThermoMPNN not cloned yet. To set up:"
    echo "  git clone https://github.com/Kuhlman-Lab/ThermoMPNN.git ${THERMOMPNN_DIR}"
fi

echo ""
echo "============================================"
echo "Setup complete!"
echo "Activate with: source ${VENV_DIR}/bin/activate"
echo "============================================"
