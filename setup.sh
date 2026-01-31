#!/bin/bash
# Setup script for ragicamp experiments
# Usage: curl -sSL https://raw.githubusercontent.com/cassuci/ragicamp/main/setup.sh | bash
#    or: ./setup.sh [experiment_config]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() { echo -e "${GREEN}[setup]${NC} $1"; }
warn() { echo -e "${YELLOW}[warn]${NC} $1"; }
error() { echo -e "${RED}[error]${NC} $1"; exit 1; }

# Configuration
REPO_URL="https://github.com/cassuci/ragicamp.git"
REPO_DIR="${RAGICAMP_DIR:-$HOME/ragicamp}"
DEFAULT_CONFIG="conf/study/smart_retrieval_slm.yaml"
CONFIG="${1:-$DEFAULT_CONFIG}"

# ============================================================================
# Step 1: System dependencies
# ============================================================================
log "Checking system dependencies..."

if ! command -v git &> /dev/null; then
    error "git is not installed. Please install git first."
fi

if ! command -v python3 &> /dev/null; then
    error "python3 is not installed. Please install Python 3.10+ first."
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
log "Python version: $PYTHON_VERSION"

# ============================================================================
# Step 1b: CUDA toolkit for Flash Attention (optional but recommended)
# ============================================================================
if command -v nvidia-smi &> /dev/null; then
    log "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    
    if ! command -v nvcc &> /dev/null; then
        log "CUDA toolkit (nvcc) not found. Installing for Flash Attention support..."
        if command -v apt &> /dev/null; then
            sudo apt update -qq && sudo apt install -y nvidia-cuda-toolkit
            if command -v nvcc &> /dev/null; then
                log "CUDA toolkit installed: $(nvcc --version | grep release)"
            else
                warn "CUDA toolkit installation failed. Flash Attention will be disabled."
            fi
        else
            warn "apt not available. Install CUDA toolkit manually for Flash Attention support."
        fi
    else
        log "CUDA toolkit found: $(nvcc --version | grep release | head -1)"
    fi
    
    # Set CUDA_HOME if not set
    if [ -z "$CUDA_HOME" ]; then
        if [ -d "/usr/local/cuda" ]; then
            export CUDA_HOME="/usr/local/cuda"
        elif [ -d "/usr/lib/cuda" ]; then
            export CUDA_HOME="/usr/lib/cuda"
        fi
        [ -n "$CUDA_HOME" ] && log "CUDA_HOME set to: $CUDA_HOME"
    fi
else
    log "No GPU detected, skipping CUDA toolkit installation"
fi

# ============================================================================
# Step 2: Install uv (fast Python package manager)
# ============================================================================
if ! command -v uv &> /dev/null; then
    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"
    export PATH="$HOME/.cargo/bin:$PATH"
    
    if ! command -v uv &> /dev/null; then
        error "Failed to install uv. Please install manually: https://github.com/astral-sh/uv"
    fi
    log "uv installed successfully"
else
    log "uv already installed: $(uv --version)"
fi

# ============================================================================
# Step 3: Clone or update repository
# ============================================================================
if [ -d "$REPO_DIR" ]; then
    log "Repository exists at $REPO_DIR, pulling latest..."
    cd "$REPO_DIR"
    git pull
else
    log "Cloning repository to $REPO_DIR..."
    git clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi

# ============================================================================
# Step 4: Setup Python environment with uv
# ============================================================================
log "Setting up Python environment..."

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    log "Creating virtual environment..."
    uv venv --python python3
fi

# Determine which extras to install based on available hardware
EXTRAS="rag"

if command -v nvidia-smi &> /dev/null; then
    EXTRAS="$EXTRAS,vllm"
    log "GPU detected, including vLLM for optimized inference"
    
    # Only try flash-attn if nvcc is available
    if command -v nvcc &> /dev/null; then
        EXTRAS="$EXTRAS,flash"
        log "CUDA toolkit available, including Flash Attention"
    else
        warn "nvcc not found, skipping Flash Attention (embeddings will still work)"
    fi
fi

# Install dependencies with appropriate extras
log "Installing dependencies with extras: [$EXTRAS]..."
uv pip install -e ".[$EXTRAS]" --preview-features extra-build-dependencies

# ============================================================================
# Step 5: Show system info
# ============================================================================
log "System information:"
echo "  - Python: $(uv run python --version)"
echo "  - Available CPUs: $(python3 -c 'import os; print(len(os.sched_getaffinity(0)))' 2>/dev/null || nproc)"
echo "  - Total memory: $(free -h 2>/dev/null | awk '/^Mem:/{print $2}' || echo 'unknown')"

if command -v nvidia-smi &> /dev/null; then
    echo "  - GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
    echo "  - CUDA: $(nvcc --version 2>/dev/null | grep release | sed 's/.*release //' | sed 's/,.*//' || echo 'toolkit not installed')"
else
    echo "  - GPU: none detected"
fi

# Check installed optimizations
echo ""
log "Installed optimizations:"
uv run python -c "
try:
    import vllm; print('  - vLLM: ✓ installed')
except ImportError:
    print('  - vLLM: ✗ not installed')

try:
    import flash_attn; print('  - Flash Attention: ✓ installed')
except ImportError:
    print('  - Flash Attention: ✗ not installed (optional)')

import torch
if torch.cuda.is_available():
    print(f'  - PyTorch CUDA: ✓ {torch.version.cuda}')
else:
    print('  - PyTorch CUDA: ✗ CPU only')

if hasattr(torch, 'compile'):
    print('  - torch.compile: ✓ available')
"

# ============================================================================
# Step 6: Run experiment
# ============================================================================
if [ -f "$CONFIG" ]; then
    log "Starting experiment with config: $CONFIG"
    echo ""
    echo "========================================"
    echo "  Running: uv run ragicamp run $CONFIG"
    echo "========================================"
    echo ""
    
    uv run ragicamp run "$CONFIG"
else
    warn "Config file not found: $CONFIG"
    log "Available configs:"
    find conf -name "*.yaml" -type f 2>/dev/null | head -10 || echo "  (none found)"
    echo ""
    log "To run manually:"
    echo "  cd $REPO_DIR"
    echo "  uv run ragicamp run <config.yaml>"
fi
