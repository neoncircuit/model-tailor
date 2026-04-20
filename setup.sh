#!/bin/bash
set -e

echo "=== model-tailor setup ==="

# -----------------------------------------------------------------------
# 1. Python version check
# -----------------------------------------------------------------------
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED="3.10"

if [ "$(printf '%s\n' "$REQUIRED" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED" ]; then
    echo "ERROR: Python >= $REQUIRED is required. Found: $PYTHON_VERSION"
    exit 1
fi
echo "Python $PYTHON_VERSION detected."

# -----------------------------------------------------------------------
# 2. Virtual environment
# -----------------------------------------------------------------------
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment (.venv)..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# -----------------------------------------------------------------------
# 3. Install dependencies
# -----------------------------------------------------------------------
echo "Installing dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

echo "Installing dev dependencies..."
pip install pytest ruff --quiet

# -----------------------------------------------------------------------
# 4. Environment file
# -----------------------------------------------------------------------
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Created .env from .env.example -- fill in your API keys."
fi

# -----------------------------------------------------------------------
# 5. Auto-activate .venv on cd (WSL / bash)
# -----------------------------------------------------------------------
HOOK_MARKER="# model-tailor auto-venv"

if ! grep -q "$HOOK_MARKER" ~/.bashrc 2>/dev/null; then
    echo ""
    echo "Adding auto-venv activation hook to ~/.bashrc..."
    cat >> ~/.bashrc << 'BASHRC_HOOK'

# model-tailor auto-venv
_auto_venv() {
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    fi
}
cd() {
    builtin cd "$@" || return
    _auto_venv
}
_auto_venv
BASHRC_HOOK
    echo "Auto-activation hook installed. New terminals will auto-activate .venv."
fi

# -----------------------------------------------------------------------
# 6. Create data directories
# -----------------------------------------------------------------------
mkdir -p data/raw data/curated data/formatted data/seeds models checkpoints

# -----------------------------------------------------------------------
# 7. GPU check
# -----------------------------------------------------------------------
echo ""
echo "=== GPU Check ==="
python3 -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / 1e9
    print(f'GPU: {name} ({mem:.1f} GB)')
    if mem < 10:
        print('WARNING: < 10 GB VRAM. QLoRA on 8B models needs ~10-12 GB.')
    else:
        print('GPU is sufficient for QLoRA fine-tuning.')
else:
    print('No GPU detected. Training will require a cloud GPU.')
    print('Recommended: RunPod or Lambda Labs with A100/H100')
"

# -----------------------------------------------------------------------
# 8. Verify installation
# -----------------------------------------------------------------------
echo ""
echo "=== Verification ==="
python3 -c "from src.llm.client import TeacherClient; print('src.llm ........... OK')"
python3 -c "from src.generate import strategies; print('src.generate ....... OK')"
python3 -c "from src.curate import dedup; print('src.curate ......... OK')"
python3 -c "from src.format import templates; print('src.format ......... OK')"
python3 -c "from src.train import lora; print('src.train .......... OK')"
python3 -c "from src.deploy import serve; print('src.deploy ......... OK')"
python3 -c "import mlflow; print(f'mlflow {mlflow.__version__} ...... OK')"

# Note: src.evaluate.metrics requires NLTK data download
echo ""
echo "Note: First run of src.evaluate.metrics will download NLTK data (takes ~30s)"

echo ""
echo "=== Setup complete ==="
echo "Activate with: source .venv/bin/activate"
echo "Run checks with: make check"
