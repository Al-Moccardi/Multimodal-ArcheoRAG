#!/bin/bash
# ============================================
# Pompeii Multimodal Framework — Setup Script
# ============================================
set -e

echo "=========================================="
echo "  Pompeii Multimodal Framework — Setup"
echo "=========================================="

# ── 1. Check Python ──
echo ""
echo "[1/5] Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "  ✗ Python3 not found. Please install Python 3.10+"
    exit 1
fi
PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  ✓ Python $PYVER"

# ── 2. Check Ollama ──
echo ""
echo "[2/5] Checking Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "  ✗ Ollama not found."
    echo "    Install from: https://ollama.ai"
    echo "    Linux:   curl -fsSL https://ollama.ai/install.sh | sh"
    echo "    macOS:   brew install ollama"
    exit 1
fi
echo "  ✓ Ollama found"

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "  ⚠ Ollama is not running. Starting..."
    ollama serve &
    sleep 3
fi
echo "  ✓ Ollama server is running"

# ── 3. Pull models ──
echo ""
echo "[3/5] Pulling Ollama models (this may take a while)..."

echo "  → Pulling llava:7b (VLM, ~4.7GB)..."
ollama pull llava:7b

echo "  → Pulling mistral:7b (LLM, ~4.1GB)..."
ollama pull mistral:7b

echo "  ✓ Models ready"

# ── 4. Install Python dependencies ──
echo ""
echo "[4/5] Installing Python dependencies..."
pip install -r requirements.txt

echo "  ✓ Dependencies installed"

# ── 5. Verify structure ──
echo ""
echo "[5/5] Verifying project structure..."

DIRS=(
    "knowledge_base/ceramics/historical"
    "knowledge_base/ceramics/cataloguing"
    "knowledge_base/paintings/historical"
    "knowledge_base/paintings/cataloguing"
    "knowledge_base/architecture/historical"
    "knowledge_base/architecture/cataloguing"
    "vector_stores"
    "outputs/metadata"
    "outputs/annotations"
    "outputs/images"
)

for d in "${DIRS[@]}"; do
    mkdir -p "$d"
    echo "  ✓ $d"
done

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "  1. Place your PDFs in the knowledge_base/ folders:"
echo "     knowledge_base/ceramics/historical/    ← Peacock, Opait, Olcese..."
echo "     knowledge_base/ceramics/cataloguing/   ← Dressel typology, Gilboa..."
echo "     knowledge_base/paintings/historical/   ← Cuní, Springer, Bergmann..."
echo "     knowledge_base/paintings/cataloguing/  ← Mau styles, RePAIR..."
echo "     knowledge_base/architecture/historical/ ← Strickland, opus craticium..."
echo "     knowledge_base/architecture/cataloguing/← Opus techniques, MDPI..."
echo ""
echo "  2. Index the PDFs:"
echo "     python indexer.py"
echo ""
echo "  3. Launch the app:"
echo "     python app.py"
echo ""
echo "  4. Open browser at http://localhost:7860"
echo ""
