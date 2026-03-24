"""
Configuration for the Pompeii Multimodal Archaeological Framework.
"""
from pathlib import Path

# ── Output directories ──────────────────────────
OUTPUT_DIR          = Path("./output")
OUTPUT_METADATA     = OUTPUT_DIR / "metadata"
OUTPUT_ANNOTATIONS  = OUTPUT_DIR / "annotations"
OUTPUT_IMAGES       = OUTPUT_DIR / "images"

for _d in [OUTPUT_METADATA, OUTPUT_ANNOTATIONS, OUTPUT_IMAGES]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Knowledge base directories ──────────────────
KB_BASE_DIR = Path("./knowledge_bases")

KB_PATHS = {
    "ceramics": {
        "historical":  KB_BASE_DIR / "ceramics"  / "historical",
        "cataloguing": KB_BASE_DIR / "ceramics"  / "cataloguing",
    },
    "paintings": {
        "historical":  KB_BASE_DIR / "paintings" / "historical",
        "cataloguing": KB_BASE_DIR / "paintings" / "cataloguing",
    },
    "architecture": {
        "historical":  KB_BASE_DIR / "architecture" / "historical",
        "cataloguing": KB_BASE_DIR / "architecture" / "cataloguing",
    },
}

for _domain, _paths in KB_PATHS.items():
    for _type, _path in _paths.items():
        _path.mkdir(parents=True, exist_ok=True)

# ── Models (Ollama) ─────────────────────────────
VLM_MODEL        = "llava:7b"
LLM_MODEL        = "mistral:7b"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"

# ── RAG settings ────────────────────────────────
RAG_TOP_K         = 5
RAG_CHUNK_SIZE    = 512
RAG_CHUNK_OVERLAP = 64

# Aliases (used by indexer.py)
CHUNK_SIZE    = RAG_CHUNK_SIZE
CHUNK_OVERLAP = RAG_CHUNK_OVERLAP

# ── Domains ─────────────────────────────────────
DOMAINS = ["ceramics", "paintings", "architecture"]
