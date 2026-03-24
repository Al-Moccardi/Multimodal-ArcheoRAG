"""
Pompeii Multimodal Framework — PDF Indexer

Reads PDFs from knowledge_bases/{domain}/{historical|cataloguing}/
and indexes them into ChromaDB vector stores.

Directory structure expected:
    knowledge_bases/
    ├── ceramics/
    │   ├── historical/    ← art history, trade, dating PDFs
    │   └── cataloguing/   ← Dressel typology, classification manuals
    ├── paintings/
    │   ├── historical/    ← Mau, Pompeian painting history
    │   └── cataloguing/   ← style classification schemas
    └── architecture/
        ├── historical/    ← Roman building history
        └── cataloguing/   ← opus classification manuals

Usage:
    python indexer.py
    python indexer.py --domain ceramics
    python indexer.py --domain ceramics --type historical
"""

import os
import sys
import argparse
import logging
from pathlib import Path

from config import KB_PATHS, CHUNK_SIZE, CHUNK_OVERLAP, DOMAINS

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ─── PDF text extraction ─────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file. Tries multiple backends."""
    text = ""

    # Try PyPDF2 first
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
        pages = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                pages.append(t)
        text = "\n\n".join(pages)
        if text.strip():
            return text
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"  PyPDF2 failed on {pdf_path}: {e}")

    # Try pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            pages = []
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    pages.append(t)
            text = "\n\n".join(pages)
            if text.strip():
                return text
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"  pdfplumber failed on {pdf_path}: {e}")

    # Try pymupdf (fitz)
    try:
        import fitz
        doc = fitz.open(pdf_path)
        pages = []
        for page in doc:
            t = page.get_text()
            if t:
                pages.append(t)
        text = "\n\n".join(pages)
        doc.close()
        if text.strip():
            return text
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"  pymupdf failed on {pdf_path}: {e}")

    if not text.strip():
        logger.warning(f"  Could not extract text from: {pdf_path}")
        logger.warning(f"  Install one of: pip install PyPDF2 pdfplumber pymupdf")

    return text


def extract_text_from_file(file_path: str) -> str:
    """Read a plain text or markdown file."""
    try:
        return Path(file_path).read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.warning(f"  Failed to read {file_path}: {e}")
        return ""


# ─── Chunking ────────────────────────────────────────────────────

def chunk_text(text: str, source: str,
               chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        if len(chunk_words) < 20:
            continue
        chunks.append({
            "text": " ".join(chunk_words),
            "source": source,
            "chunk_index": len(chunks),
        })

    return chunks


# ─── Indexing ────────────────────────────────────────────────────

def index_folder(rag_engine, domain: str, db_type: str, folder: Path):
    """
    Index all PDFs and text files in a folder into the vector store.
    """
    if not folder.exists():
        logger.info(f"    Folder not found: {folder} — skipping")
        return 0

    supported = (".pdf", ".txt", ".md")
    files = sorted(f for f in folder.iterdir() if f.suffix.lower() in supported)

    if not files:
        logger.info(f"    No documents in: {folder}")
        return 0

    all_chunks = []

    for filepath in files:
        logger.info(f"    Processing: {filepath.name}")

        if filepath.suffix.lower() == ".pdf":
            text = extract_text_from_pdf(str(filepath))
        else:
            text = extract_text_from_file(str(filepath))

        if not text.strip():
            logger.warning(f"    → Empty or unreadable, skipping")
            continue

        chunks = chunk_text(text, source=filepath.name)
        logger.info(f"    → {len(chunks)} chunks ({len(text):,} chars)")
        all_chunks.extend(chunks)

    if all_chunks:
        rag_engine.index_documents(domain, db_type, all_chunks)
        logger.info(f"    ✓ Indexed {len(all_chunks)} chunks into [{domain}_{db_type}]")
    else:
        logger.info(f"    No content to index for [{domain}_{db_type}]")

    return len(all_chunks)


# ─── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Index PDFs into vector stores")
    parser.add_argument("--domain", choices=DOMAINS, default=None,
                        help="Index only this domain (default: all)")
    parser.add_argument("--type", choices=["historical", "cataloguing"], default=None,
                        help="Index only this DB type (default: both)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Pompeii Multimodal Framework — PDF Indexer")
    print("=" * 60)

    # Initialize RAG engine
    print("\n[1] Initializing RAG engine and embedding model...")
    from pipeline.rag_engine import DualRAGEngine
    rag = DualRAGEngine()

    # Show current stats
    print("\n[2] Current vector store stats:")
    stats = rag.get_store_stats()
    for store, count in sorted(stats.items()):
        print(f"    {store}: {count} chunks")

    # Index — iterate over KB_PATHS (dict), not DOMAINS (list)
    print("\n[3] Indexing documents...\n")
    total_chunks = 0

    for domain, paths in KB_PATHS.items():
        if args.domain and domain != args.domain:
            continue

        print(f"  [{domain.upper()}]")

        for db_type, folder in paths.items():
            if args.type and db_type != args.type:
                continue

            print(f"  ── {db_type} ({folder}) ──")
            n = index_folder(rag, domain, db_type, Path(folder))
            total_chunks += n

        print()

    # Final stats
    print("[4] Updated vector store stats:")
    stats = rag.get_store_stats()
    for store, count in sorted(stats.items()):
        icon = "✓" if count > 0 else "○"
        print(f"    [{icon}] {store}: {count} chunks")

    print(f"\n{'=' * 60}")
    print(f"  Done. Indexed {total_chunks} total chunks.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
