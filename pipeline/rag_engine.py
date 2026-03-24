"""
Dual RAG Engine.
Each domain has TWO vector stores:
  - historical:  art history, dating, provenance, context
  - cataloguing: classification manuals, field schemas, normative rules

Uses ChromaDB for vector storage and sentence-transformers for embeddings.
Falls back gracefully if stores are empty.
"""

import json
import logging
from pathlib import Path

from config import KB_PATHS, DOMAINS, RAG_TOP_K

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False
    logger.warning("chromadb not installed. RAG will return empty context.")

try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False


CHROMA_DIR = Path("./vector_stores")


class DualRAGEngine:
    """
    Dual-retrieval RAG: queries both historical and cataloguing stores
    for a given domain, returns separate context chunks.
    """

    def __init__(self, top_k: int = None):
        self.top_k = top_k or RAG_TOP_K
        self.stores: dict[str, "chromadb.Collection"] = {}

        if HAS_CHROMA:
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=str(CHROMA_DIR),
                settings=Settings(anonymized_telemetry=False),
            )
            self._load_stores()
        else:
            self.client = None

    # ──────────────────────────────────────────
    #  Store management
    # ──────────────────────────────────────────

    def _load_stores(self):
        """Load or create ChromaDB collections for each domain × type."""
        for domain in DOMAINS:
            for db_type in ["historical", "cataloguing"]:
                name = f"{domain}_{db_type}"
                try:
                    coll = self.client.get_or_create_collection(
                        name=name,
                        metadata={"hnsw:space": "cosine"},
                    )
                    self.stores[name] = coll
                    count = coll.count()
                    if count > 0:
                        logger.info(f"  Loaded store [{name}]: {count} chunks")
                except Exception as e:
                    logger.error(f"  Failed to load store [{name}]: {e}")

    def get_store_stats(self) -> dict[str, int]:
        """Return chunk counts for all stores (used by Gradio UI)."""
        stats = {}
        for domain in DOMAINS:
            for db_type in ["historical", "cataloguing"]:
                name = f"{domain}_{db_type}"
                if name in self.stores:
                    stats[name] = self.stores[name].count()
                else:
                    stats[name] = 0
        return stats

    # ──────────────────────────────────────────
    #  Query (called by app.py)
    # ──────────────────────────────────────────

    def query(self, domain: str, query_text: str) -> dict:
        """
        Query both historical and cataloguing stores for a domain.

        Args:
            domain: "ceramics", "paintings", or "architecture"
            query_text: natural language query

        Returns:
            {
                "historical_context": [str, ...],
                "cataloguing_rules":  [str, ...],
            }
        """
        historical = self._query_store(f"{domain}_historical", query_text)
        cataloguing = self._query_store(f"{domain}_cataloguing", query_text)

        return {
            "historical_context": historical,
            "cataloguing_rules": cataloguing,
        }

    def _query_store(self, store_name: str, query_text: str) -> list[str]:
        """Query a single ChromaDB collection."""
        if store_name not in self.stores:
            return []

        coll = self.stores[store_name]
        if coll.count() == 0:
            return []

        try:
            results = coll.query(
                query_texts=[query_text],
                n_results=min(self.top_k, coll.count()),
            )
            docs = results.get("documents", [[]])[0]
            return docs
        except Exception as e:
            logger.error(f"Query failed on [{store_name}]: {e}")
            return []

    # ──────────────────────────────────────────
    #  Indexing (called by indexer.py)
    # ──────────────────────────────────────────

    def index_documents(self, domain: str, db_type: str, chunks: list[dict]):
        """
        Index pre-chunked documents into a store.

        Args:
            domain: "ceramics", "paintings", "architecture"
            db_type: "historical" or "cataloguing"
            chunks: list of {"text": str, "source": str, "chunk_index": int}
        """
        name = f"{domain}_{db_type}"
        if name not in self.stores:
            logger.error(f"Store [{name}] not found.")
            return

        coll = self.stores[name]
        texts = [c["text"] for c in chunks]
        metas = [{"source": c.get("source", ""), "chunk_index": c.get("chunk_index", i)} for i, c in enumerate(chunks)]
        ids = [f"{name}_chunk_{i}" for i in range(len(chunks))]

        coll.add(documents=texts, metadatas=metas, ids=ids)
        logger.info(f"Indexed {len(chunks)} chunks into [{name}]")
