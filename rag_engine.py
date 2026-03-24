"""
Dual RAG Engine.
Each domain has TWO vector databases:
  - Historical DB: art history texts, dating, provenance, context
  - Cataloguing DB: classification manuals, field schemas, normative rules

The engine queries both and returns separate context chunks
that will be merged in the refinement step.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class DualRAGEngine:
    """
    Dual-retrieval RAG engine for a single archaeological domain.
    
    Architecture:
        Query → [Historical Vector DB] → historical chunks
        Query → [Cataloguing Vector DB] → cataloguing chunks
        Both are returned separately for the refinement LLM to merge.
    """

    def __init__(
        self,
        domain: str,
        historical_db_path: str,
        cataloguing_db_path: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k: int = 5
    ):
        self.domain = domain
        self.historical_db_path = historical_db_path
        self.cataloguing_db_path = cataloguing_db_path
        self.embedding_model = embedding_model
        self.top_k = top_k

        self.historical_db = None
        self.cataloguing_db = None
        self.embedder = None

    # ──────────────────────────────────────────
    #  Initialization
    # ──────────────────────────────────────────

    def initialize(self):
        """Load or create vector databases."""
        self._load_embedder()
        self.historical_db = self._load_or_create_db(self.historical_db_path, "historical")
        self.cataloguing_db = self._load_or_create_db(self.cataloguing_db_path, "cataloguing")

    def _load_embedder(self):
        """Load the sentence embedding model."""
        if self.embedder is not None:
            return

        logger.info(f"[{self.domain}] Loading embedding model: {self.embedding_model}")
        
        # ── Option A: sentence-transformers ─────
        # from sentence_transformers import SentenceTransformer
        # self.embedder = SentenceTransformer(self.embedding_model)

        # ── Option B: HuggingFace ───────────────
        # from transformers import AutoTokenizer, AutoModel
        # self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model)
        # self.embedder = AutoModel.from_pretrained(self.embedding_model)

        logger.info(f"[{self.domain}] Embedder loaded (placeholder).")

    def _load_or_create_db(self, db_path: str, db_type: str):
        """Load an existing vector DB or create from documents."""
        db_dir = Path(db_path)

        if db_dir.exists() and any(db_dir.iterdir()):
            logger.info(f"[{self.domain}] Loading {db_type} DB from: {db_path}")
            return self._load_existing_db(db_path)
        else:
            logger.info(f"[{self.domain}] {db_type} DB not found at {db_path}.")
            logger.info(f"[{self.domain}] Call ingest_documents() to create it.")
            return None

    def _load_existing_db(self, db_path: str):
        """Load a pre-built vector database."""
        # ── Option A: ChromaDB ──────────────────
        # import chromadb
        # client = chromadb.PersistentClient(path=db_path)
        # collection = client.get_collection(name="documents")
        # return collection

        # ── Option B: FAISS ─────────────────────
        # import faiss
        # index = faiss.read_index(str(Path(db_path) / "index.faiss"))
        # return index

        return None  # Placeholder

    # ──────────────────────────────────────────
    #  Document ingestion
    # ──────────────────────────────────────────

    def ingest_documents(self, documents_dir: str, db_type: str):
        """
        Ingest documents into a vector database.
        
        Args:
            documents_dir: Directory containing PDFs/text files.
            db_type: "historical" or "cataloguing".
        """
        docs_path = Path(documents_dir)
        if not docs_path.exists():
            raise FileNotFoundError(f"Documents directory not found: {documents_dir}")

        logger.info(f"[{self.domain}] Ingesting {db_type} documents from: {documents_dir}")

        # ── Step 1: Load documents ──────────────
        documents = self._load_documents(docs_path)
        logger.info(f"[{self.domain}] Loaded {len(documents)} documents.")

        # ── Step 2: Chunk documents ─────────────
        chunks = self._chunk_documents(documents)
        logger.info(f"[{self.domain}] Created {len(chunks)} chunks.")

        # ── Step 3: Embed and store ─────────────
        db_path = (
            self.historical_db_path if db_type == "historical"
            else self.cataloguing_db_path
        )
        self._embed_and_store(chunks, db_path)
        logger.info(f"[{self.domain}] {db_type} DB created at: {db_path}")

    def _load_documents(self, docs_path: Path) -> list[dict]:
        """Load documents from directory (PDF, TXT, MD)."""
        documents = []

        for file_path in sorted(docs_path.rglob("*")):
            if file_path.suffix.lower() == ".txt":
                text = file_path.read_text(encoding="utf-8", errors="ignore")
                documents.append({"source": str(file_path), "text": text})

            elif file_path.suffix.lower() == ".pdf":
                # from PyPDF2 import PdfReader
                # reader = PdfReader(str(file_path))
                # text = "\n".join(page.extract_text() or "" for page in reader.pages)
                # documents.append({"source": str(file_path), "text": text})
                documents.append({"source": str(file_path), "text": "[PDF placeholder]"})

            elif file_path.suffix.lower() == ".md":
                text = file_path.read_text(encoding="utf-8", errors="ignore")
                documents.append({"source": str(file_path), "text": text})

        return documents

    def _chunk_documents(
        self, documents: list[dict],
        chunk_size: int = 512, overlap: int = 64
    ) -> list[dict]:
        """Split documents into overlapping chunks."""
        chunks = []

        for doc in documents:
            text = doc["text"]
            words = text.split()

            for i in range(0, len(words), chunk_size - overlap):
                chunk_words = words[i:i + chunk_size]
                if len(chunk_words) < 20:
                    continue

                chunks.append({
                    "text": " ".join(chunk_words),
                    "source": doc["source"],
                    "chunk_index": len(chunks)
                })

        return chunks

    def _embed_and_store(self, chunks: list[dict], db_path: str):
        """Embed chunks and store in vector database."""
        Path(db_path).mkdir(parents=True, exist_ok=True)

        # ── ChromaDB implementation ─────────────
        # import chromadb
        # client = chromadb.PersistentClient(path=db_path)
        # collection = client.get_or_create_collection(
        #     name="documents",
        #     metadata={"hnsw:space": "cosine"}
        # )
        # texts = [c["text"] for c in chunks]
        # metadatas = [{"source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks]
        # ids = [f"chunk_{i}" for i in range(len(chunks))]
        # collection.add(documents=texts, metadatas=metadatas, ids=ids)

        # ── FAISS implementation ────────────────
        # embeddings = self.embedder.encode(texts, show_progress_bar=True)
        # import faiss, numpy as np, json
        # dim = embeddings.shape[1]
        # index = faiss.IndexFlatIP(dim)
        # faiss.normalize_L2(embeddings)
        # index.add(embeddings)
        # faiss.write_index(index, str(Path(db_path) / "index.faiss"))
        # with open(Path(db_path) / "chunks.json", "w") as f:
        #     json.dump(chunks, f)

        logger.info(f"Stored {len(chunks)} chunks (placeholder).")

    # ──────────────────────────────────────────
    #  Query methods
    # ──────────────────────────────────────────

    def query_historical(self, query: str) -> str:
        """
        Query the historical vector DB.
        Returns concatenated relevant chunks about art history, dating, context.
        """
        return self._query_db(self.historical_db, query, "historical")

    def query_cataloguing(self, query: str) -> str:
        """
        Query the cataloguing vector DB.
        Returns concatenated relevant chunks about classification schemas, field rules.
        """
        return self._query_db(self.cataloguing_db, query, "cataloguing")

    def _query_db(self, db, query: str, db_type: str) -> str:
        """Execute a similarity search on a vector database."""
        if db is None:
            logger.warning(
                f"[{self.domain}] {db_type} DB not initialized. "
                f"Returning empty context."
            )
            return f"[No {db_type} context available — DB not loaded]"

        logger.debug(f"[{self.domain}] Querying {db_type} DB: {query[:80]}...")

        # ── ChromaDB query ──────────────────────
        # results = db.query(query_texts=[query], n_results=self.top_k)
        # chunks = results["documents"][0]
        # sources = [m["source"] for m in results["metadatas"][0]]
        # context = "\n\n---\n\n".join(
        #     f"[Source: {src}]\n{chunk}"
        #     for chunk, src in zip(chunks, sources)
        # )
        # return context

        # ── FAISS query ─────────────────────────
        # query_vec = self.embedder.encode([query])
        # faiss.normalize_L2(query_vec)
        # scores, indices = db.search(query_vec, self.top_k)
        # chunks = [self.chunk_store[i] for i in indices[0]]
        # return "\n\n---\n\n".join(c["text"] for c in chunks)

        return f"[Placeholder {db_type} context for domain={self.domain}]"
