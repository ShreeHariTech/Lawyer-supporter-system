"""
rag_pipeline.py — Core RAG Engine
Handles: PDF ingestion → chunking → embedding → FAISS indexing → retrieval
"""

import os
import logging
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# PDF processing
import pdfplumber

# LangChain components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────
EMBED_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"   # free, fast, good quality
CHUNK_SIZE       = 800       # characters per chunk
CHUNK_OVERLAP    = 150       # overlap to preserve context across chunk boundaries
TOP_K            = 6         # number of chunks to retrieve per query
VECTOR_DB_DIR    = Path(__file__).parent / "vector_db"
FAISS_INDEX_FILE = VECTOR_DB_DIR / "faiss_index"
META_FILE        = VECTOR_DB_DIR / "meta.pkl"


# ─────────────────────────────────────────────────────────────────
#  PDF TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text page-by-page from a PDF using pdfplumber.
    Returns list of dicts: { page_num, text, source }
    """
    log.info(f"Extracting text from: {pdf_path}")
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        log.info(f"  PDF has {total} pages")

        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                pages.append({
                    "page_num": i,
                    "text": text.strip(),
                    "source": Path(pdf_path).name,
                    "total_pages": total,
                })

    log.info(f"  Extracted text from {len(pages)}/{total} pages")
    return pages


# ─────────────────────────────────────────────────────────────────
#  DOCUMENT CHUNKING
# ─────────────────────────────────────────────────────────────────
def chunk_documents(pages: List[Dict[str, Any]]) -> List[Document]:
    """
    Split page text into overlapping chunks.
    Preserves page number and source as metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    docs: List[Document] = []
    for page in pages:
        chunks = splitter.split_text(page["text"])
        for j, chunk in enumerate(chunks):
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "source":      page["source"],
                    "page":        page["page_num"],
                    "chunk_index": j,
                    "total_pages": page["total_pages"],
                }
            ))

    log.info(f"  Created {len(docs)} chunks from {len(pages)} pages")
    return docs


# ─────────────────────────────────────────────────────────────────
#  RAG PIPELINE CLASS
# ─────────────────────────────────────────────────────────────────
class RAGPipeline:
    def __init__(self):
        self.vectorstore: Optional[FAISS] = None
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.is_ready: bool = False
        self.pdf_name: str = ""
        self.pdf_hash: str = ""
        self.num_chunks: int = 0
        self.num_pages: int = 0

        VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

    # ── LAZY LOAD EMBEDDING MODEL ──────────────────────────────
    def _get_embeddings(self) -> HuggingFaceEmbeddings:
        if self.embeddings is None:
            log.info(f"Loading embedding model: {EMBED_MODEL}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBED_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            log.info("  Embedding model loaded.")
        return self.embeddings

    # ── PDF HASH (for cache validation) ───────────────────────
    @staticmethod
    def _hash_pdf(pdf_path: str) -> str:
        h = hashlib.md5()
        with open(pdf_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    # ── BUILD INDEX FROM PDF ───────────────────────────────────
    def build_index(self, pdf_path: str) -> Dict[str, Any]:
        """
        Full pipeline: PDF → text → chunks → embeddings → FAISS index.
        Saves index to disk for reuse.
        """
        pdf_path = str(pdf_path)
        log.info(f"=== Building RAG index from: {pdf_path} ===")

        # 1. Extract text
        pages = extract_text_from_pdf(pdf_path)
        if not pages:
            raise ValueError("No text could be extracted from the PDF. It may be scanned/image-based.")

        # 2. Chunk
        docs = chunk_documents(pages)
        if not docs:
            raise ValueError("Document chunking produced no results.")

        # 3. Embed + index
        log.info(f"  Embedding {len(docs)} chunks (this may take a moment)…")
        emb = self._get_embeddings()
        self.vectorstore = FAISS.from_documents(docs, emb)

        # 4. Persist
        self.vectorstore.save_local(str(FAISS_INDEX_FILE))
        meta = {
            "pdf_name":   Path(pdf_path).name,
            "pdf_hash":   self._hash_pdf(pdf_path),
            "num_chunks": len(docs),
            "num_pages":  len(pages),
        }
        with open(META_FILE, "wb") as f:
            pickle.dump(meta, f)

        # 5. Update state
        self.pdf_name   = meta["pdf_name"]
        self.pdf_hash   = meta["pdf_hash"]
        self.num_chunks = meta["num_chunks"]
        self.num_pages  = meta["num_pages"]
        self.is_ready   = True

        log.info(f"=== Index built: {len(docs)} chunks, {len(pages)} pages ===")
        return meta

    # ── LOAD SAVED INDEX ───────────────────────────────────────
    def load_saved_index(self) -> bool:
        """Load FAISS index + metadata from disk if available."""
        if not FAISS_INDEX_FILE.exists() or not META_FILE.exists():
            log.info("No saved index found.")
            return False
        try:
            log.info("Loading saved FAISS index…")
            emb = self._get_embeddings()
            self.vectorstore = FAISS.load_local(
                str(FAISS_INDEX_FILE), emb, allow_dangerous_deserialization=True
            )
            with open(META_FILE, "rb") as f:
                meta = pickle.load(f)
            self.pdf_name   = meta.get("pdf_name", "unknown.pdf")
            self.pdf_hash   = meta.get("pdf_hash", "")
            self.num_chunks = meta.get("num_chunks", 0)
            self.num_pages  = meta.get("num_pages", 0)
            self.is_ready   = True
            log.info(f"  Loaded index: {self.num_chunks} chunks from '{self.pdf_name}'")
            return True
        except Exception as e:
            log.warning(f"Failed to load saved index: {e}")
            return False

    # ── RETRIEVE RELEVANT CHUNKS ───────────────────────────────
    def retrieve(self, query: str, k: int = TOP_K) -> List[Dict[str, Any]]:
        """
        Semantic search: returns top-k most relevant chunks for the query.
        Returns list of { content, page, source, score }
        """
        if not self.is_ready or self.vectorstore is None:
            raise RuntimeError("RAG pipeline is not ready. Please upload a PDF first.")

        results: List[Tuple[Document, float]] = self.vectorstore.similarity_search_with_score(
            query, k=k
        )

        retrieved = []
        for doc, score in results:
            retrieved.append({
                "content":  doc.page_content,
                "page":     doc.metadata.get("page", "?"),
                "source":   doc.metadata.get("source", "PDF"),
                "score":    round(float(score), 4),
            })

        return retrieved

    # ── BUILD CONTEXT STRING FOR LLM ──────────────────────────
    @staticmethod
    def format_context(chunks: List[Dict[str, Any]]) -> str:
        """Formats retrieved chunks into a single readable context block."""
        parts = []
        for i, ch in enumerate(chunks, start=1):
            parts.append(
                f"[Excerpt {i} — {ch['source']}, Page {ch['page']}]\n{ch['content']}"
            )
        return "\n\n" + ("\n\n" + "─" * 60 + "\n\n").join(parts) + "\n"

    # ── STATUS ─────────────────────────────────────────────────
    def status(self) -> Dict[str, Any]:
        return {
            "is_ready":   self.is_ready,
            "pdf_name":   self.pdf_name,
            "num_pages":  self.num_pages,
            "num_chunks": self.num_chunks,
        }
