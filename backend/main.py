"""
main.py — FastAPI Backend for Nyay RAG Legal AI
Endpoints:
  POST /upload-pdf    → ingest PDF, build FAISS index
  POST /analyze       → RAG query + OpenRouter LLM
  GET  /status        → index status
  GET  /health        → server health
"""

import os
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, field_validator

from backend.rag_pipeline import RAGPipeline
import re

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ─────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
MODEL              = "openai/gpt-4o-mini"
API_URL            = "https://openrouter.ai/api/v1/chat/completions"
UPLOADS_DIR        = Path(__file__).parent / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

# Pre-load PDF path (optional — drop a PDF here to auto-index on startup)
DEFAULT_PDF_PATH = os.getenv("DEFAULT_PDF_PATH", "")

SYSTEM_PROMPT = """You are an advanced Indian Legal AI Assistant with deep knowledge of Indian law.

You have been provided with RETRIEVED LEGAL CONTEXT — actual excerpts from Indian law documents. Your analysis MUST be grounded in this context. Do not invent laws or sections.

Respond in this EXACT structured format:

🧾 CASE SUMMARY
Summarize the user's legal situation in 2–3 clear sentences.

⚖️ APPLICABLE LAWS
Based on the retrieved context, list the specific Articles, Sections, or provisions that apply. Format each as:
• Article/Section X — [Law Name]: Brief explanation of relevance.
If a law applies but wasn't in the retrieved context, say "Note: [Law] may also apply — consult a lawyer."

📊 ANALYSIS
Objective breakdown of the case:
- Legal strengths (what supports the user's position)
- Legal weaknesses (what may work against them)
- Key legal issues at stake

📈 POSSIBLE OUTCOMES
List 2–3 realistic outcomes as numbered points, each with a brief reason.

🎯 ESTIMATED WINNING CHANCE
State ONLY: Low / Medium / High
Followed by one sentence explaining the reasoning. Never use percentages.

🛡️ LEGAL OPTIONS
Numbered list of specific, actionable legal steps the user can take right now.

📑 REQUIRED EVIDENCE
Bullet list of documents, records, witnesses, or digital evidence that will strengthen the case.

⚠️ RISKS
Bullet list of legal risks, time limits (limitation periods), and critical mistakes to avoid.

🧠 FINAL ADVICE
2–3 sentences of the most important, practical next steps.

STRICT RULES:
- Base answers ONLY on retrieved legal context + established Indian law principles
- If the retrieved context does not cover something, explicitly say so
- Never fabricate section numbers or laws
- Never guarantee any outcome
- Keep language simple and practical
- If input is non-English, detect the language, acknowledge it, and respond in English"""


# ─────────────────────────────────────────────────────────────────
#  APP SETUP
# ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Nyay AI — Indian Legal RAG Assistant",
    version="2.0.0",
    description="RAG-powered legal analysis using Indian law PDFs"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Global RAG pipeline instance
rag = RAGPipeline()

def clean_output(text: str) -> str:
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F900-\U0001F9FF"
        "\U00002700-\U000027BF"
        "]+",
        flags=re.UNICODE,
    )

    text = emoji_pattern.sub("", text)

    replacements = {
        "CASE SUMMARY": "CASE SUMMARY",
        "APPLICABLE LAWS": "APPLICABLE LAWS",
        "ANALYSIS": "ANALYSIS",
        "POSSIBLE OUTCOMES": "POSSIBLE OUTCOMES",
        "ESTIMATED WINNING CHANCE": "ESTIMATED WINNING CHANCE",
        "LEGAL OPTIONS": "LEGAL OPTIONS",
        "REQUIRED EVIDENCE": "REQUIRED EVIDENCE",
        "RISKS": "RISKS",
        "FINAL ADVICE": "FINAL ADVICE",
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    return text.strip()
# ─────────────────────────────────────────────────────────────────
#  STARTUP — try to load saved index or default PDF
# ─────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    log.info("=== Nyay AI Backend Starting ===")

    # Try loading saved FAISS index first (fastest path)
    if rag.load_saved_index():
        log.info(f"✓ Loaded saved index: {rag.pdf_name}")
        return

    # If a default PDF is configured, build index from it
    if DEFAULT_PDF_PATH and Path(DEFAULT_PDF_PATH).is_file():
        log.info(f"Building index from default PDF: {DEFAULT_PDF_PATH}")
        try:
            rag.build_index(DEFAULT_PDF_PATH)
            log.info("✓ Default PDF indexed successfully.")
        except Exception as e:
            log.error(f"Failed to index default PDF: {e}")
    else:
        log.info("No saved index or default PDF found. Upload a PDF to begin.")


# ─────────────────────────────────────────────────────────────────
#  SCHEMAS
# ─────────────────────────────────────────────────────────────────
class CaseRequest(BaseModel):
    query: str

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        v = v.strip()
        if len(v) < 20:
            raise ValueError("Please describe your case in more detail (minimum 20 characters).")
        if len(v) > 6000:
            raise ValueError("Query too long (max 6000 characters).")
        return v


class AnalysisResponse(BaseModel):
    result: str
    cleaned: str   # 👈 NEW
    sources: list
    model: str
    chunks_used: int


class StatusResponse(BaseModel):
    is_ready: bool
    pdf_name: str
    num_pages: int
    num_chunks: int
    model: str


# ─────────────────────────────────────────────────────────────────
#  ENDPOINTS
# ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "Nyay AI RAG Backend", "rag_ready": rag.is_ready}


@app.get("/status", response_model=StatusResponse)
def get_status():
    s = rag.status()
    return StatusResponse(model=MODEL, **s)


@app.post("/upload-pdf")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Upload a PDF legal document. Triggers background indexing.
    Returns immediately; poll /status to track progress.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Check file size (max 50 MB)
    contents = await file.read()
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="PDF too large. Maximum size is 50 MB.")
    if len(contents) < 1000:
        raise HTTPException(status_code=400, detail="PDF appears to be empty or corrupted.")

    # Save to uploads dir
    save_path = UPLOADS_DIR / file.filename
    save_path.write_bytes(contents)
    log.info(f"PDF saved: {save_path} ({len(contents)//1024} KB)")

    # Index in background (non-blocking)
    background_tasks.add_task(_index_pdf_task, str(save_path))

    return {
        "message":  f"PDF '{file.filename}' received. Indexing in progress…",
        "filename": file.filename,
        "size_kb":  round(len(contents) / 1024, 1),
        "status":   "indexing",
    }


def _index_pdf_task(pdf_path: str):
    """Background task: build FAISS index from PDF."""
    try:
        log.info(f"Background indexing: {pdf_path}")
        rag.build_index(pdf_path)
        log.info("Background indexing complete.")
    except Exception as e:
        log.error(f"Background indexing failed: {e}")


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_case(request: CaseRequest):
    """
    Full RAG pipeline:
    1. Retrieve relevant legal chunks from FAISS
    2. Build context-aware prompt
    3. Call OpenRouter LLM
    4. Return structured analysis + sources
    """
    if not rag.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base not ready. Please upload a legal PDF first."
        )

    # ── Step 1: Retrieve context ──────────────────────────────
    try:
        chunks = rag.retrieve(request.query, k=6)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")

    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant legal context found for your query.")

    # ── Step 2: Build context block ───────────────────────────
    legal_context = rag.format_context(chunks)

    # ── Step 3: Build full prompt ─────────────────────────────
    user_message = f"""RETRIEVED LEGAL CONTEXT:
{legal_context}

USER'S LEGAL CASE:
{request.query}

Please analyze this case using the retrieved legal context above and Indian law principles."""

    # ── Step 4: Call OpenRouter ───────────────────────────────
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://nyayai.app",
        "X-Title":       "Nyay AI Legal Assistant",
    }
    payload = {
        "model":       MODEL,
        "messages":    [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        "temperature": 0.25,   # lower = more factual
        "max_tokens":  2200,
    }

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        result_text = data["choices"][0]["message"]["content"]
        cleaned_text = clean_output(result_text)

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="AI service timed out. Please try again.")
    except httpx.HTTPStatusError as e:
        code = e.response.status_code
        if code == 401:
            raise HTTPException(status_code=401, detail="Invalid OpenRouter API key.")
        if code == 429:
            raise HTTPException(status_code=429, detail="Rate limit reached. Please wait a moment.")
        raise HTTPException(status_code=502, detail=f"AI service error (HTTP {code}).")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

    # ── Step 5: Return response + sources ────────────────────
    sources = [
        {"page": c["page"], "source": c["source"], "preview": c["content"][:180] + "…"}
        for c in chunks
    ]

    return AnalysisResponse(
        result=result_text,      # emoji version (UI)
        cleaned=cleaned_text,    # clean version (download)
        sources=sources,
        model=MODEL,
        chunks_used=len(chunks),
    )


# ─────────────────────────────────────────────────────────────────
#  SERVE FRONTEND (production mode)
# ─────────────────────────────────────────────────────────────────
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.is_dir():
    app.mount("/app", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

    @app.get("/")
    def root():
        return FileResponse(str(frontend_dir / "index.html"))
