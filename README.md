# ⚖️ Nyay AI — Indian Legal RAG Assistant

**RAG-powered** web application that reads your Indian law PDFs and gives grounded,
context-aware legal analysis — not generic answers.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│              FastAPI Backend            │
│                                         │
│  1. Receive query                       │
│  2. Embed query (sentence-transformers) │
│  3. FAISS similarity search             │  ←── PDF chunks indexed
│     → retrieve top-6 law excerpts       │
│  4. Build context prompt                │
│  5. Call OpenRouter (GPT-4o-mini)       │
│  6. Return analysis + sources           │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│             HTML Frontend               │
│  - Upload PDF                           │
│  - Describe case (any language)         │
│  - View 9-section structured analysis  │
│  - See which PDF pages were retrieved  │
│  - Download as PDF / Copy              │
└─────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
nyay-rag/
├── backend/
│   ├── main.py            ← FastAPI app (upload, analyze, status endpoints)
│   ├── rag_pipeline.py    ← RAG engine (PDF → chunks → FAISS → retrieval)
│   ├── uploads/           ← Uploaded PDFs saved here (auto-created)
│   └── vector_db/         ← FAISS index saved here (auto-created)
├── frontend/
│   └── index.html         ← Full SaaS frontend (no build step needed)
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Run

### Prerequisites
- Python 3.9+
- OpenRouter API key (free at https://openrouter.ai)

---

### Step 1 — Clone / Download Project

```bash
cd nyay-rag
```

---

### Step 2 — Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

---

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> ⏱ First install downloads `sentence-transformers` model (~90 MB). One-time only.

---

### Step 4 — Set API Key

```bash
# macOS / Linux
export OPENROUTER_API_KEY="sk-or-xxxxxxxxxxxxxxxxxxxx"

# Windows CMD
set OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxxxxxx

# Windows PowerShell
$env:OPENROUTER_API_KEY="sk-or-xxxxxxxxxxxxxxxxxxxx"
```

**OR** edit `backend/main.py` line 20:
```python
OPENROUTER_API_KEY = "sk-or-xxxxxxxxxxxxxxxxxxxx"
```

---

### Step 5 — Start Backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

You should see:
```
INFO: Application startup complete.
INFO: Uvicorn running on http://127.0.0.1:8000
```

---

### Step 6 — Open Frontend

Simply open `frontend/index.html` in your browser.

**Or** serve it:
```bash
cd frontend
python -m http.server 3000
# Open http://localhost:3000
```

---

### Step 7 — Upload a Legal PDF & Analyze

1. **Upload PDF** — Drag & drop any Indian law PDF (Constitution, IPC, IT Act, etc.)
2. **Wait for indexing** — Status shows "Indexing…" then "✓ PDF loaded" (1–3 min first time)
3. **Describe your case** — Any language (English, Hindi, Gujarati, Hinglish…)
4. **Click Analyze** — Get structured 9-section analysis with source page references

---

## 🔁 Auto-load a PDF on Startup

Set the `DEFAULT_PDF_PATH` env variable to skip manual upload:

```bash
export DEFAULT_PDF_PATH="/path/to/constitution_of_india.pdf"
uvicorn main:app --reload --port 8000
```

The FAISS index is also **cached to disk** — restart won't re-index if the same PDF is used.

---

## 🌐 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health`      | Server health check |
| `GET`  | `/status`      | Index status (pdf name, pages, chunks) |
| `POST` | `/upload-pdf`  | Upload PDF file (multipart/form-data) |
| `POST` | `/analyze`     | Analyze a legal case query (JSON body: `{query}`) |

**API Docs:** http://localhost:8000/docs

---

## ✨ Features

| Feature | Details |
|---------|---------|
| 📄 PDF Upload | Drag & drop, max 50 MB |
| 🧠 Local Embeddings | `sentence-transformers/all-MiniLM-L6-v2` — free, no API key |
| 🔍 Semantic Search | FAISS cosine similarity, top-6 chunks retrieved |
| 💾 Index Caching | FAISS index saved to disk — survives restart |
| 🌐 Multi-language | Query in any language, response in English |
| 📋 9-section Output | Summary → Laws → Analysis → Outcomes → Winning Chance → Options → Evidence → Risks → Advice |
| 📎 Source Citations | Shows which PDF pages were used |
| 🌙 Dark/Light mode | With persistence |
| 📄 PDF Export | Download full analysis as PDF |
| 📁 History | Last 15 cases in browser local storage |

---

## 📚 Recommended PDFs to Upload

- **Constitution of India** — bare act PDF
- **Indian Penal Code (IPC) 1860** — with all sections
- **Code of Criminal Procedure (CrPC)**
- **IT Act 2000**
- **Consumer Protection Act 2019**
- **Hindu Marriage Act / Family Laws**

Download from: https://indiacode.nic.in or https://legislative.gov.in

---

## 🚀 Production Deployment

### Backend (Railway / Render / VPS)
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
Set `OPENROUTER_API_KEY` and optionally `DEFAULT_PDF_PATH` as env vars.

### Frontend
Update `const API = 'http://localhost:8000'` in `index.html` to your backend URL.
Deploy to Netlify / Vercel (static file).

---

## ⚠️ Disclaimer

This application provides AI-generated legal information grounded in uploaded PDF documents,
for **educational purposes only**. It is NOT a substitute for professional legal advice.
Always consult a licensed advocate before taking legal action.
