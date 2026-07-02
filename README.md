# Multi-Document Comparator

A RAG-powered Streamlit application for comparing research papers and technical documents. Upload PDFs and ask questions to find similarities, differences, contradictions, or get a full comparison across documents.

## Features

- **PDF Ingestion** — Load and process multiple PDF documents with semantic chunking
- **Vector Search** — Embed chunks via Ollama (mxbai-embed-large) and store in ChromaDB for fast retrieval
- **Intent-Aware Queries** — Automatically detects whether you want similarities, differences, contradictions, missing info, summary, or a full comparison
- **LLM-Powered Answers** — Generates grounded answers via OpenRouter (or any OpenAI-compatible endpoint) using only retrieved context
- **MMR Reranking** — Re-ranks retrieved chunks with Maximum Marginal Relevance for diversity
- **Security Layer** — Input validation, prompt injection guard, and output filtering
- **Dark Mode** — Built-in dark mode toggle
- **File Management** — Upload, view, and delete PDFs from the UI

## Architecture

```
app.py              → Streamlit entry point
config.py           → Centralised configuration from .env
rag/                → RAG pipeline
  ├── loader.py     → PDF loading (pdfplumber)
  ├── chunker.py    → Semantic chunking with configurable token limits
  ├── embeddings.py → Embedding generation via Ollama
  ├── vectordb.py   → ChromaDB vector store
  ├── retriever.py  → Retrieval + MMR reranking
  ├── reranker.py   → MMR reranker
  ├── prompts.py    → Intent classification + system/user prompts
  ├── llm.py        → LLM interface (OpenRouter / OpenAI)
  └── pipeline.py   → Orchestration (ingest → retrieve → generate)
frontend/           → Streamlit UI
  ├── sidebar.py   → Navigation sidebar
  ├── upload.py    → Upload page + file management
  ├── compare.py   → Query page + ingestion controls
  └── components.py→ Shared UI components
security/           → Security modules
  ├── input_validator.py
  ├── prompt_guard.py
  └── output_filter.py
utils/              → Logger
chroma_db/          → Vector database persistence
uploaded_files/     → Uploaded PDF storage
logs/               → Application logs
```

## Quick Start

### Prerequisites

- Python 3.12+
- [Ollama](https://ollama.com/) running locally with an embedding model (default: `mxbai-embed-large`)
- An OpenRouter API key (or any OpenAI-compatible LLM endpoint)

### Setup

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd Multi-Document-Comparator

# 2. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
#    Edit .env with your API keys and settings
```

### Configure `.env`

```env
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=openai/gpt-4o-mini
OLLAMA_BASE_URL=http://localhost:11434
EMBED_MODEL=mxbai-embed-large:latest
```

### Run

```bash
.\venv\Scripts\streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Usage

1. **Upload Documents** — Select one or more PDF files (max 10 files, 50 MB each)
2. **Ingest** — Go to the *Compare Documents* page and click *Ingest All Documents*
3. **Ask Questions** — Type a question about the documents. The system detects intent automatically:
   - "What are the similarities between these papers?"
   - "What are the key differences?"
   - "Are there any contradictions?"
   - "Summarise both documents"
   - "Compare the architectures"

## Configuration

Key settings in `.env`:

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | — | LLM API key |
| `OPENROUTER_MODEL` | `openai/gpt-oss-120b:free` | Model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `EMBED_MODEL` | `mxbai-embed-large:latest` | Embedding model |
| `TOP_K` | `8` | Retrieved chunks |
| `FETCH_K` | `20` | Candidates for MMR |
| `MAX_CONTEXT_TOKENS` | `12000` | Max tokens sent to LLM |
| `TEMPERATURE` | `0` | LLM temperature |
| `CHUNK_SIZE_TOKENS` | `500` | Chunk size in tokens |
| `MAX_FILE_SIZE_MB` | `50` | Per-file upload limit |
| `MAX_FILES` | `10` | Max files per session |
