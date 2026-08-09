# AI Knowledge Base Assistant

An AI assistant that answers questions from company documents using
Retrieval-Augmented Generation (RAG) — query expansion, hybrid search
(vector + keyword), reranking, structured outputs, and guardrails, served
through a FastAPI backend.

Built as part of a GenAI development learning program (Week 5: Enterprise
GenAI Development).

## Features

- Multi-query expansion + hybrid search (vector + BM25) + cross-encoder reranking
- Structured, validated answers with source citations (`answer`, `confidence`, `sources`)
- Input guardrails (prompt-injection detection, off-topic filtering) and output guardrails (fabricated-source detection)
- FastAPI backend with `/ask` and `/health` endpoints and auto-generated docs at `/docs`
- Structured logging with per-request correlation IDs
- Evaluation harness with retrieval + LLM-judged quality metrics
- pytest test suite for guardrails and chunking logic

## Tech Stack

| Component | Choice |
|---|---|
| LLM & Embeddings | Google Gemini (`gemini-3.6-flash`, `gemini-embedding-001`) |
| Vector store | ChromaDB |
| Keyword search | BM25 (`rank_bm25`) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| API framework | FastAPI + Uvicorn |
| Testing | pytest |

## Project Structure

```
app/
├── main.py            # FastAPI app entrypoint
├── api/                # Routes and request/response models
├── core/               # Config, logging, exceptions
├── models/             # LLM structured-output schemas
└── services/            # RAG pipeline: ingestion, retrieval, generation, guardrails
eval/                   # Evaluation dataset, metrics, and reports
scripts/                # build_index.py, evaluate.py
tests/                  # pytest unit tests
data/documents/          # Source PDFs
```

## Setup

**1. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Configure your API key**
```bash
cp .env.example .env
```
Add your Gemini API key to `.env` (free key from [Google AI Studio](https://aistudio.google.com/apikey)):
```
GOOGLE_API_KEY=your_actual_api_key_here
```

**4. Add your documents**

Place PDFs in `data/documents/`. This project ships with two sample PDFs for a fictional company, Acme Corp (an HR policy handbook and a product FAQ) — swap in real documents to use it for an actual knowledge base.

**5. Build the vector index**
```bash
python scripts/build_index.py
```
Re-run this whenever the documents in `data/documents/` change.

## Running the API

```bash
uvicorn app.main:app --reload
```

Visit `http://localhost:8000/docs` for interactive API docs, or test directly:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How many days of PTO do I get per year?"}'
```

```json
{
  "query": "How many days of PTO do I get per year?",
  "answer": "Employees accrue 18 days of paid time off per year.",
  "answer_found": true,
  "confidence": "high",
  "sources": [{"source": "hr_policy_handbook.pdf", "page": 1}],
  "latency_ms": 2140.3
}
```

## Testing

```bash
pytest tests/ -v              # fast unit tests, no API calls
python scripts/evaluate.py    # full pipeline evaluation against a golden dataset
```

The evaluation script reports retrieval hit rate, mean reciprocal rank, answer correctness, and faithfulness (hallucination check), and saves results to `eval/reports/`.

