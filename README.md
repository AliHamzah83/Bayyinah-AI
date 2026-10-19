# Arabic RAG System

Arabic RAG System is a prototype for indexing Arabic PDF documents and answering
questions from their contents with retrieved source passages. It consists of a
FastAPI backend, a Streamlit interface, Sentence Transformers embeddings, and a
Qdrant vector store.

## Current capabilities

- **PDF upload:** Upload PDFs through the API or Streamlit. Native page text is
  extracted; image-only pages can be processed with Tesseract OCR and Arabic and
  English language data. Uploads are limited to 50 MiB and 500 pages by default.
- **Indexing:** Arabic text is normalized and split into chunks. Each chunk gets
  a dense embedding and source/page metadata, and is indexed in Qdrant. Structured
  extraction metadata is written locally as JSON under `data/processed/`.
- **Retrieval:** Search returns the highest-scoring passages with page and source
  metadata. The answer endpoint uses retrieved passages to generate an answer
  with citations.
- **Streamlit UI:** The browser interface uploads and indexes PDFs, shows
  indexing counts, previews retrieved passages, and submits questions for
  cited answers.
- **Embedding model:** `intfloat/multilingual-e5-small`, producing normalized
  384-dimensional vectors. The model is loaded on first use and may be
  downloaded by Sentence Transformers if it is not cached locally.

## Requirements

- Python 3.11 or newer.
- A Qdrant service reachable by the API (defaults to `http://localhost:6333`).
- An OpenAI API key for generated answers. PDF indexing and passage retrieval
  use the embedding model and Qdrant, but do not require the OpenAI key.
- Tesseract OCR with Arabic (`ara`) and English (`eng`) trained data for scanned
  PDFs. OCR is not needed for PDFs with extractable text.

## Run locally

Create and activate a virtual environment, then install dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and update the values as needed. In particular,
set `OPENAI_API_KEY` to use answer generation and configure `QDRANT_HOST`,
`QDRANT_PORT`, and `QDRANT_COLLECTION_NAME` for your Qdrant service. The default
collection is `arabic_documents_e5_small`. Do not commit `.env` or place real
credentials in `.env.example`.

Start the API and UI in separate terminals:

```powershell
uvicorn app.main:app --reload
streamlit run app/frontend/streamlit_app.py
```

The Streamlit UI defaults to the API at `http://127.0.0.1:8000`; set
`API_BASE_URL` if it is hosted elsewhere. The API exposes:

- `POST /api/v1/documents/upload` — multipart PDF upload using the `file` field.
- `POST /api/v1/retrieval/search` — retrieve passages for a JSON `query` and
  optional `top_k` (1–20, default 5).
- `POST /api/v1/retrieval/answer` — generate a cited answer for a JSON `query`
  and optional `top_k`.

The API docs are available at `http://127.0.0.1:8000/docs` when it is running.

## Known limitations

- This is an early-stage prototype, not a production-hardened service. It has no
  authentication, authorization, or multi-user document isolation.
- PDF is the only supported upload format. Scanned-document OCR requires a
  separately installed Tesseract executable and Arabic/English language data;
  extraction quality varies with document quality and layout.
- Retrieval is dense-vector search over the configured Qdrant collection; answer
  quality depends on the source text, chunking, embedding model, retrieved
  context, and configured OpenAI model.
- The embedding model may need to be downloaded on first use. Model inference
  and OCR can require substantial memory and processing time.
- Persisted extracted documents and vector data are local/runtime data and are
  intentionally excluded from version control. Back them up separately if
  needed.
- Default settings are for local development and are not a production security
  configuration.

## Project structure

See [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for the directory and
component overview.

## Tests

Run the test suite from the repository root:

```powershell
python -m pytest
```
