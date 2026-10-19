# Arabic Intelligent Knowledge Assistant

This project is the foundation for an Arabic Retrieval-Augmented Generation (RAG) system designed to answer questions from Arabic PDF documents with grounded citations and enterprise-friendly architecture.

## Document indexing API

The API accepts PDF uploads, extracts and normalizes their text, chunks each page,
generates dense embeddings, and stores the vectors and source metadata in Qdrant.
The structured extraction and document metadata are also persisted as JSON under
`data/processed/`.

Embeddings use `intfloat/multilingual-e5-small` (384 dimensions) and are stored in
the configured `QDRANT_COLLECTION_NAME` collection. The default collection is
`arabic_documents_e5_small`, sized for the 384-dimensional multilingual-E5-small
embeddings. This collection is separate from the existing 1024-dimensional
`arabic_documents` collection.

Start the API with `uvicorn app.main:app --reload`. Configure Qdrant with
`QDRANT_HOST`, `QDRANT_PORT`, and `QDRANT_COLLECTION_NAME`; the default connection is
`http://localhost:6333`. Install the packages in `requirements.txt`, including
`python-multipart`, to enable file uploads. The model is loaded by Sentence Transformers
on the first indexing or search request. The API console logs INFO-level stage markers for
upload, ingestion, chunking, embedding-model initialization, embedding, Qdrant indexing,
retrieval, and answer generation; the last marker before an unexpected process exit
identifies the operation in progress.

### Upload a PDF

`POST /api/v1/documents/upload` accepts a multipart form field named `file`. A
successful request returns HTTP 201 with the document ID, stored source metadata,
chunking statistics, and embedding statistics. PDFs with invalid content or no
extractable text return HTTP 422; unsupported file extensions return HTTP 415.
Uploads over 50 MiB return HTTP 413, and PDFs over 500 pages return HTTP 422 by
default. Configure `MAX_PDF_SIZE_BYTES` and `MAX_PDF_PAGES` to change these limits.
Embedding and Qdrant persistence run in bounded batches; INFO logs include sampled
RSS, peak RSS, and peak increase for extraction, chunking, embedding, and Qdrant stages.
The Streamlit client forwards the uploaded file stream directly instead of creating
an additional full-size in-memory byte copy.

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
	-F "file=@./arabic-report.pdf;type=application/pdf"
```

### Search indexed documents

Use the returned document metadata as part of the source citation in the retrieval
response. Search the indexed collection with:

```bash
curl -X POST http://localhost:8000/api/v1/retrieval/search \
	-H "Content-Type: application/json" \
	-d '{"query":"ما موضوع التقرير؟","top_k":5}'
```

### Generate a cited answer

Set `OPENAI_API_KEY` and `OPENAI_MODEL` in `.env` (the default model is
`gpt-4o-mini`). The answer endpoint retrieves relevant chunks, builds a grounded
Arabic prompt, and returns the answer with page-level source citations. If no
matching chunks are found, it returns an Arabic no-context response without
calling the LLM.

```bash
curl -X POST http://localhost:8000/api/v1/retrieval/answer \
	-H "Content-Type: application/json" \
	-d '{"query":"ما موضوع التقرير؟","top_k":5}'
```

The response includes `answer`, the configured `model`, and a `sources` array
containing each citation ID, document ID, filename, page number, passage, and
retrieval score. Configure `OPENAI_API_KEY` before requesting an answer with
retrieved context.

## Streamlit MVP

Run the API and UI in separate terminals. The UI defaults to
`http://127.0.0.1:8000`; set `API_BASE_URL` if the API is hosted elsewhere.

```bash
uvicorn app.main:app --reload
streamlit run app/frontend/streamlit_app.py
```

The UI uploads PDFs for indexing, displays page/chunk/vector counts, and sends
questions to the cited-answer endpoint. The retrieval preview can search the
indexed collection independently and show matching passages with source pages and
relevance scores. Answer generation requires a configured `OPENAI_API_KEY`;
indexing and retrieval also require the configured Qdrant service and model access.
