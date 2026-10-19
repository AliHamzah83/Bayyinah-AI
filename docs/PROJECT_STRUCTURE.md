# Project structure

```text
app/
  api/
    dependencies/   API service construction and dependency injection
    routers/        PDF indexing and retrieval endpoints
  frontend/         Streamlit user interface
  main.py           FastAPI application entry point
src/
  core/             Settings, shared models, enums, and exceptions
  ingestion/        PDF text/OCR extraction and Arabic normalization
  storage/          Qdrant storage and retrieval implementation
  answering.py      Grounded answer generation and citations
  chunking.py       Document text chunking
  embeddings.py     Multilingual E5 embedding generation
  indexing.py       Document indexing orchestration
tests/
  unit/             Unit and API/integration-style tests
docs/               Project documentation
requirements.txt    Runtime and development Python dependencies
pyproject.toml      Project metadata and Python tooling configuration
```

Runtime content such as uploads, extracted JSON, local vector-store files,
model weights, caches, and temporary files is not source code and should not be
committed. The application creates the relevant data directories as needed.
