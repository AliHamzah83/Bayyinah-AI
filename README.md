# Bayyinah AI
An Arabic Retrieval-Augmented Generation (RAG) system that enables users to upload Arabic PDF documents and ask natural-language questions with source-grounded answers.

##  Overview
Bayyinah AI is designed to improve information retrieval from Arabic documents by combining:

- Arabic document processing
- Text chunking
- Semantic embeddings
- Vector search with Qdrant
- Local Large Language Models (LLMs)
- Retrieval-Augmented Generation (RAG)

##  Features
- ✅ Arabic PDF document ingestion
- ✅ Automatic text extraction
- ✅ Intelligent text chunking
- ✅ Embedding generation
- ✅ Qdrant vector database integration
- ✅ Semantic retrieval
- ✅ Source-aware answers
- ✅ Arabic question answering
- ✅ FastAPI backend
- ✅ Streamlit frontend
- ✅ Local AI execution using Ollama
## System Architecture

```text
PDF Documents
      │
      ▼
Text Extraction
      │
      ▼
Chunking
      │
      ▼
Embeddings
      │
      ▼
Qdrant Vector Store
      │
      ▼
Retriever
      │
      ▼
LLM (Qwen via Ollama)
      │
      ▼
Grounded Answer + Sources
```
## Technology Stack
### Backend
- FastAPI
- Python
### Frontend
- Streamlit
### Vector Database
- Qdrant
### Database
- PostgreSQL
### Local LLM Inference
- Ollama
- Qwen 2.5
## Project Structure
backend/
frontend/
app/
services/
core/
database/
vector_store/
embeddings/
documents/
config/
## Example Workflow
Upload an Arabic PDF document.
Extract document text.
Split text into semantic chunks.
Generate embeddings.
Store vectors in Qdrant.
Ask a question in Arabic.
Retrieve relevant chunks.
Generate a grounded answer with citations.
## Example Questions
Who is the document author?
What is the applicant's current position?
What are the key qualifications mentioned?
Summarize the document.
Extract contact information.
## Current Status
Current version includes:
End-to-end Arabic RAG pipeline
PDF ingestion
Embedding generation
Vector retrieval
Grounded answer generation
Local model deployment using Ollama
## Future improvements:
Hybrid search
Reranking
Multi-document retrieval
Conversation memory
Evaluation dashboard
Docker deployment
## Why Bayyinah AI?
Many Arabic documents contain valuable information that is difficult to search through using traditional keyword-based systems.

Bayyinah AI addresses this challenge by combining semantic search and Retrieval-Augmented Generation to provide accurate, document-grounded answers in Arabic.

## Author
Dr. Ali A. Hamzah

AI Researcher | NLP |  Applied AI Engineering | Information Security 
## Update

Last updated 19-Nov-2026