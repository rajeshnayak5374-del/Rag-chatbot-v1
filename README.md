# Rag-chatbot-v1
Description: SAGE - Enterprise Document Intelligence
Enterprise RAG-based document Q&A system built with
Python, LangChain, ChromaDB, FastAPI, and Groq LLaMA.

## Project Structure
sage-project/
├── rag_pipeline/
│ ├── embeddings.py # Embedding model wrapper
│ ├── ingest.py # Document ingestion pipeline
│ ├── retrieve.py # Retrieval and generation
│ └── prompts.py # Prompt templates
├── api/
│ └── main.py # FastAPI endpoints
├── tests/
│ ├── test_ingest.py # Ingestion tests
│ └── test_retrieve.py # Retrieval tests
├── sample_docs/ # Sample documents
├── docs/
│ └── defect_log.md # Defect tracking
├── .env # Environment variables
└── requirements.txt # Dependencies


## Tech Stack
- Python 3.13
- LangChain — RAG orchestration
- ChromaDB — Vector database
- HuggingFace MiniLM — Embeddings
- Groq LLaMA — LLM
- FastAPI — REST API

## Setup
```bash
git clone https://github.com/YOUR_USERNAME/sage-project.git
cd sage-project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Environment Variables
Create `.env` file:


## API Endpoints
- POST /ingest — Upload document
- POST /ask — Ask question
- GET /health — System status

## Team
- Senior AI Engineer — Architecture and reviews
- Junior AI Engineer — RAG pipeline and FastAPI