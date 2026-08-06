# Semantic Search API

A semantic search engine built from scratch using FastAPI, Jina AI Embeddings, and Qdrant.

Instead of searching by keywords, this project retrieves documents based on semantic meaning using vector embeddings.

## Features

- Load Markdown documents
- Split documents into overlapping chunks
- Generate embeddings with Jina AI
- Store vectors and metadata in Qdrant
- Perform semantic similarity search
- Automatic collection recreation during ingestion

## Tech Stack

- Python
- FastAPI
- Jina AI Embeddings (`jina-embeddings-v3`)
- Qdrant
- Pydantic
- AsyncIO

## Project Structure

```
app/
├── ai/
│   └── client.py
├── db/
│   └── qdrant.py
├── schemas/
│   ├── document.py
│   └── chunk.py
├── services/
│   ├── loader.py
│   ├── chunker.py
│   └── embedder.py

scripts/
├── ingest_docs.py
└── search.py

data/
├── fastapi/
├── python/
└── sql/
```

## Pipeline

```
Markdown Documents
        │
        ▼
Document Loader
        │
        ▼
Chunk Documents
        │
        ▼
Generate Embeddings
        │
        ▼
Store in Qdrant
        │
        ▼
Semantic Search
```

## Example

Query:

```
How do FastAPI dependencies work?
```

Top Result:

```
File: dependencies.md

FastAPI features a powerful Dependency Injection system that allows developers to share logic, database sessions, and security checks across multiple routes...
```

## Running the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file.

```
JINA_API_KEY=your_api_key
```

### 3. Start Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 4. Ingest documents

```bash
python -m scripts.ingest_docs
```

This will:

- Recreate the collection
- Load all Markdown documents
- Chunk documents
- Generate embeddings
- Upload vectors into Qdrant

### 5. Search

```bash
python -m scripts.search
```

Example output:

```
================================================================================
Score: 0.818
Source: fastapi
File: dependencies.md
Chunk: 0

FastAPI features a powerful Dependency Injection system...
```

## What I Learned

- Vector embeddings
- Semantic similarity search
- Qdrant collections and payloads
- Metadata filtering
- Document chunking with overlap
- Batch embedding generation
- Building an end-to-end document ingestion pipeline
- Separating ingestion, embedding, and retrieval into reusable services

## Future Improvements

- Sentence-aware chunking
- Metadata filtering during search
- Hybrid search
- PDF and DOCX document support
- RAG question answering using retrieved context
- Docker Compose
- FastAPI API endpoints
