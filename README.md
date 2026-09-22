# RAG From Scratch — AI Documentation Assistant

A production-oriented **RAG + LangGraph agent** built with FastAPI. The system answers questions from technical documentation using hybrid retrieval, reranking, persistent conversation memory, and structurally enforced grounding.

Built from scratch with a focus on understanding **RAG architecture, agent orchestration, retrieval quality, failure modes, and production hardening**.

## Architecture

```text
                         User
                          │
                          ▼
                     Guardrails
                          │
                          ▼
                    Session / DB
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
        /chat/stream           /chat/agent/stream
          Plain RAG              LangGraph Agent
              │                       │
       Query Rewriting              Router
              │                       │
       Hybrid Retrieval        ┌──────┴──────┐
              │                │             │
          Reranking         No Search      Search
              │                              │
       Parent Recovery                  Documentation
              │                              Tool
       Context Selection                     │
              │                         Strict Answer
              ▼                              │
          Groq LLM                    Supplement Check
                                             │
                                      Optional General
                                           Answer
                                             │
                                             ▼
                                          Finalize
```

## Tech Stack

- **Python / FastAPI**
- **Groq** — `openai/gpt-oss-120b`
- **LangGraph** — agent orchestration
- **LlamaIndex** — document processing / retrieval components
- **FastEmbed + BAAI BGE** — local embeddings
- **Qdrant** — vector database
- **BM25 + Reciprocal Rank Fusion** — hybrid retrieval
- **Jina Reranker** — reranking
- **PostgreSQL** — sessions and agent memory
- **Langfuse** — observability
- **DeepEval** — RAG evaluation
- **Docker / GitHub Actions**

No OpenAI API dependency in the application.

## RAG Pipeline

```text
Question
   ↓
Query Rewriting
   ↓
Dense Retrieval ──┐
                  ├─→ RRF Fusion → Jina Reranker
BM25 Retrieval ───┘
                           ↓
                    Parent Recovery
                           ↓
                    Context Selection
                           ↓
                       Groq LLM
                           ↓
                      Stream Answer
```

The system uses local embeddings to reduce API costs while retaining the existing retrieval quality.

A retrieval stress test also identified failures involving heavy typos and similar-but-wrong frameworks. Typo correction was implemented as a **threshold-gated fallback**, avoiding an additional LLM call on normal queries.

## Agent Architecture

The agent uses a hand-built LangGraph `StateGraph` rather than `create_react_agent`.

The graph separates documentation-grounded answers from general supplemental information:

```text
START
  ↓
Router
  ↓
Search Documentation
  ↓
Strict Documentation Answer
  ↓
Complete?
 ┌───────┴───────┐
Yes             No
 │               │
 ▼               ▼
Finalize     General Answer
                 │
                 ▼
              Finalize
```

The final response is assembled by application code rather than allowing the LLM to self-label which information came from documentation.

This was introduced after testing exposed a false-citation/grounding issue with the earlier ReAct implementation.

## Document Support

Supports:

- Markdown
- PDF
- DOCX

Includes:

- Upload
- Incremental ingestion
- Deduplication
- Deletion
- Retrieval validation

## Production Hardening

- Input guardrails on both chat routes
- Admin-key authentication for document management
- `hmac.compare_digest` for key comparison
- `15/min` IP-based chat rate limiting
- Upload size limits
- Server-generated session IDs
- Production API documentation disabled
- Persistent LangGraph memory with PostgreSQL
- Langfuse tracing

## Evaluation

A GitHub Actions regression suite runs against a small golden dataset using DeepEval.

Current evaluation includes:

- Faithfulness
- Answer relevancy

The evaluation judge is separated from the production LLM.

The current free-tier CI setup successfully runs Faithfulness evaluation, while the Ollama `llama3.2:1b` judge currently encounters cancellation during `AnswerRelevancyMetric`. This remains an open CI evaluation issue rather than being hidden by lowering the regression threshold.

Target threshold:

```text
0.70
```

## Project Structure

```text
app/
├── ai/
│   ├── agent.py
│   ├── agent_tools.py
│   ├── eval_model.py
│   ├── llama.py
│   └── query_rewriter.py
├── core/
│   ├── config.py
│   ├── exceptions.py
│   ├── security.py
│   └── rate_limit.py
├── routers/
│   ├── chat.py
│   └── documents.py
├── scripts/
│   ├── ingest_docs.py
│   ├── test_retrieval.py
│   ├── stress_test_retrieval.py
│   └── run_eval.py
├── services/
│   ├── rag_service.py
│   ├── retrieval_service.py
│   ├── agent_service.py
│   └── document_service.py
└── tests/
    └── data/
        └── golden_dataset.json

.github/
└── workflows/
    └── rag_evals.yml
```

## Running Locally

```bash
git clone <repository-url>
cd rag-from-scratch

docker compose up -d

python -m app.scripts.ingest_docs

uvicorn app.main:app --reload
```

Configure the required API keys and database/Qdrant settings through environment variables.

## Current Status

### Done

- Full RAG pipeline
- Hybrid retrieval + reranking
- Multi-format ingestion
- Streaming responses
- LangGraph agent
- Persistent agent memory
- Grounding enforcement
- Guardrails
- Security hardening
- Retrieval stress testing
- Langfuse observability
- DeepEval evaluation setup

### Next

- Complete deployment using managed PostgreSQL + Qdrant Cloud
- Finish CI evaluation setup
- Final CORS/error-handling cleanup
- Decide whether to retain the standalone `/chat/stream` route alongside the agent route
