# Simple Local RAG

A local-first retrieval-augmented generation (RAG) API for searching Markdown and text files in a knowledge base and using the retrieved notes to answer questions.

## Stack

- **Python 3.14** — application runtime.
- **uv** — dependency and project management, with dependencies declared in `pyproject.toml` and locked in `uv.lock`.
- **Litestar** — HTTP API framework; **Uvicorn** runs the API server.
- **Chonkie** — document fetching, text processing, chunking, and the ingestion pipeline.
- **LanceDB** — local vector database and hybrid text/vector search, stored in `.lancedb/`.
- **Model2Vec** — embeddings using `mikeee/m2v-gemma-embedding-300m`.
- **LanceDB CrossEncoderReranker** — reranks retrieved passages with `BAAI/bge-reranker-v2-m3`.
- **Pydantic AI** — agent orchestration and knowledge-base search tool calls.
- **Ollama** — local LLM provider through its OpenAI-compatible API. The default model is `nemotron-3-nano:4b`.
- **MLflow** — experiment tracking and Pydantic AI tracing, configured for `http://localhost:5000`.
- **SQLite** — local conversation history in `history.db`.
- **pathspec** — Git-ignore-style filtering via `.ragignore` files.

## How it works

1. The ingestion endpoint scans `knowledge_base/` for `.md` and `.txt` files, applying `.ragignore` rules.
2. Chonkie chunks the text, creates Model2Vec embeddings, and stores the chunks in LanceDB.
3. Search combines full-text and vector retrieval, then reranks the results and returns the most relevant matches.
4. The inference endpoint uses Pydantic AI with the local Ollama model. The agent searches the knowledge base before answering, and conversation history is persisted in SQLite by session.
5. MLflow records inference traces and experiment data locally.

## Run locally

Install the project dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Start an MLflow tracking server in one terminal:

```bash
run_mlflow.bat
```

In another terminal, start the API:

```bash
uv run api
```

The API listens on `http://127.0.0.1:8123`. Ollama must also be running locally and have the configured model available. The embedding and reranker models need to be available in the local model cache.

## API endpoints

All routes are GET endpoints:

- `/` — health/basic greeting response.
- `/ingests` — rebuild the LanceDB knowledge table from `knowledge_base/`.
- `/query?q=...` — search the knowledge base.
- `/infer?q=...&session_id=...` — answer a question using retrieved notes; `session_id` defaults to `default`.
- `/test?q=...` — exercise a small embedding/vector-search example.

## Configuration

The LLM model and Ollama-compatible API base URL can be overridden with environment variables:

- `LLM_MODEL` — defaults to `nemotron-3-nano:4b`.
- `LLM_BASE_URL` — defaults to `http://localhost:11434/v1`.
