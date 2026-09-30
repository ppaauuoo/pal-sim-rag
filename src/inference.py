import asyncio
import os
import sqlite3

from litestar import get
from mlflow.entities import SpanType
from pydantic_ai import Agent, ModelMessagesTypeAdapter
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_core import to_json

from ingest import search

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("pal-sim-rag")
mlflow.pydantic_ai.autolog()

# ollama serves the OpenAI-compatible API on 11434; override with LLM_BASE_URL
model = OpenAIChatModel(
    os.environ.get("LLM_MODEL", "nemotron-3-nano:4b"),
    provider=OpenAIProvider(
        base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
        api_key="dummy",
    ),
)


def _search(q: str) -> list[dict]:
    """Search the ingested knowledge base (Obsidian vault) for relevant notes."""
    rows = search(q)
    return [{k: v for k, v in row.items() if k != "vector"} for row in rows]


def _trim(messages: list, limit: int = 20) -> list:
    """Keep the last `limit` messages, cut at the oldest user prompt so we
    never split a tool-call/response pair (server would reject that)."""
    if len(messages) <= limit:
        return messages
    cut = len(messages) - limit
    while cut < len(messages):
        m = messages[cut]
        if m.kind == "request" and any(
            p.part_kind == "user-prompt" for p in m.parts
        ):
            break
        cut += 1
    return messages[cut:]


agent = Agent(
    model,
    instructions=(
        "Always call the query tool to search the knowledge base before "
        "answering. Never answer from memory. If the search results don't "
        "contain the answer, say the information is not in the knowledge "
        "base — do not guess."
    ),
    tools=[_search],
    capabilities=[ProcessHistory(_trim)],
)

# ponytail: one sqlite file, one connection; per-session history via
# ModelMessagesTypeAdapter. Swap for a real DB if this ever leaves localhost.
_db = sqlite3.connect("history.db", check_same_thread=False)
_db.execute(
    "CREATE TABLE IF NOT EXISTS history ("
    "session_id TEXT PRIMARY KEY, messages TEXT NOT NULL)"
)


def _load_history(session_id: str) -> list:
    row = _db.execute(
        "SELECT messages FROM history WHERE session_id = ?", (session_id,)
    ).fetchone()
    return ModelMessagesTypeAdapter.validate_json(row[0]) if row else []


def _save_history(session_id: str, messages: list) -> None:
    _db.execute(
        "INSERT INTO history (session_id, messages) VALUES (?, ?) "
        "ON CONFLICT(session_id) DO UPDATE SET messages = excluded.messages",
        (session_id, to_json(_trim(messages)).decode()),
    )
    _db.commit()


@get("/infer")
@mlflow.trace(name="infer", span_type=SpanType.CHAIN)
async def infer(q: str, session_id: str = "default") -> str:
    history = await asyncio.to_thread(_load_history, session_id)
    result = await agent.run(q, message_history=history)
    await asyncio.to_thread(_save_history, session_id, result.all_messages())
    return result.output
