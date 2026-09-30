import asyncio
import inspect
import json
import os

from openai import AsyncOpenAI
from litestar import get

from ingest import search

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("pal-sim-rag")
mlflow.openai.autolog()

# ollama serves the OpenAI-compatible API on 11434; override with LLM_BASE_URL
client = AsyncOpenAI(
    base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
    api_key="dummy",
)

TOOL_FUNCTIONS = {"query": search}

tools = [
    {
        "type": "function",
        "function": {
            "name": "query",
            "description": "Search the ingested knowledge base (Obsidian vault) for relevant notes",
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {"type": "string", "description": "Search query"},
                },
                "required": ["q"],
            },
        },
    },
]

_model_id: str | None = None


async def _model() -> str:
    """Resolve the model id lazily so a down LLM server doesn't kill app import."""
    global _model_id
    if _model_id is None:
        _model_id = os.environ.get("LLM_MODEL", "nemotron-3-nano:4b")
    return _model_id


@get("/infer")
async def infer(q: str) -> str:
    model = await _model()
    messages: list = [{"role": "user", "content": q}]

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    # ponytail: loop until the model answers without calling tools (max 3 rounds)
    for _ in range(3):
        msg = response.choices[0].message

        # collect tool calls; nemotron sometimes emits them as plain JSON content
        calls = [
            (c.function.name, c.function.arguments, c.id)
            for c in (msg.tool_calls or [])
        ]
        if not calls and msg.content:
            try:
                text = (
                    msg.content.strip()
                    .removeprefix("```json")
                    .removesuffix("```")
                    .strip()
                )
                parsed = json.loads(text)
                if isinstance(parsed, dict) and any(
                    k in parsed for k in ("query", "q", "name")
                ):
                    if "query" in parsed and "q" not in parsed:
                        parsed["q"] = parsed.pop("query")
                    calls = [(parsed.get("name", "query"), json.dumps(parsed), "call_0")]
            except json.JSONDecodeError:
                pass

        if not calls:
            break

        messages.append(msg.model_dump(exclude_none=True))
        for name, raw_args, call_id in calls:
            fn = TOOL_FUNCTIONS.get(name)
            try:
                args = json.loads(raw_args) if fn else {}
                if fn and isinstance(args, dict):
                    # nemotron sometimes emits empty/extra args; keep valid ones,
                    # default q to the user's question
                    valid = inspect.signature(fn).parameters
                    args = {k: v for k, v in args.items() if k in valid}
                    args.setdefault("q", q)
                result = (
                    await asyncio.to_thread(fn, **args)
                    if fn
                    else {"error": f"unknown function: {name}"}
                )
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                result = {"error": str(e)}
            if isinstance(result, list):  # drop embedding vectors — noise for the LLM
                result = [
                    {k: v for k, v in row.items() if k != "vector"} for row in result
                ]
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(result, default=str),
                }
            )
        response = await client.chat.completions.create(model=model, messages=messages)

    return msg.content or ""
