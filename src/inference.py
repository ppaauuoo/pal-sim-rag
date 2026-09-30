import asyncio
import inspect
import json
import os
import re

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

# nemotron leaks tool calls as content in several shapes: JSON dicts and
# <function=name><parameter=k>v</parameter></function> blocks
_FUNC_RE = re.compile(r"<function=(\w+)>(.*?)</function>", re.S)
_PARAM_RE = re.compile(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", re.S)


def _parse_content_tool_calls(content: str, fallback_q: str) -> list[tuple]:
    """Extract tool calls the model emitted as plain text instead of tool_calls."""
    calls: list[tuple[str, str, str]] = []
    text = content.strip().removeprefix("```json").removesuffix("```").strip()
    for i, match in enumerate(_FUNC_RE.finditer(text)):
        params = {m.group(1): m.group(2) for m in _PARAM_RE.finditer(match.group(2))}
        calls.append((match.group(1), json.dumps(params), f"call_{i}"))
    if calls:
        return calls
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, dict) and any(k in parsed for k in ("query", "q", "name")):
        if "query" in parsed and "q" not in parsed:
            parsed["q"] = parsed.pop("query")
        return [(parsed.get("name", "query"), json.dumps(parsed), "call_0")]
    return []


async def _model() -> str:
    """Resolve the model id lazily so a down LLM server doesn't kill app import."""
    global _model_id
    if _model_id is None:
        _model_id = os.environ.get("LLM_MODEL", "nemotron-3-nano:4b")
    return _model_id


async def _execute_tool(name: str, raw_args: str, fallback_q: str):
    """Run a tool call; traced as a TOOL span even when the model leaked the call as content."""
    with mlflow.start_span(name=f"tool:{name}", span_type=SpanType.TOOL) as span:
        fn = TOOL_FUNCTIONS.get(name)
        try:
            args = json.loads(raw_args) if fn else {}
            if fn and isinstance(args, dict):
                # nemotron sometimes emits empty/extra args; keep valid ones,
                # default q to the user's question
                valid = inspect.signature(fn).parameters
                args = {k: v for k, v in args.items() if k in valid}
                args.setdefault("q", fallback_q)
            span.set_inputs({"name": name, "args": args})
            result = (
                await asyncio.to_thread(fn, **args)
                if fn
                else {"error": f"unknown function: {name}"}
            )
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            result = {"error": str(e)}
        if isinstance(result, list):  # drop embedding vectors — noise for the LLM
            result = [{k: v for k, v in row.items() if k != "vector"} for row in result]
        span.set_outputs({"results": result})
        return result


@get("/infer")
@mlflow.trace(name="infer", span_type=SpanType.CHAIN)
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
            calls = _parse_content_tool_calls(msg.content, q)

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

    # ponytail: model burned all rounds on tool calls — force a final answer
    if not (msg.content or "").strip():
        messages.append(
            {
                "role": "user",
                "content": "Answer the original question now using the search results above. Do not call tools.",
            }
        )
        response = await client.chat.completions.create(model=model, messages=messages)
        msg = response.choices[0].message

    return msg.content or ""
