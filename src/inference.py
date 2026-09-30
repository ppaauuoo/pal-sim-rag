from openai import OpenAI
from ingest import query
from litestar import get
import json

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

tool_functions = {"query": query}

tools = [
    {
        "type": "function",
        "function": {
            "name": "query",
            "description": "Get knowledge",
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {"type": "string", "description": "Query for knowledge"},
                },
                "required": ["q"],
            },
        },
    },
]


@get("/infer")
def infer(q: str):
    response = client.chat.completions.create(
        model=client.models.list().data[0].id,
        messages=[{"role": "user", "content": q}],
        tools=tools,
        tool_choice="auto",
    )

    tool_call = response.choices[0].message.tool_calls[0].function
    print(f"Function called: {tool_call.name}")
    print(f"Arguments: {tool_call.arguments}")
    print(
        f"Result: {tool_functions[tool_call.name](**json.loads(tool_call.arguments))}"
    )
    return response
