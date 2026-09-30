import uvicorn
from litestar import Litestar, get
from ingest import ingests, query, test
from inference import infer
from litestar.openapi import OpenAPIConfig


@get("/")
async def index() -> str:
    return "Hello, world!"


app = Litestar(
    [index, ingests, query, test, infer],
    openapi_config=OpenAPIConfig(title="My API", version="1.0.0"),
)


def main():
    # This fires up Uvicorn and hands it your app instance
    uvicorn.run(app, host="127.0.0.1", port=8000)
