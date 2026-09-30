import lancedb
import uvicorn
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from litestar import Litestar, get

db = lancedb.connect(".lancedb/")
func = (
    get_registry()
    .get("ollama")
    .create(name="embeddinggemma", host="http://localhost:11434")
)


class Knowledge(LanceModel):
    text: str = func.SourceField()
    vector: Vector(func.ndims()) = func.VectorField()


table = db.create_table("knowledge", schema=Knowledge, exist_ok=True)


@get("/")
async def index() -> str:
    return "Hello, world!"


@get("/ingests")
async def ingests(texts: list[Knowledge]) -> str:
    table.add(texts)
    return "Done"


@get("/query")
async def query(query: str) -> str:
    actual = table.search(query).limit(1).to_pydantic(Knowledge)[0]
    return actual.text


@get("/test")
async def test(query: str) -> str:
    table = db.create_table("test", schema=Knowledge, mode="overwrite")
    table.add(
        [
            {"text": "hello world"},
            {"text": "goodbye world"},
        ]
    )
    actual = table.search(query).limit(1).to_pydantic(Knowledge)[0]
    return actual.text


app = Litestar([index, ingests, query, test])


def main():
    # This fires up Uvicorn and hands it your app instance
    uvicorn.run(app, host="127.0.0.1", port=8000)
