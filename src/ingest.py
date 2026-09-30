import lancedb
import uvicorn
from chonkie import LanceDBHandshake
from chonkie.pipeline import Pipeline
from lancedb.pydantic import LanceModel, Vector
from lancedb.rerankers import CrossEncoderReranker
from litestar import Litestar, get
from litestar.openapi import OpenAPIConfig
from pathlib import Path

from fetch import VaultFetcher

EMBEDDING_MODEL = "model2vec://mikeee/m2v-gemma-embedding-300m"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

ROOT = Path(__file__).resolve().parent.parent  # repo root, so paths work from any CWD

db = lancedb.connect(ROOT / ".lancedb")

handshake = LanceDBHandshake(
    connection=db, table_name="knowledge", embedding_model=EMBEDDING_MODEL
)

reranker = CrossEncoderReranker(model_name=RERANKER_MODEL)


class Knowledge(LanceModel):
    text: str
    vector: Vector(handshake.dimension)


@get("/")
async def index() -> str:
    return "Hello, world!"


@get("/ingests")
async def ingests() -> str:
    handshake.table.delete("true")  ## reset knowledge
    (
        Pipeline()
        .fetch_from("vault", dir=str(ROOT / "knowledge_base"), ext=[".md", ".txt"])
        .process_with("text")
        .chunk_with("semantic", threshold=0.8, chunk_size=1024, similarity_window=3)
        .refine_with("overlap", context_size=100)
        .store_in(
            "lancedb",
            connection=db,
            table_name="knowledge",
            embedding_model=EMBEDDING_MODEL,
        )
        .run()
    )
    count = db.open_table(
        "knowledge"
    ).count_rows()  # fresh open: handle snapshots go stale
    print(f"Knowledge table now has {count} chunks")
    return f"Done ({count} chunks)"


@get("/query")
async def query(q: str) -> list[dict[str, str]]:
    # handshake.table = db.open_table("knowledge")
    # results = handshake.search(query=q, limit=5)
    # for result in results:
    #     print(result["score"], result["text"])

    tbl = db.open_table("knowledge")
    results = tbl.search(q, query_type="hybrid").rerank(reranker=reranker).to_pandas()

    return results


@get("/test")
async def test(q: str) -> str:
    model = handshake.embedding_model
    table = db.create_table("test", schema=Knowledge, mode="overwrite")
    table.add(
        [
            {"text": "hello world", "vector": model.embed("hello world").tolist()},
            {"text": "goodbye world", "vector": model.embed("goodbye world").tolist()},
        ]
    )
    actual = table.search(model.embed(q).tolist()).limit(1).to_pydantic(Knowledge)[0]
    return actual.text


app = Litestar(
    [index, ingests, query, test],
    openapi_config=OpenAPIConfig(title="My API", version="1.0.0"),
)


def main():
    # This fires up Uvicorn and hands it your app instance
    uvicorn.run(app, host="127.0.0.1", port=8000)
