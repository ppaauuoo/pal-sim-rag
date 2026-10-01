import asyncio
import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # models are cached; proxy stalls 90s/reload otherwise

import lancedb
from chonkie import LanceDBHandshake
from chonkie.pipeline import Pipeline
from lancedb.pydantic import LanceModel, Vector
from lancedb.rerankers import CrossEncoderReranker
from litestar import get
from pathlib import Path

from fetch import VaultFetcher  # noqa: F401 — registers the "vault" fetcher alias

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


@get("/ingests")
async def ingests() -> str:
    handshake.table.delete("true")  ## reset knowledge
    (
        Pipeline()
        .fetch_from("vault", dir=str(ROOT / "knowledge_base"), ext=[".md", ".txt"])
        .process_with("text")
        .chunk_with("recursive", chunk_size=512, min_characters_per_chunk=24)
        # ponytail: semantic chunking fragmented bullet-list notes into 10-token
        # heading-only chunks; recursive keeps sections intact. Revisit if notes
        # become long-form prose.
        .store_in(
            "lancedb",
            connection=db,
            table_name="knowledge",
            embedding_model=EMBEDDING_MODEL,
        )
        .run()
    )
    tbl = db.open_table("knowledge")  # fresh open: handle snapshots go stale
    tbl.create_fts_index("text", replace=True, with_position=True)  # required for hybrid /query; positions needed for phrase queries
    count = tbl.count_rows()
    print(f"Knowledge table now has {count} chunks")
    return f"Done ({count} chunks)"


def search(q: str) -> list[dict]:
    """Plain hybrid search — used by /query and as the LLM tool in inference.py."""
    tbl = db.open_table("knowledge")
    if "text_idx" not in {
        i.name for i in tbl.list_indices()
    }:  # lazy FTS for pre-index tables
        tbl.create_fts_index("text", with_position=True)
    vector = handshake.embedding_model.embed(q).tolist()
    results = (
        tbl.search(query_type="hybrid")
        .text(q)
        .vector(vector)
        .rerank(reranker=reranker)
        .limit(20)
        .to_list()
    )
    for row in results:  # unpack chunk_metadata JSON so filename etc. are top-level
        meta = row.pop("chunk_metadata", None)
        if meta:
            try:
                row.update(json.loads(meta))
            except json.JSONDecodeError:
                pass
    # ponytail: cross-encoder scores are bimodal (relevant ≥0.5, junk <0.02);
    # floor at 0.3 so unrelated chunks never reach the LLM. Raise if junk leaks.
    return [r for r in results if r.get("_relevance_score", 0) >= 0.3]


@get("/query")
async def query(q: str) -> list[dict]:
    return await asyncio.to_thread(search, q)


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
