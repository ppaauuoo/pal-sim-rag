import os
import subprocess
import lancedb
import uvicorn
from chonkie import LanceDBHandshake
from chonkie.fetcher import BaseFetcher
from chonkie.pipeline import Pipeline, fetcher
from lancedb.pydantic import LanceModel, Vector
from lancedb.rerankers import CrossEncoderReranker
from litestar import Litestar, get
from litestar.openapi import OpenAPIConfig
from pathlib import Path
from pathspec import GitIgnoreSpec

EMBEDDING_MODEL = "model2vec://mikeee/m2v-gemma-embedding-300m"

ROOT = Path(__file__).resolve().parent.parent  # repo root, so paths work from any CWD

db = lancedb.connect(ROOT / ".lancedb")


@fetcher("vault")
class VaultFetcher(BaseFetcher):
    """Deep-scans a directory tree recursively, pruning hidden dirs (.obsidian, .trash, .git).

    Honors gitignore-style rules from .ragignore inside the scanned directory,
    matched relative to it.
    """

    def fetch(
        self,
        dir: str | os.PathLike,
        ext: list[str] | None = None,
    ) -> list[Path]:
        dir_path = Path(dir)
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {dir}")

        ignore_file = dir_path / ".ragignore"
        if ignore_file.is_file():
            spec = GitIgnoreSpec.from_lines(
                ignore_file.read_text(encoding="utf-8").splitlines()
            )
        else:
            spec = None

        files: list[Path] = []
        self._walk(dir_path, ext, spec, files, visited=set())
        return files

    def _walk(
        self,
        base: Path,
        ext: list[str] | None,
        spec: GitIgnoreSpec | None,
        files: list[Path],
        visited: set[Path],
    ) -> None:
        """Walk base recursively; .lnk shortcuts to dirs/files are followed too."""
        real = base.resolve()
        if real in visited:  # guard against shortcut loops
            return
        visited.add(real)

        for root, dirs, filenames in os.walk(base, followlinks=False):
            rel_root = Path(root).relative_to(base)
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".")
                and not (spec and spec.match_file(f"{rel_root / d}/"))
            ]
            for filename in filenames:
                if filename.startswith("."):
                    continue
                rel = rel_root / filename
                if spec and spec.match_file(str(rel)):
                    continue
                path = Path(root) / filename
                if filename.lower().endswith(".lnk"):
                    target = _resolve_lnk(path)
                    if target is None:
                        continue
                    if target.is_dir():
                        self._walk(target, ext, spec, files, visited)
                    elif ext is None or target.suffix in ext:
                        files.append(target)
                elif ext is None or os.path.splitext(filename)[1] in ext:
                    files.append(path)


def _resolve_lnk(path: Path) -> Path | None:
    """Resolve a Windows .lnk shortcut to its target path, or None."""
    try:
        # escape single quotes for the PowerShell string literal
        ps_path = str(path).replace("'", "''")
        out = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"(New-Object -ComObject WScript.Shell).CreateShortcut('{ps_path}').TargetPath",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        target = out.stdout.strip()
        return Path(target) if target and Path(target).exists() else None
    except Exception:
        return None


def _reset_knowledge() -> None:
    """Clear the knowledge table so each ingest is a clean overwrite."""
    # ponytail: delete rows instead of drop/recreate — drop races table handles
    handshake.table.delete("true")


handshake = LanceDBHandshake(
    connection=db, table_name="knowledge", embedding_model=EMBEDDING_MODEL
)

reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-v2-m3")


class Knowledge(LanceModel):
    text: str
    vector: Vector(handshake.dimension)


@get("/")
async def index() -> str:
    return "Hello, world!"


@get("/ingests")
async def ingests() -> str:
    _reset_knowledge()
    # ponytail: Pipeline.run() returns None after a write step; count from the table
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
    handshake.table = db.open_table("knowledge")  # fresh snapshot; handles go stale
    results = handshake.search(query=q, limit=5)
    for result in results:
        print(result["score"], result["text"])

    results = (
        table.search("your search query", query_type="hybrid")
        .rerank(reranker=reranker)
        .to_pandas()
    )

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
