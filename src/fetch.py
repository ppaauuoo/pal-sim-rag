import os
import subprocess
from chonkie.fetcher import BaseFetcher
from chonkie.pipeline import fetcher
from pathlib import Path
from pathspec import GitIgnoreSpec


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
                    target = self._resolve_lnk(path)
                    if target is None:
                        continue
                    if target.is_dir():
                        self._walk(target, ext, spec, files, visited)
                    elif ext is None or target.suffix in ext:
                        files.append(target)
                elif ext is None or os.path.splitext(filename)[1] in ext:
                    files.append(path)

    @staticmethod
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
