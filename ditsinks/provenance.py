"""Immutable run layout and provenance for confirmatory causal experiments."""
from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


RUN_SCHEMA_VERSION = 1
QUESTIONS = ("q1", "q2", "q3", "q5", "q6")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if hasattr(value, "__dict__"):
        return {k: _jsonable(v) for k, v in vars(value).items() if not k.startswith("_")}
    return value if isinstance(value, (str, int, float, bool)) or value is None else repr(value)


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def create_run_layout(root: Path | str, run_id: str, *, config: Any,
                      metadata: Mapping[str, Any]) -> Path:
    """Create a new run directory. Refuses to overwrite, so old evidence is preserved."""
    root = Path(root)
    run = root / "results" / "causal_runs" / str(run_id)
    run.mkdir(parents=True, exist_ok=False)
    (run / "diagnostics").mkdir()
    (run / "figures").mkdir()
    for question in QUESTIONS:
        (run / question).mkdir()
    (run / "config.json").write_text(json.dumps(_jsonable(config), indent=2,
                                                 sort_keys=True) + "\n")
    packages = {}
    for name in ("torch", "diffusers", "transformers", "numpy", "pandas", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    payload = {"schema_version": RUN_SCHEMA_VERSION,
               "created_utc": datetime.now(timezone.utc).isoformat(),
               "git_commit": _git_commit(root), "python": platform.python_version(),
               "packages": packages, **_jsonable(metadata)}
    (run / "provenance.json").write_text(json.dumps(payload, indent=2,
                                                     sort_keys=True) + "\n")
    return run

