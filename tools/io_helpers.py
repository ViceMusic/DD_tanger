"""Small helpers for saving experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_result_json(
    result: dict[str, Any],
    output_dir: str | Path = "log_json",
    file_name: str | None = None,
) -> Path:
    directory = ensure_dir(output_dir)
    resolved_name = file_name or result.get("run_name", "result")
    output_path = directory / f"{resolved_name}.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return output_path
