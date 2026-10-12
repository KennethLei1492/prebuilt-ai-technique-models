"""``python -m parallelization "task" [--domain NAME] [--offline] [--json] [--input key=value ...]``."""
from __future__ import annotations

from .cli import main

if __name__ == "__main__":  # pragma: no cover - thin entry point
    raise SystemExit(main())
