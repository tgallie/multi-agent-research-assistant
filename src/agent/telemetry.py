"""Per-run structured telemetry with a local append-only sink."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from threading import Lock
from typing import Any

from pydantic import BaseModel


class JsonlTelemetry:
    """Append JSON records atomically within one process."""

    def __init__(self, path: Path) -> None:
        """Configure the telemetry destination."""

        self._path = path
        self._lock = Lock()

    def emit(self, record: BaseModel | dict[str, Any]) -> None:
        """Write one record; logging failure never changes run behavior."""

        payload = record.model_dump(mode="json") if isinstance(record, BaseModel) else record
        try:
            with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, sort_keys=True) + "\n")
        except OSError as exc:
            print(f"telemetry_error={exc}", file=sys.stderr)
