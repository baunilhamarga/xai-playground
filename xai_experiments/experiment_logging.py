"""Local JSON experiment logging."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import uuid


SCHEMA_VERSION = "xai-tip-explanation-run/v1"


@dataclass
class JSONExperimentLogger:
    log_dir: Path = Path("logs")

    def log(self, record: dict[str, object]) -> Path:
        experiment_id = str(record.get("experiment_id") or uuid.uuid4())
        timestamp = str(
            record.get("timestamp_utc")
            or datetime.now(timezone.utc).isoformat(timespec="microseconds")
        )
        record["schema_version"] = SCHEMA_VERSION
        record["experiment_id"] = experiment_id
        record["timestamp_utc"] = timestamp

        day = timestamp[:10]
        mode_slug = _slugify(str(record.get("mode", "experiment")))
        filename_timestamp = (
            timestamp.replace("+00:00", "Z").replace(":", "").replace(".", "")
        )
        path = self.log_dir / day / f"{filename_timestamp}_{mode_slug}_{experiment_id[:8]}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        return path


def output_metrics(text: str) -> dict[str, int]:
    return {
        "char_count": len(text),
        "word_count": len(text.split()),
        "line_count": len(text.splitlines()),
    }


def _slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "experiment"
