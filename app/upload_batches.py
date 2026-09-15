"""Persistent upload groups. Document artifacts remain in their existing locations."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


def create_batch(output_dir: Path, name: str, documents: list[dict[str, str]]) -> dict[str, Any]:
    requested_name = name.strip() or "Upload dokumen"
    used_names = {batch.get("name") for batch in list_batches(output_dir)}
    batch_name = requested_name
    ordinal = 1
    while batch_name in used_names:
        batch_name = f"{requested_name} ({ordinal})"
        ordinal += 1
    batch = {
        "id": uuid4().hex,
        "name": batch_name,
        "created_at": datetime.now(UTC).isoformat(),
        "documents": list({doc["stem"]: doc for doc in documents}.values()),
    }
    directory = output_dir / "batches" / batch["id"]
    directory.mkdir(parents=True, exist_ok=False)
    temporary = directory / "manifest.tmp"
    temporary.write_text(json.dumps(batch, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(directory / "manifest.json")
    return batch


def list_batches(output_dir: Path) -> list[dict[str, Any]]:
    batches = []
    for path in (output_dir / "batches").glob("*/manifest.json"):
        try:
            batch = json.loads(path.read_text(encoding="utf-8"))
            if batch["id"] != path.parent.name or not isinstance(batch["documents"], list):
                continue
            if any(
                not doc["stem"] or doc["stem"] in {".", ".."}
                or "/" in doc["stem"] or "\\" in doc["stem"]
                for doc in batch["documents"]
            ):
                continue
            batches.append(batch)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(batches, key=lambda batch: batch["created_at"], reverse=True)
