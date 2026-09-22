"""Persistent upload groups. Document artifacts remain in their existing locations."""
from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


def _safe_document_stem(stem: str) -> bool:
    return bool(stem) and stem not in {".", ".."} and "/" not in stem and "\\" not in stem


def _batch_directory(output_dir: Path, batch_id: str) -> Path:
    if not batch_id or "/" in batch_id or "\\" in batch_id or batch_id in {".", ".."}:
        raise ValueError("Identitas batch tidak valid")
    directory = (output_dir / "batches" / batch_id).resolve()
    if directory.parent != (output_dir / "batches").resolve():
        raise ValueError("Lokasi batch tidak valid")
    return directory


def _write_manifest(batch_directory: Path, batch: dict[str, Any]) -> None:
    temporary = batch_directory / "manifest.tmp"
    temporary.write_text(json.dumps(batch, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(batch_directory / "manifest.json")


def create_batch(output_dir: Path, name: str, documents: list[dict[str, str]]) -> dict[str, Any]:
    requested_name = name.strip() or "Uploaded files"
    used_names = {batch.get("name") for batch in list_batches(output_dir)}
    batch_name = requested_name
    ordinal = 1
    while batch_name in used_names:
        batch_name = f"{requested_name} ({ordinal})"
        ordinal += 1
    batch = {
        "id": uuid4().hex,
        "name": batch_name,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "uploaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "documents": list({doc["stem"]: doc for doc in documents}.values()),
    }
    directory = output_dir / "batches" / batch["id"]
    directory.mkdir(parents=True, exist_ok=False)
    _write_manifest(directory, batch)
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


def _other_batch_stems(output_dir: Path, excluded_batch_id: str) -> set[str]:
    stems: set[str] = set()
    for batch in list_batches(output_dir):
        if batch.get("id") == excluded_batch_id:
            continue
        stems.update(
            doc.get("stem", "")
            for doc in batch.get("documents", [])
            if _safe_document_stem(doc.get("stem", ""))
        )
    return stems


def _remove_unreferenced_document_outputs(
    output_dir: Path, stems: set[str], *, excluded_batch_id: str
) -> list[str]:
    protected = _other_batch_stems(output_dir, excluded_batch_id)
    output_root = output_dir.resolve()
    removed: list[str] = []
    for stem in stems - protected:
        if not _safe_document_stem(stem):
            continue
        document_directory = (output_dir / stem).resolve()
        if document_directory.parent != output_root:
            continue
        if document_directory.is_dir():
            shutil.rmtree(document_directory)
            removed.append(stem)
    return removed


def delete_batch(output_dir: Path, batch_id: str) -> dict[str, Any]:
    """Delete a batch manifest/ZIP and unreferenced document result directories."""
    batch = next((item for item in list_batches(output_dir) if item.get("id") == batch_id), None)
    if batch is None:
        raise FileNotFoundError(f"Batch '{batch_id}' tidak ditemukan")
    stems = {
        doc.get("stem", "")
        for doc in batch.get("documents", [])
        if _safe_document_stem(doc.get("stem", ""))
    }
    directory = _batch_directory(output_dir, batch_id)
    if directory.exists():
        shutil.rmtree(directory)
    removed_stems = _remove_unreferenced_document_outputs(
        output_dir, stems, excluded_batch_id=batch_id
    )
    return {"batch": batch, "removed_stems": removed_stems}


def delete_document_from_batch(
    output_dir: Path, batch_id: str, stem: str
) -> dict[str, Any]:
    """Remove one document from a batch and delete its unreferenced output."""
    if not _safe_document_stem(stem):
        raise ValueError("Identitas dokumen tidak valid")
    batch = next((item for item in list_batches(output_dir) if item.get("id") == batch_id), None)
    if batch is None:
        raise FileNotFoundError(f"Batch '{batch_id}' tidak ditemukan")
    documents = batch.get("documents", [])
    remaining = [doc for doc in documents if doc.get("stem") != stem]
    if len(remaining) == len(documents):
        raise FileNotFoundError(f"Dokumen '{stem}' tidak ditemukan dalam batch")
    directory = _batch_directory(output_dir, batch_id)
    if not remaining:
        if directory.exists():
            shutil.rmtree(directory)
        removed_stems = _remove_unreferenced_document_outputs(
            output_dir, {stem}, excluded_batch_id=batch_id
        )
        return {"batch": batch, "deleted_batch": True, "removed_stems": removed_stems}
    updated = dict(batch)
    updated["documents"] = remaining
    (directory / "hasil.zip").unlink(missing_ok=True)
    _write_manifest(directory, updated)
    removed_stems = _remove_unreferenced_document_outputs(
        output_dir, {stem}, excluded_batch_id=batch_id
    )
    return {"batch": updated, "deleted_batch": False, "removed_stems": removed_stems}
