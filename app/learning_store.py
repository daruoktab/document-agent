"""Local correction snapshots and atomic prompt releases, outside downloadable outputs."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def learning_root() -> Path:
    from .config import _load_local_dotenv

    _load_local_dotenv()
    project_root = Path(__file__).resolve().parents[1]
    directory = Path(
        os.environ.get("LEARNING_DATA_DIR")
        or project_root / "data" / "learning"
    )
    return directory if directory.is_absolute() else project_root / directory


class LearningStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or learning_root()
        self.path = self.root / "learning.sqlite"

    @contextmanager
    def connection(self):
        self.root.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS observations (
                    id INTEGER PRIMARY KEY, image_hash TEXT, path TEXT, data TEXT
                );
                CREATE TABLE IF NOT EXISTS corrections (
                    id INTEGER PRIMARY KEY, document_hash TEXT, page INTEGER,
                    image_hash TEXT, image BLOB, data TEXT, ready INTEGER
                );
                CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, data TEXT);
                CREATE TABLE IF NOT EXISTS releases (
                    id INTEGER PRIMARY KEY, version TEXT, created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS partitions (key TEXT PRIMARY KEY, split TEXT);
            """)
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def observe(
        self, image_path: str, original: str, context: str, version: str
    ) -> None:
        image = Path(image_path)
        data = {"original": original, "context": context, "version": version}
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO observations(image_hash,path,data) VALUES(?,?,?)",
                (
                    digest(image.read_bytes()),
                    str(image.resolve()),
                    json.dumps(data),
                ),
            )

    def observation(self, image_path: Path) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        with self.connection() as conn:
            row = conn.execute(
                "SELECT data FROM observations WHERE image_hash=? AND path=? ORDER BY id DESC LIMIT 1",
                (digest(image_path.read_bytes()), str(image_path.resolve())),
            ).fetchone()
        return json.loads(row["data"]) if row else None

    def save_correction(
        self,
        *,
        document_hash: str,
        page: int,
        image_path: Path,
        original: str,
        corrected: str,
        context: str,
        version: str,
        ready: bool,
        document_name: str,
    ) -> int:
        if not corrected.strip() or page < 1:
            raise ValueError("Isi koreksi dan nomor halaman harus valid.")
        image = image_path.read_bytes()
        data = {
            "original": original,
            "corrected": corrected,
            "context": context,
            "version": version,
            "document_name": document_name,
            "created_at": datetime.now(UTC).isoformat(),
            "suffix": image_path.suffix,
        }
        with self.connection() as conn:
            cursor = conn.execute(
                "INSERT INTO corrections(document_hash,page,image_hash,image,data,ready) VALUES(?,?,?,?,?,?)",
                (
                    document_hash,
                    page,
                    digest(image),
                    image,
                    json.dumps(data),
                    int(ready),
                ),
            )
            return int(cursor.lastrowid or 0)

    def latest_correction(self, document_hash: str, page: int) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        with self.connection() as conn:
            row = conn.execute(
                "SELECT data,ready FROM corrections WHERE document_hash=? AND page=? ORDER BY id DESC LIMIT 1",
                (document_hash, page),
            ).fetchone()
        return {**json.loads(row["data"]), "ready": bool(row["ready"])} if row else None

    def examples(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.connection() as conn:
            # Select latest BEFORE filtering readiness, so a new draft revokes approval.
            rows = conn.execute("""
                SELECT * FROM corrections WHERE id IN (
                    SELECT MAX(id) FROM corrections GROUP BY document_hash,page
                ) AND ready=1 ORDER BY id
            """).fetchall()
        return [{**dict(row), **json.loads(row["data"])} for row in rows]

    def runs(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.connection() as conn:
            return [
                json.loads(row[0])
                for row in conn.execute("SELECT data FROM runs ORDER BY rowid DESC")
            ]

    def dataset_split(
        self, examples: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            assignments = dict(
                conn.execute("SELECT key,split FROM partitions").fetchall()
            )
            result = split_examples(examples, assignments)
            for split, subset in result.items():
                for e in subset:
                    for key in (
                        "doc:" + e["document_hash"],
                        "image:" + e["image_hash"],
                    ):
                        conn.execute(
                            "INSERT OR IGNORE INTO partitions VALUES(?,?)", (key, split)
                        )
        return result

    def save_run(self, data: dict[str, Any]) -> str:
        data.setdefault("id", uuid4().hex)
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO runs VALUES(?,?) ON CONFLICT(id) DO UPDATE SET data=excluded.data",
                (data["id"], json.dumps(data)),
            )
        return data["id"]

    def active(self) -> str:
        if not self.path.exists():
            return "baseline"
        with self.connection() as conn:
            row = conn.execute(
                "SELECT version FROM releases ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return row[0] if row else "baseline"

    def active_run(self) -> dict[str, Any] | None:
        version = self.active()
        if version == "baseline":
            return None
        return next((r for r in self.runs() if r["id"] == version), None)

    def activate(self, run_id: str, model_fingerprint: str) -> None:
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT data FROM runs WHERE id=?", (run_id,)).fetchone()
            if not row:
                raise ValueError("Kandidat tidak ditemukan.")
            run = json.loads(row[0])
            active_row = conn.execute(
                "SELECT version FROM releases ORDER BY id DESC LIMIT 1"
            ).fetchone()
            current = active_row[0] if active_row else "baseline"
            current_ids = [
                r[0]
                for r in conn.execute(
                    "SELECT id FROM corrections WHERE id IN (SELECT MAX(id) FROM corrections GROUP BY document_hash,page) AND ready=1 ORDER BY id"
                )
            ]
            if run.get("status") != "completed" or not run.get("eligible"):
                raise ValueError("Kandidat belum lulus pengujian terpisah.")
            if (
                run["baseline_version"] != current
                or run["model_fingerprint"] != model_fingerprint
            ):
                raise ValueError(
                    "Prompt aktif atau konfigurasi model berubah; jalankan evaluasi baru."
                )
            if current_ids != run["example_ids"]:
                raise ValueError(
                    "Koreksi berubah sejak evaluasi; jalankan evaluasi baru."
                )
            conn.execute(
                "INSERT INTO releases(version,created_at) VALUES(?,?)",
                (run_id, datetime.now(UTC).isoformat()),
            )

    def rollback(self) -> str:
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT version FROM releases ORDER BY id DESC LIMIT 2"
            ).fetchall()
            if not rows:
                raise ValueError("Belum ada versi yang diterapkan.")
            previous = rows[1][0] if len(rows) > 1 else "baseline"
            conn.execute(
                "INSERT INTO releases(version,created_at) VALUES(?,?)",
                (previous, datetime.now(UTC).isoformat()),
            )
        return previous


def split_examples(
    examples: list[dict[str, Any]], assignments: dict[str, str] | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Group by document AND shared page bytes; copied pages never cross splits."""
    parents = {e["document_hash"]: e["document_hash"] for e in examples}

    def root(key: str) -> str:
        while parents[key] != key:
            key = parents[key]
        return key

    seen: dict[str, str] = {}
    for example in examples:
        doc = example["document_hash"]
        other = seen.setdefault(example["image_hash"], doc)
        parents[root(doc)] = root(other)
    groups: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        groups.setdefault(root(example["document_hash"]), []).append(example)
    keys = sorted(groups)
    if len(keys) < 6:
        raise ValueError(
            "Perlu minimal 6 dokumen berbeda dengan koreksi siap evaluasi (salinan dihitung satu kelompok)."
        )
    n_test = max(1, len(keys) // 5)
    partition = {
        "train": keys[: -2 * n_test],
        "validation": keys[-2 * n_test : -n_test],
        "test": keys[-n_test:],
    }
    if not assignments:
        return {
            name: [example for key in subset for example in groups[key]]
            for name, subset in partition.items()
        }
    result: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for key, group in groups.items():
        pinned = {
            assignments[k]
            for e in group
            for k in ("doc:" + e["document_hash"], "image:" + e["image_hash"])
            if k in assignments
        }
        if len(pinned) > 1:
            raise ValueError(
                "Dokumen salinan menghubungkan kelompok dataset berbeda; tinjau koreksi sebelum melanjutkan."
            )
        split = (
            next(iter(pinned))
            if pinned
            else ("train", "train", "train", "validation", "test")[
                int(digest(key.encode()), 16) % 5
            ]
        )
        result[split].extend(group)
    if not all(result.values()):
        raise ValueError(
            "Setiap kelompok dataset tetap harus memiliki contoh siap evaluasi."
        )
    return result
