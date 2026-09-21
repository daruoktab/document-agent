"""
Job Tracker & Background Process Manager untuk Streamlit Document Extraction.

Fitur:
  - Eksekusi background thread & subprocess independen dari lifecycle tab browser.
  - Logging real-time langsung di-flush baris demi baris ke file:
      * output/logs/{stem}_latest.log (selalu menunjuk ke run terakhir)
      * output/logs/{stem}_{timestamp}.log (arsip riwayat)
  - Tracking status & progres granular ke file:
      * output/logs/{stem}_status.json (metadata terstruktur)
      * output/logs/{stem}_progress.txt (ringkasan 6-baris ramah dibaca manusia)
  - Deteksi otomatis nomor halaman saat ini vs total halaman dari output stream.
  - Tahan terhadap minimize window, tab sleep, atau refresh UI Streamlit.
"""

from __future__ import annotations

import collections
import datetime as dt
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SLOT_FILE_NAME = ".extraction_slot.json"
_SLOT_POLL_SECONDS = 1.0


def get_max_concurrent_extractions() -> int:
    """Ambil batas proses ekstraksi bersamaan; satu proses adalah default aman untuk VLM."""
    raw_value = os.environ.get("MAX_CONCURRENT_EXTRACTIONS", "1").strip()
    try:
        return max(1, int(raw_value))
    except ValueError:
        logger.warning(
            "MAX_CONCURRENT_EXTRACTIONS=%r tidak valid; menggunakan nilai 1.",
            raw_value,
        )
        return 1


def is_pid_alive(pid: int | None) -> bool:
    """Cek apakah proses dengan PID tertentu masih aktif di sistem operasi."""
    if pid is None or pid <= 0:
        return False
    try:
        import psutil

        return psutil.pid_exists(pid)
    except Exception:  # noqa: BLE001, S110
        pass

    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except (ProcessLookupError, OSError):
        return False
    else:
        return True


def get_python_executable() -> str:
    """Temukan interpreter python aktif saat ini (mis. Conda/Venv) atau fallback ke .venv proyek."""
    if sys.executable and Path(sys.executable).exists():
        return sys.executable
    venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    venv_python_unix = PROJECT_ROOT / ".venv" / "bin" / "python"
    if venv_python_unix.exists():
        return str(venv_python_unix)
    return sys.executable


@dataclass
class JobInfo:
    job_id: str
    file_name: str
    input_path: Path
    output_dir: Path
    out_file: Path
    db_file: Path | None
    log_path: Path
    latest_log_path: Path
    status_file: Path
    progress_file: Path
    status: str = "queued"  # "queued" | "running" | "paused" | "completed" | "failed" | "canceled"
    queue_position: int = 0
    current_page: int = 0
    total_pages: int = 0
    stage: str = "Memulai ekstraksi..."
    last_message: str = ""
    started_at: str = field(
        default_factory=lambda: dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    )
    updated_at: str = field(
        default_factory=lambda: dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    )
    completed_at: str | None = None
    pid: int | None = None
    returncode: int | None = None
    error_message: str | None = None
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    extraction_options: dict[str, Any] = field(default_factory=dict)
    recent_logs: collections.deque[str] = field(
        default_factory=lambda: collections.deque(maxlen=200)
    )

    def progress_percentage(self) -> float:
        """Hitung persentase progres 0.0 - 100.0."""
        if self.status == "completed":
            return 100.0
        if self.total_pages > 0 and self.current_page > 0:
            pct = (self.current_page / self.total_pages) * 100.0
            return min(98.0, max(1.0, round(pct, 1)))
        if self.status == "running":
            return 5.0
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "file_name": self.file_name,
            "input_path": str(self.input_path),
            "output_dir": str(self.output_dir),
            "out_file": str(self.out_file),
            "db_file": str(self.db_file) if self.db_file else None,
            "log_path": str(self.log_path),
            "latest_log_path": str(self.latest_log_path),
            "status_file": str(self.status_file),
            "progress_file": str(self.progress_file),
            "status": self.status,
            "queue_position": self.queue_position,
            "current_page": self.current_page,
            "total_pages": self.total_pages,
            "progress_pct": self.progress_percentage(),
            "stage": self.stage,
            "last_message": self.last_message,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "pid": self.pid,
            "returncode": self.returncode,
            "error_message": self.error_message,
            "run_id": self.run_id,
            "extraction_options": self.extraction_options,
        }

    def save_status(self) -> None:
        """Simpan status JSON dan ringkasan TXT ke disk secara atomic/aman."""
        try:
            self.status_file.parent.mkdir(parents=True, exist_ok=True)
            self.updated_at = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")

            # 1. Simpan JSON terstruktur
            self.status_file.write_text(
                json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            # 2. Simpan TXT ringkas yang mudah dimonitor via Notepad / terminal
            pct_str = f"{self.progress_percentage():.1f}%"
            page_str = (
                f"{self.current_page} / {self.total_pages}"
                if self.total_pages > 0
                else f"{self.current_page} (menghitung total...)"
            )
            txt_content = (
                "==================================================\n"
                "STATUS EKSTRAKSI DOKUMEN (REAL-TIME MONITOR)\n"
                "==================================================\n"
                f"File Dokumen   : {self.file_name}\n"
                f"Status         : {self.status.upper()} (PID: {self.pid or '-'})\n"
                f"Progres        : Halaman {page_str} ({pct_str})\n"
                f"Tahapan Saat Ini: {self.stage}\n"
                f"Pesan Terakhir : {self.last_message}\n"
                f"Waktu Mulai    : {self.started_at}\n"
                f"Update Terakhir: {self.updated_at}\n"
                f"File Log Utama : {self.latest_log_path}\n"
                f"File Markdown  : {self.out_file}\n"
                f"Database SQLite: {self.db_file or '-'}\n"
                "==================================================\n"
            )
            self.progress_file.write_text(txt_content, encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            logger.warning("Gagal menyimpan file status job: %s", e)


class JobManager:
    """Manajer singleton in-process untuk menjalankan dan melacak job ekstraksi."""

    _instance: ClassVar[JobManager | None] = None
    _jobs: ClassVar[dict[str, JobInfo]] = {}
    _processes: ClassVar[dict[str, subprocess.Popen]] = {}
    _threads: ClassVar[dict[str, threading.Thread]] = {}
    _execution_slots: ClassVar[dict[str, Path]] = {}
    # RLock diperlukan karena start_job/list_all_documents memanggil get_job
    # saat lock sudah dipegang oleh thread yang sama.
    _lock: ClassVar[threading.RLock] = threading.RLock()

    @classmethod
    def get_instance(cls) -> JobManager:
        if cls._instance is None:
            cls._instance = JobManager()
        return cls._instance

    def start_job(
        self,
        input_path: Path,
        output_dir: Path,
        *,
        doc_type: str | None = None,
        dpi: int = 200,
        force_all_tables: bool = False,
        preview_chunks: bool = False,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        resume: bool = False,
        queue_position: int = 0,
    ) -> JobInfo:
        """Antrekan ekstraksi baru jika dokumen yang sama belum diproses."""
        with self._lock:
            stem = input_path.stem
            existing_job = self.get_job(stem, output_dir=output_dir)
            if existing_job and existing_job.status in {"queued", "running"}:
                # Sudah diantrekan atau berjalan, kembalikan job yang aktif.
                return existing_job

            doc_output_dir = output_dir / stem
            doc_output_dir.mkdir(parents=True, exist_ok=True)
            log_dir = doc_output_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)

            timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
            log_path = log_dir / f"{stem}_{timestamp}.log"
            latest_log_path = log_dir / f"{stem}_latest.log"
            status_file = log_dir / f"{stem}_status.json"
            progress_file = log_dir / f"{stem}_progress.txt"
            out_file = doc_output_dir / f"{stem}.md"
            checkpoint_file = log_dir / f"{stem}_checkpoint.json"
            db_dir = doc_output_dir / "databases"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_file = db_dir / f"{stem}.sqlite"
            effective_resume = resume or checkpoint_file.exists()

            # Tulis header awal ke file log agar langsung tersedia
            start_iso = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
            header = (
                "================================================================================\n"
                f"LOG EKSTRAKSI DOKUMEN: {input_path.name}\n"
                f"Waktu Mulai: {start_iso}\n"
                f"Target Markdown: {out_file}\n"
                f"Target SQLite: {db_file}\n"
                "================================================================================\n\n"
            )
            log_path.write_text(header, encoding="utf-8")
            latest_log_path.write_text(header, encoding="utf-8")

            job = JobInfo(
                job_id=stem,
                file_name=input_path.name,
                input_path=input_path,
                output_dir=output_dir,
                out_file=out_file,
                db_file=db_file,
                log_path=log_path,
                latest_log_path=latest_log_path,
                status_file=status_file,
                progress_file=progress_file,
                status="queued",
                queue_position=queue_position,
                stage="Menunggu slot ekstraksi VLM...",
                last_message="Job masuk antrean ekstraksi.",
                started_at=start_iso,
                extraction_options={
                    "doc_type": doc_type,
                    "dpi": dpi,
                    "force_all_tables": force_all_tables,
                    "preview_chunks": preview_chunks,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "resume": effective_resume,
                },
            )
            job.save_status()
            self._jobs[stem] = job

            self._launch_job(job, resume=effective_resume)
            return job

    @staticmethod
    def _checkpoint_path(job: JobInfo) -> Path:
        return job.out_file.parent / "logs" / f"{job.out_file.stem}_checkpoint.json"

    def _launch_job(self, job: JobInfo, *, resume: bool = False) -> None:
        """Jalankan worker untuk job baru atau job yang dipulihkan dari checkpoint."""
        cmd = [
            get_python_executable(),
            str(PROJECT_ROOT / "main.py"),
            str(job.input_path),
            "-o",
            str(job.out_file),
            "--dpi",
            str(job.extraction_options.get("dpi", 200)),
            "--log-file",
            str(job.log_path),
        ]
        doc_type = job.extraction_options.get("doc_type")
        if doc_type:
            cmd.extend(["-t", str(doc_type)])
        if job.extraction_options.get("force_all_tables"):
            cmd.append("--force-all-tables")
        if job.extraction_options.get("preview_chunks"):
            cmd.extend(
                [
                    "--preview-chunks",
                    "--chunk-size",
                    str(job.extraction_options.get("chunk_size", 1000)),
                    "--chunk-overlap",
                    str(job.extraction_options.get("chunk_overlap", 150)),
                ]
            )
        if resume:
            cmd.append("--resume")

        worker = threading.Thread(
            target=self._run_worker,
            args=(job, cmd),
            daemon=True,
            name=f"Worker-{job.job_id}",
        )
        self._threads[job.job_id] = worker
        worker.start()

    def resume_pending_jobs(self, output_dir: Path) -> int:
        """Mulai ulang job terhenti yang memiliki checkpoint halaman."""
        resumed = 0
        for document in self.list_all_documents(output_dir):
            if document["status"] != "queued":
                continue
            job = self.get_job(document["stem"], output_dir=output_dir)
            if job is None or not self._checkpoint_path(job).exists():
                continue
            with self._lock:
                worker = self._threads.get(job.job_id)
                if worker is not None and worker.is_alive():
                    continue
                self._launch_job(job, resume=True)
            resumed += 1
        return resumed

    @staticmethod
    def _slot_paths(output_dir: Path) -> list[Path]:
        slot_count = get_max_concurrent_extractions()
        if slot_count == 1:
            return [output_dir / _SLOT_FILE_NAME]
        return [output_dir / _SLOT_FILE_NAME] + [
            output_dir / f".extraction_slot_{slot_number}.json"
            for slot_number in range(2, slot_count + 1)
        ]

    @staticmethod
    def _read_slot_owner(slot_path: Path) -> dict[str, Any] | None:
        try:
            payload = json.loads(slot_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _write_slot_owner(
        slot_path: Path, job: JobInfo, *, subprocess_pid: int | None
    ) -> None:
        payload = {
            "run_id": job.run_id,
            "job_id": job.job_id,
            "launcher_pid": os.getpid(),
            "subprocess_pid": subprocess_pid,
            "updated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        }
        slot_path.write_text(json.dumps(payload), encoding="utf-8")

    def _acquire_execution_slot(self, job: JobInfo) -> Path | None:
        """Dapatkan slot lintas thread/proses lewat file lock yang otomatis dapat dipulihkan."""
        slot_paths = self._slot_paths(job.output_dir)
        job.output_dir.mkdir(parents=True, exist_ok=True)
        last_wait_notice = 0.0

        while True:
            with self._lock:
                if job.status in {"canceled", "paused"}:
                    return None
                earlier_active = sum(
                    other.status in {"queued", "running"}
                    and other.queue_position < job.queue_position
                    and other.output_dir.resolve() == job.output_dir.resolve()
                    for other in self._jobs.values()
                )
                should_wait_for_order = (
                    earlier_active >= get_max_concurrent_extractions()
                )
            if should_wait_for_order:
                time.sleep(_SLOT_POLL_SECONDS)
                continue

            for slot_path in slot_paths:
                try:
                    descriptor = os.open(
                        slot_path,
                        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    )
                except FileExistsError:
                    owner = self._read_slot_owner(slot_path) or {}
                    owner_pid = owner.get("subprocess_pid") or owner.get("launcher_pid")
                    if isinstance(owner_pid, int) and not is_pid_alive(owner_pid):
                        try:
                            slot_path.unlink()
                            logger.warning(
                                "Menghapus slot ekstraksi usang dari job %s (PID %s).",
                                owner.get("job_id", "tidak diketahui"),
                                owner_pid,
                            )
                            break
                        except FileNotFoundError:
                            break
                        except OSError as exc:
                            logger.warning(
                                "Gagal memulihkan slot ekstraksi usang: %s", exc
                            )
                    continue

                try:
                    payload = {
                        "run_id": job.run_id,
                        "job_id": job.job_id,
                        "launcher_pid": os.getpid(),
                        "subprocess_pid": None,
                        "updated_at": dt.datetime.now(dt.UTC).isoformat(
                            timespec="seconds"
                        ),
                    }
                    os.write(descriptor, json.dumps(payload).encode("utf-8"))
                except OSError:
                    slot_path.unlink(missing_ok=True)
                    raise
                finally:
                    os.close(descriptor)
                return slot_path

            now = time.monotonic()
            if now - last_wait_notice >= 5:
                with self._lock:
                    job.stage = "Menunggu slot ekstraksi VLM..."
                    job.last_message = "Menunggu dokumen lain menyelesaikan ekstraksi."
                    job.save_status()
                last_wait_notice = now
            time.sleep(_SLOT_POLL_SECONDS)

    @staticmethod
    def _release_execution_slot(slot_path: Path | None, run_id: str) -> None:
        if slot_path is None:
            return
        owner = JobManager._read_slot_owner(slot_path)
        if owner and owner.get("run_id") != run_id:
            return
        try:
            slot_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Gagal melepaskan slot ekstraksi: %s", exc)

    def _run_worker(self, job: JobInfo, cmd: list[str]) -> None:
        """Worker background yang membaca stdout proses dan mengalirkan log ke disk."""
        start_time = time.time()
        slot_path: Path | None = None
        try:
            slot_path = self._acquire_execution_slot(job)
            if slot_path is None:
                return

            proc = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            with self._lock:
                self._processes[job.job_id] = proc
                self._execution_slots[job.job_id] = slot_path
                job.pid = proc.pid
                job.status = "running"
                job.stage = "Proses CLI aktif, menunggu parsing halaman..."
                job.save_status()
            self._write_slot_owner(slot_path, job, subprocess_pid=proc.pid)

            # Buka kedua file log dalam mode append dengan autoflush
            with (
                open(job.log_path, "a", encoding="utf-8", buffering=1) as f_hist,
                open(
                    job.latest_log_path, "a", encoding="utf-8", buffering=1
                ) as f_latest,
            ):
                if proc.stdout is not None:
                    for raw_line in proc.stdout:
                        f_hist.write(raw_line)
                        f_latest.write(raw_line)
                        f_hist.flush()
                        f_latest.flush()

                        line = raw_line.strip()
                        if not line:
                            continue

                        job.recent_logs.append(line)
                        self._parse_line_progress(job, line)

            returncode = proc.wait()
            elapsed = time.time() - start_time
            mins, secs = divmod(int(elapsed), 60)
            duration_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"

            footer = (
                "\n================================================================================\n"
                f"STATUS: {'SELESAI SUKSES' if returncode == 0 else f'GAGAL (Exit Code: {returncode})'}\n"
                f"Waktu Selesai : {dt.datetime.now(dt.UTC).isoformat(timespec='seconds')}\n"
                f"Total Durasi  : {duration_str}\n"
                "================================================================================\n"
            )
            with open(job.log_path, "a", encoding="utf-8") as f:
                f.write(footer)
            with open(job.latest_log_path, "a", encoding="utf-8") as f:
                f.write(footer)

            with self._lock:
                job.returncode = returncode
                if returncode == 0:
                    job.status = "completed"
                    job.stage = "Selesai"
                    job.last_message = f"Ekstraksi sukses dalam {duration_str}!"
                elif job.status != "canceled":
                    job.status = "failed"
                    job.stage = "Gagal"
                    job.error_message = (
                        f"Proses berhenti dengan kode error {returncode}."
                    )
                    job.last_message = job.error_message

                if job.status in {"completed", "failed", "canceled"}:
                    job.completed_at = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")

                job.save_status()

        except Exception as exc:
            logger.exception("Kesalahan pada worker background")
            with self._lock:
                job.status = "failed"
                job.stage = "Error Sistem"
                job.error_message = str(exc)
                job.last_message = f"Exception: {exc}"
                job.save_status()
        finally:
            with self._lock:
                held_slot = self._execution_slots.pop(job.job_id, slot_path)
            self._release_execution_slot(held_slot, job.run_id)

    @staticmethod
    def _recover_terminated_job(job: JobInfo) -> None:
        """Pulihkan status akhir saat Streamlit kehilangan pengawas subprocess."""
        if job.out_file.is_file() and job.out_file.stat().st_size > 0:
            job.status = "completed"
            job.stage = "Selesai (dipulihkan dari hasil di disk)"
            job.last_message = "Hasil Markdown ditemukan setelah pengawas dimuat ulang."
            job.error_message = None
        else:
            job.status = "failed"
            job.stage = "Subprocess berhenti tanpa status akhir"
            job.error_message = (
                "Subprocess tidak aktif sebelum pengawas menerima status akhirnya. "
                "Periksa log lengkap dan log layanan Streamlit untuk restart atau kehabisan sumber daya."
            )
            job.last_message = job.error_message
        job.save_status()

    @staticmethod
    def _recover_abandoned_queue(job: JobInfo) -> None:
        """Tandai antrean yang tertinggal setelah proses Streamlit dimuat ulang."""
        job.status = "failed"
        job.stage = "Antrean terhenti saat layanan dimuat ulang"
        job.error_message = (
            "Job masih menunggu antrean ketika layanan Streamlit berhenti atau dimuat ulang. "
            "Silakan jalankan ulang ekstraksi."
        )
        job.last_message = job.error_message
        job.save_status()

    def _parse_line_progress(self, job: JobInfo, line: str) -> None:
        """Deteksi pola progres (halaman X / Y, slide, SQL, guardrail) dari baris log."""
        updated = False

        # 1. Pola Halaman PDF: "Memproses Halaman 3 / 10 dari 'dokumen.pdf'..."
        m_page = re.search(
            r"Memproses Halaman\s+(\d+)\s*/\s*(\d+)", line, re.IGNORECASE
        )
        if m_page:
            job.current_page = int(m_page.group(1))
            job.total_pages = int(m_page.group(2))
            job.stage = f"Ekstraksi VLM Halaman {job.current_page} / {job.total_pages}"
            job.last_message = line
            updated = True

        # 2. Pola Slide PPT: "[Vision PPT] [Slide 2/8] Memproses slide..."
        m_slide = re.search(r"\[Slide\s+(\d+)\s*/\s*(\d+)\]", line, re.IGNORECASE)
        if m_slide:
            job.current_page = int(m_slide.group(1))
            job.total_pages = int(m_slide.group(2))
            job.stage = f"Ekstraksi VLM Slide {job.current_page} / {job.total_pages}"
            job.last_message = line
            updated = True

        # 3. Pola Total Slide selesai dirender: "Selesai render 8 slide gambar"
        m_slides_total = re.search(
            r"Selesai render\s+(\d+)\s+slide gambar", line, re.IGNORECASE
        )
        if m_slides_total:
            job.total_pages = int(m_slides_total.group(1))
            job.stage = f"Rendering selesai ({job.total_pages} slide), memulai VLM..."
            job.last_message = line
            updated = True

        # 4. Tahap Preprocessing & Rendering
        if "Mengekstrak PDF multi-halaman" in line:
            job.stage = "Inisialisasi PDF & Rendering Halaman"
            job.last_message = line
            updated = True
        elif "Memulai rendering slide menjadi gambar" in line:
            job.stage = "Rendering Slide Presentasi (LibreOffice)"
            job.last_message = line
            updated = True

        # 5. Tahap Sub-Agent SQL Tabular
        elif "Sub-Agent SQL" in line:
            job.stage = f"Sub-Agent SQL Ingesti (Halaman {job.current_page or 1})"
            job.last_message = line
            updated = True

        # 6. Tahap Guardrail Cross-Verification
        elif "Guardrail Cross-Verification" in line:
            job.stage = "Audit Guardrail Supervisor (Markdown vs SQLite)"
            job.last_message = line
            updated = True

        # 8. Markdown Berhasil Disimpan
        elif "Hasil Markdown berhasil disimpan ke:" in line:
            job.stage = "Penyimpanan Markdown Selesai"
            job.last_message = line
            updated = True

        if updated:
            job.save_status()

    def get_job(self, stem: str, output_dir: Path | None = None) -> JobInfo | None:
        """Ambil info job dari memori atau rekonstruksi dari status.json di disk."""
        with self._lock:
            # 1. Cek dari memori
            if stem in self._jobs:
                job = self._jobs[stem]
                # Verifikasi jika status masih running, apakah proses OS benar-benar aktif
                if job.status == "running":
                    proc = self._processes.get(stem)
                    if proc is not None and proc.poll() is not None:
                        # Subprocess sudah selesai
                        returncode = proc.poll()
                        job.returncode = returncode
                        job.status = "completed" if returncode == 0 else "failed"
                        job.stage = "Selesai" if returncode == 0 else "Gagal"
                        if returncode != 0:
                            job.error_message = (
                                f"Proses berhenti dengan kode error {returncode}."
                            )
                        job.save_status()
                    # Worker thread may still be starting the subprocess.
                    # Do not mark a job failed before it receives a PID.
                    elif job.pid is not None and not is_pid_alive(job.pid):
                        if self._checkpoint_path(job).exists():
                            job.status = "queued"
                            job.pid = None
                            job.returncode = None
                            job.stage = "Checkpoint ditemukan, menyiapkan resume..."
                            job.last_message = "Layanan sebelumnya berhenti; ekstraksi akan dilanjutkan."
                            job.save_status()
                        else:
                            self._recover_terminated_job(job)
                return job

        # 2. Jika tidak ada di memori (misal server streamlit sempat restart), coba load dari disk
        target_dir = output_dir or (PROJECT_ROOT / "output")
        status_file = target_dir / stem / "logs" / f"{stem}_status.json"
        if not status_file.exists():
            status_file = target_dir / "logs" / f"{stem}_status.json"
        if status_file.exists():
            try:
                data = json.loads(status_file.read_text(encoding="utf-8"))
                latest_log = Path(
                    data.get(
                        "latest_log_path",
                        target_dir / "logs" / f"{stem}_latest.log",
                    )
                )
                recent = collections.deque(maxlen=200)
                if latest_log.exists():
                    lines = latest_log.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()
                    recent.extend(lines[-50:])

                job = JobInfo(
                    job_id=data.get("job_id", stem),
                    file_name=data.get("file_name", f"{stem}"),
                    input_path=Path(data.get("input_path", "")),
                    output_dir=Path(data.get("output_dir", target_dir)),
                    out_file=Path(data.get("out_file", target_dir / f"{stem}.md")),
                    db_file=Path(data["db_file"]) if data.get("db_file") else None,
                    log_path=Path(data.get("log_path", latest_log)),
                    latest_log_path=latest_log,
                    status_file=status_file,
                    progress_file=Path(
                        data.get(
                            "progress_file",
                            target_dir / "logs" / f"{stem}_progress.txt",
                        )
                    ),
                    status=data.get("status", "failed"),
                    queue_position=int(data.get("queue_position", 0)),
                    current_page=data.get("current_page", 0),
                    total_pages=data.get("total_pages", 0),
                    stage=data.get("stage", "Tidak diketahui"),
                    last_message=data.get("last_message", ""),
                    started_at=data.get("started_at", ""),
                    updated_at=data.get("updated_at", ""),
                    pid=data.get("pid"),
                    returncode=data.get("returncode"),
                error_message=data.get("error_message"),
                    completed_at=data.get("completed_at"),
                    run_id=data.get("run_id", uuid.uuid4().hex),
                    extraction_options=data.get("extraction_options", {}),
                    recent_logs=recent,
                )

                checkpoint_exists = self._checkpoint_path(job).exists()
                # Job antrean atau gagal dengan checkpoint dapat dilanjutkan setelah restart.
                if job.status in {"queued", "failed"} and checkpoint_exists:
                    job.status = "queued"
                    job.pid = None
                    job.returncode = None
                    job.stage = "Checkpoint ditemukan, menyiapkan resume..."
                    job.last_message = (
                        "Layanan sebelumnya berhenti; ekstraksi akan dilanjutkan."
                    )
                    job.save_status()
                # Job antrean tanpa checkpoint memang belum memiliki worker lagi.
                elif job.status == "queued":
                    self._recover_abandoned_queue(job)
                # Validasi jika status tersimpan "running" tapi PID sudah mati.
                elif job.status == "running" and (
                    job.pid is None or not is_pid_alive(job.pid)
                ):
                    if checkpoint_exists:
                        job.status = "queued"
                        job.pid = None
                        job.returncode = None
                        job.stage = "Checkpoint ditemukan, menyiapkan resume..."
                        job.last_message = (
                            "Layanan sebelumnya berhenti; ekstraksi akan dilanjutkan."
                        )
                        job.save_status()
                    else:
                        self._recover_terminated_job(job)

                with self._lock:
                    self._jobs[stem] = job
                return job
            except Exception as e:  # noqa: BLE001
                logger.warning("Gagal membaca status job dari disk: %s", e)

        # 3. Fallback jika status.json tidak ada namun file markdown ada di folder dokumen
        md_candidate = target_dir / stem / f"{stem}.md"
        if not md_candidate.exists():
            md_candidate = target_dir / f"{stem}.md"
        if md_candidate.exists():
            log_dir = target_dir / stem / "logs"
            latest_log = log_dir / f"{stem}_latest.log"
            job = JobInfo(
                job_id=stem,
                file_name=stem,
                input_path=target_dir / "uploads" / stem,
                output_dir=target_dir,
                out_file=md_candidate,
                db_file=target_dir / stem / "databases" / f"{stem}.sqlite"
                if (target_dir / stem / "databases" / f"{stem}.sqlite").exists()
                else None,
                log_path=latest_log,
                latest_log_path=latest_log,
                status_file=log_dir / f"{stem}_status.json",
                progress_file=log_dir / f"{stem}_progress.txt",
                status="completed",
                current_page=1,
                total_pages=1,
                stage="Selesai (Dipulihkan dari Disk)",
                last_message="Dokumen selesai diekstrak sebelumnya.",
            )
            with self._lock:
                self._jobs[stem] = job
            return job

        return None

    def cancel_job(self, stem: str) -> bool:
        """Hentikan proses ekstraksi paksa jika sedang berjalan."""
        with self._lock:
            job = self._jobs.get(stem)
            proc = self._processes.get(stem)

        if not job or job.status not in {"queued", "running", "paused"}:
            return False

        # Matikan proses sistem operasi
        if proc and proc.poll() is None:
            try:
                # SIGTERM tidak diproses oleh proses yang sedang SIGSTOP.
                if job.status == "paused" and sys.platform != "win32":
                    os.kill(proc.pid, signal.SIGCONT)
                # Di Windows, gunakan taskkill tree agar proses anak (soffice, python) ikut mati
                if sys.platform == "win32" and job.pid:
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(job.pid)],
                        capture_output=True,
                        check=False,
                    )
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
            except Exception as e:  # noqa: BLE001
                logger.warning("Gagal mematikan proses: %s", e)

        with self._lock:
            job.status = "canceled"
            job.stage = "Dibatalkan oleh pengguna"
            job.last_message = "Ekstraksi dibatalkan."
            job.save_status()

        return True

    def pause_job(self, stem: str) -> bool:
        """Jeda job antrean atau suspend subprocess yang sedang berjalan."""
        with self._lock:
            job = self.get_job(stem)
            proc = self._processes.get(stem)
        if not job or job.status not in {"queued", "running"}:
            return False

        if proc and proc.poll() is None:
            try:
                if sys.platform == "win32":
                    import psutil

                    psutil.Process(proc.pid).suspend()
                else:
                    os.kill(proc.pid, signal.SIGSTOP)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Gagal menjeda proses %s: %s", stem, exc)
                return False

            # Slot harus dibuka agar job lain dapat langsung mengambil alih.
            with self._lock:
                slot_path = self._execution_slots.pop(stem, None)
            self._release_execution_slot(slot_path, job.run_id)

        with self._lock:
            job.status = "paused"
            job.stage = "Dijeda oleh pengguna"
            job.last_message = "Ekstraksi dijeda; tekan lanjutkan untuk meneruskan."
            job.save_status()
        return True

    def resume_job(self, stem: str, output_dir: Path) -> bool:
        """Lanjutkan job yang dijeda, baik dari antrean maupun subprocess suspend."""
        job = self.get_job(stem, output_dir=output_dir)
        if not job or job.status != "paused":
            return False
        proc = self._processes.get(stem)
        if proc and proc.poll() is None:
            with self._lock:
                job.status = "queued"
                job.stage = "Menunggu slot untuk melanjutkan..."
                job.last_message = "Menunggu slot kosong untuk melanjutkan ekstraksi."
                job.save_status()
            threading.Thread(
                target=self._resume_suspended_process,
                args=(job, proc),
                daemon=True,
                name=f"Resume-{job.job_id}",
            ).start()
            return True

        with self._lock:
            job.status = "queued"
            job.stage = "Menunggu slot ekstraksi VLM..."
            job.last_message = "Job kembali ke antrean ekstraksi."
            job.save_status()
            worker = self._threads.get(job.job_id)
            if worker is None or not worker.is_alive():
                self._launch_job(job, resume=self._checkpoint_path(job).exists())
        return True

    def _resume_suspended_process(self, job: JobInfo, proc: subprocess.Popen) -> None:
        """Ambil kembali slot lalu lepaskan suspend pada subprocess yang dijeda."""
        slot_path = self._acquire_execution_slot(job)
        if slot_path is None:
            return
        try:
            if proc.poll() is not None:
                self._release_execution_slot(slot_path, job.run_id)
                return
            if sys.platform == "win32":
                import psutil

                psutil.Process(proc.pid).resume()
            else:
                os.kill(proc.pid, signal.SIGCONT)
            with self._lock:
                self._execution_slots[job.job_id] = slot_path
                job.status = "running"
                job.stage = "Proses dilanjutkan"
                job.last_message = "Ekstraksi dilanjutkan oleh pengguna."
                job.save_status()
            self._write_slot_owner(slot_path, job, subprocess_pid=proc.pid)
        except Exception as exc:  # noqa: BLE001
            self._release_execution_slot(slot_path, job.run_id)
            with self._lock:
                job.status = "failed"
                job.error_message = f"Gagal melanjutkan proses: {exc}"
                job.last_message = job.error_message
                job.save_status()

    def prioritize_job(self, stem: str, output_dir: Path | None = None) -> bool:
        """Pindahkan job yang masih dalam antrean ke urutan terdepan."""
        with self._lock:
            job = self.get_job(stem, output_dir=output_dir)
            if not job or job.status != "queued":
                return False
            min_pos = min(
                (
                    other.queue_position
                    for other in self._jobs.values()
                    if other.status in {"queued", "running"}
                ),
                default=0,
            )
            job.queue_position = min_pos - 1
            job.stage = "Diprioritaskan ke antrean terdepan"
            job.last_message = "File ini diprioritaskan oleh pengguna untuk diproses berikutnya."
            job.save_status()
            return True

    def restart_job(self, stem: str, output_dir: Path) -> JobInfo:
        """Mulai ulang langsung, dengan sumber dan opsi proses sebelumnya."""
        job = self.get_job(stem, output_dir=output_dir)
        if job and job.status == "running":
            return job
        if job and job.status == "queued" and self._checkpoint_path(job).exists():
            with self._lock:
                worker = self._threads.get(job.job_id)
                if worker is None or not worker.is_alive():
                    self._launch_job(job, resume=True)
            return job
        source = job.input_path if job else None
        if source is None or not source.is_file():
            uploads = output_dir / "uploads"
            matches = (
                [p for p in uploads.iterdir() if p.is_file() and p.stem == stem]
                if uploads.exists()
                else []
            )
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Sumber dokumen '{stem}' tidak tersedia atau ambigu. Unggah ulang file sumber."
                )
            source = matches[0]
        return self.start_job(
            source, output_dir, **(job.extraction_options if job else {})
        )

    def reset_job(self, stem: str, output_dir: Path | None = None) -> None:
        """Hapus referensi job dari memori dan bersihkan file status agar dapat diekstrak ulang."""
        with self._lock:
            self._jobs.pop(stem, None)
            self._processes.pop(stem, None)
            self._threads.pop(stem, None)

        target_dir = output_dir or (PROJECT_ROOT / "output")
        candidate_files = [
            target_dir / stem / "logs" / f"{stem}_status.json",
            target_dir / "logs" / f"{stem}_status.json",
            target_dir / stem / "logs" / f"{stem}_progress.txt",
            target_dir / "logs" / f"{stem}_progress.txt",
        ]
        for f in candidate_files:
            try:
                if f.exists():
                    f.unlink()
            except Exception as e:  # noqa: BLE001
                logger.warning("Gagal menghapus file status saat reset_job: %s", e)

    def list_all_documents(self, output_dir: Path) -> list[dict[str, Any]]:
        """Daftar seluruh dokumen yang pernah diekstrak atau sedang berjalan."""
        docs: list[dict[str, Any]] = []
        seen_stems: set[str] = set()

        # 1. Dari memori job aktif
        with self._lock:
            for stem, job in self._jobs.items():
                if job.output_dir.resolve() != output_dir.resolve():
                    continue
                seen_stems.add(stem)
                docs.append(
                    {
                        "stem": stem,
                        "status": job.status,
                        "stage": job.stage,
                        "page_count": job.total_pages or job.current_page or 0,
                        "mtime": job.status_file.stat().st_mtime
                        if job.status_file.exists()
                        else 0.0,
                    }
                )

        # 2. Dari direktori output di disk
        if output_dir.exists():
            for item in output_dir.iterdir():
                if not item.is_dir() or item.name in (
                    "logs",
                    "databases",
                    "csv",
                    "uploads",
                    "cache",
                    "batches",
                ):
                    continue
                stem = item.name
                if stem in seen_stems:
                    continue

                md_file = item / f"{stem}.md"
                db_file = item / "databases" / f"{stem}.sqlite"
                status_file = item / "logs" / f"{stem}_status.json"
                if (
                    not md_file.exists()
                    and not db_file.exists()
                    and not status_file.exists()
                ):
                    continue

                seen_stems.add(stem)
                page_count = 0
                pages_dir = item / "pages"
                slides_dir = item / "slides"
                if pages_dir.exists():
                    page_count = len(
                        list(pages_dir.glob("*.png")) + list(pages_dir.glob("*.jpg"))
                    )
                elif slides_dir.exists():
                    page_count = len(
                        list(slides_dir.glob("*.png")) + list(slides_dir.glob("*.jpg"))
                    )

                mtime = max(
                    path.stat().st_mtime
                    for path in (item, md_file, status_file)
                    if path.exists()
                )
                job = self.get_job(stem, output_dir=output_dir)
                status = job.status if job else "completed"
                stage = job.stage if job else "Selesai"

                docs.append(
                    {
                        "stem": stem,
                        "status": status,
                        "stage": stage,
                        "page_count": page_count,
                        "mtime": mtime,
                    }
                )

        docs.sort(
            key=lambda d: (d["status"] in {"queued", "running"}, d["mtime"]),
            reverse=True,
        )
        return docs

    def get_active_job_counts(self, output_dir: Path) -> dict[str, int]:
        """Hitung job ingest aktif, termasuk job yang sedang dijeda."""
        documents = self.list_all_documents(output_dir)
        running = sum(document["status"] == "running" for document in documents)
        queued = sum(document["status"] == "queued" for document in documents)
        paused = sum(document["status"] == "paused" for document in documents)
        return {
            "running": running,
            "queued": queued,
            "paused": paused,
            "active": running + queued + paused,
        }

    def get_latest_logs(self, stem: str, line_count: int = 40) -> str:
        """Ambil potongan baris log terakhir (dari memori atau langsung dari file)."""
        job = self.get_job(stem)
        if job and job.recent_logs:
            return "\n".join(list(job.recent_logs)[-line_count:])

        # Fallback baca dari file disk
        log_file = PROJECT_ROOT / "output" / stem / "logs" / f"{stem}_latest.log"
        if not log_file.exists():
            log_file = PROJECT_ROOT / "output" / "logs" / f"{stem}_latest.log"
        if log_file.exists():
            try:
                lines = log_file.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
                return "\n".join(lines[-line_count:])
            except Exception:  # noqa: BLE001, S110
                pass
        return "(Belum ada log)"
