"""
Modul Orkestrasi Dokumen Agent Mode berbasis LangGraph StateGraph.

Menyediakan StateGraph terstruktur untuk:
1. `render_pipeline`: Resolusi path -> inspeksi dokumen -> render batch 5 slide/halaman -> optimasi payload.
2. `save_pipeline`: Ingesti Markdown per-batch -> stitching multi-halaman -> auto-ingest SQLite & double verification.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .pdf import pdf_page_count, pdf_to_images
from .ppt import count_presentation_slides, render_presentation_slides_to_images
from .prompts import MARKDOWN_LINE_BREAK_RULES, MERMAID_EXTRACTION_RULES
from .tabular_db import extract_and_ingest_tables_from_markdown

logger = logging.getLogger(__name__)

# Root folder proyek
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def resolve_project_path(p: str | Path) -> Path:
    """
    Konversi path relatif menjadi path absolut terhadap root proyek (bukan C:\\Windows\\System32).
    """
    path_obj = Path(p)
    if path_obj.is_absolute():
        return path_obj.resolve()
    return (PROJECT_ROOT / path_obj).resolve()


class DocumentBatchState(TypedDict, total=False):
    """Skema State LangGraph untuk pipeline batch/per-halaman dokumen."""

    source_path: str
    resolved_path: str
    output_dir: str
    resolved_out: str
    doc_type: str  # "pptx", "pdf", "image"
    total_items: int
    current_page: int
    reset_existing: bool
    batch_start: int
    batch_size: int
    batch_end: int
    dpi: int
    rendered_images: list[str]
    has_more: bool
    next_start: int | None
    incoming_markdown: str | None
    ingest_transactional_tables: bool
    specs: str
    merged_markdown: str | None
    is_complete: bool
    saved_pages: list[int]
    missing_pages: list[int]
    page_structure: list[dict[str, Any]]
    tabular_tables: list[dict[str, Any]]
    active_tables_summary: list[dict[str, Any]]
    subagent_task_directives: dict[str, Any]
    status: str
    error: str | None
    instruction: str


def get_subagent_task_directives(
    doc_type: str = "general",
    current_page: int = 1,
    total_pages: int = 1,
    active_tables: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Menyusun katalog direktif peran sub-agent (Deep Agent Role Directives) untuk halaman yang sedang diproses.
    Memaksa model pada setiap giliran untuk mengevaluasi visual dan menerapkan aturan sub-agent terkait.
    """
    return {
        "execution_protocol": (
            "EVALUASI GAMBAR HALAMAN INI: "
            "Pilih 1 atau lebih peran spesialis yang relevan di bawah ini, dan terapkan SEMUA aturan & formatnya secara ketat dalam hasil Markdown Anda."
        ),
        "markdown_format_rules": MARKDOWN_LINE_BREAK_RULES,
        "diagram_format_rules": MERMAID_EXTRACTION_RULES,
        "available_specialist_roles": [
            "mermaid_specialist",
            "tabular_sqlite_specialist",
            "invoice_form_specialist",
            "legal_hierarchy_specialist",
            "presentation_slide_specialist",
            "scientific_math_specialist",
        ],
        "subagent_directives": {
            "mermaid_specialist": {
                "trigger": "Visual memuat alur proses, flowchart, sequence, pohon keputusan, class diagram, ERD, finite state machine, atau mindmap",
                "mandatory_action": (
                    "WAJIB diekstrak sebagai blok kode ```mermaid (misal flowchart TD / flowchart LR / sequenceDiagram). "
                    "DILARANG KERAS menggunakan panah teks biasa (↓, ->, -->) dalam daftar teks untuk diagram alir. "
                    "Gunakan identifier bersih (tanpa spasi/simbol) dan beri tanda kutip ganda pada label teks node. "
                    "Jika satu label node terdiri dari beberapa baris atau memuat line break visual, "
                    "WAJIB pertahankan pemisah baris tersebut sebagai tag HTML <br/> di dalam label yang diapit tanda kutip ganda; "
                    "contoh: A[\"Baris pertama<br/>Baris kedua\"]. Jangan mengganti <br/> dengan spasi atau newline literal "
                    "di dalam satu baris kode Mermaid, dan jangan menaruh <br/> di luar label node."
                ),
            },
            "tabular_sqlite_specialist": {
                "trigger": "Visual memuat tabel log transaksi, mutasi keuangan, invoice, daftar harga, rekapitulasi numerik bertanggal/bernominal",
                "mandatory_action": (
                    "Tulis tabel dalam format GFM Markdown standar dengan header yang jelas. "
                    "Gunakan format tanggal ISO jika memungkinkan (YYYY-MM-DD) dan angka numerik bersih tanpa pemisah ribuan. "
                    "Sistem secara otomatis akan meng-ingest tabel ini ke database SQLite dan memverifikasi integritas baris serta kalkulasi agregat."
                ),
                "existing_active_tables": active_tables or [],
            },
            "legal_hierarchy_specialist": {
                "trigger": "Dokumen hukum/regulasi (UU, PP, Permen, SK, Perda)",
                "mandatory_action": (
                    "Gunakan hierarki heading terstruktur: `# JUDUL`, `## BAB`, `### Bagian`, `#### Paragraf`, `##### Pasal`. "
                    "Format ayat `(1)` dan butir rincian `a.` / `1.` sebagai list Markdown terindentasi."
                ),
            },
            "presentation_slide_specialist": {
                "trigger": "Slide presentasi PowerPoint / PDF slide",
                "mandatory_action": (
                    f"Setiap slide wajib diawali header `<!-- slide: {current_page} -->` diikuti judul slide `# Judul`. "
                    "Ekstrak bullet points, tabel ringkas, serta speaker notes jika ada."
                ),
            },
            "scientific_math_specialist": {
                "trigger": "Jurnal ilmiah 2-kolom atau rumus matematika",
                "mandatory_action": (
                    "Baca kolom kiri dari atas ke bawah hingga tuntas sebelum membaca kolom kanan. "
                    "Tulis rumus matematika dalam format LaTeX standard (`$...$` inline, `$$...$$` block)."
                ),
            },
        },
        "context_info": {
            "page_number": current_page,
            "total_pages": total_pages,
            "doc_type": doc_type,
        },
    }


class AgentDocumentGraph:
    """
    Kompilasi dan orkestrasi StateGraph LangGraph untuk memproses dokumen secara deterministik.
    """

    def __init__(self) -> None:
        self.render_graph: CompiledStateGraph = self._build_render_graph()
        self.save_graph: CompiledStateGraph = self._build_save_graph()
        self.advance_graph: CompiledStateGraph = self._build_advance_graph()

    # --- Node Implementations ---

    @staticmethod
    def _node_resolve_and_inspect(state: DocumentBatchState) -> dict[str, Any]:
        """Node 1: Resolusi path absolut & inspeksi jumlah halaman/slide."""
        raw_source = state.get("source_path", "")
        if not raw_source:
            return {"status": "error", "error": "Path file sumber tidak boleh kosong"}

        resolved = resolve_project_path(raw_source)
        if not resolved.exists():
            return {
                "status": "error",
                "error": f"File dokumen tidak ditemukan: {resolved}",
            }

        ext = resolved.suffix.lower()
        if ext in (".pptx", ".ppt"):
            doc_type = "pptx"
            try:
                total_items = count_presentation_slides(resolved)
            except Exception as exc:  # noqa: BLE001
                return {
                    "status": "error",
                    "error": f"Gagal menghitung slide PPTX: {exc}",
                }
        elif ext == ".pdf":
            doc_type = "pdf"
            try:
                total_items = pdf_page_count(resolved)
            except Exception as exc:  # noqa: BLE001
                return {
                    "status": "error",
                    "error": f"Gagal menghitung halaman PDF: {exc}",
                }
        elif ext in (".png", ".jpg", ".jpeg", ".webp"):
            doc_type = "image"
            total_items = 1
        else:
            return {
                "status": "error",
                "error": f"Format file tidak didukung: {ext}",
            }

        raw_out = state.get("output_dir", "")
        if raw_out:
            resolved_out = resolve_project_path(raw_out)
        else:
            resolved_out = PROJECT_ROOT / "output" / resolved.stem

        return {
            "resolved_path": str(resolved),
            "doc_type": doc_type,
            "total_items": total_items,
            "resolved_out": str(resolved_out),
            "status": "inspected",
        }

    @staticmethod
    def _node_render_batch(state: DocumentBatchState) -> dict[str, Any]:
        """Node 2: Render batch slide/halaman menjadi gambar resolusi tinggi."""
        if state.get("status") == "error":
            return {}

        resolved = Path(state["resolved_path"])
        doc_type = state.get("doc_type", "image")
        total_items = state.get("total_items", 1)
        start_idx = max(1, state.get("batch_start", 1))
        batch_size = max(1, state.get("batch_size", 5))
        dpi = state.get("dpi", 150)
        resolved_out = Path(state["resolved_out"])
        resolved_out.mkdir(parents=True, exist_ok=True)

        end_idx = min(start_idx + batch_size - 1, total_items)
        rendered_images: list[str] = []

        if doc_type == "pptx":
            target_indices = list(range(start_idx - 1, end_idx))
            slides_out = resolved_out / "slides"
            slides_out.mkdir(parents=True, exist_ok=True)
            try:
                images = render_presentation_slides_to_images(
                    resolved,
                    output_dir=slides_out,
                    slides=target_indices,
                    dpi=dpi,
                    image_ext="jpg",
                )
                rendered_images = [str(p) for p in images]
            except Exception as exc:  # noqa: BLE001
                return {
                    "status": "error",
                    "error": f"Gagal merender slide presentasi: {exc}",
                }

        elif doc_type == "pdf":
            pages_out = resolved_out / "pages"
            pages_out.mkdir(parents=True, exist_ok=True)
            try:
                all_pages = pdf_to_images(
                    resolved, output_dir=pages_out, dpi=dpi, image_ext="jpg"
                )
                rendered_images = [
                    str(p) for p in all_pages[start_idx - 1 : end_idx]
                ]
            except Exception as exc:  # noqa: BLE001
                return {
                    "status": "error",
                    "error": f"Gagal merender halaman PDF: {exc}",
                }

        elif doc_type == "image":
            rendered_images = [str(resolved)]

        has_more = end_idx < total_items
        next_start = end_idx + 1 if has_more else None

        from .tabular_db import TabularDatabaseManager

        db_path = resolved_out / "databases" / f"{resolved.stem}.sqlite"
        active_tables = []
        if db_path.exists():
            active_tables = TabularDatabaseManager(db_path).get_active_tables_summary()

        directives = get_subagent_task_directives(
            doc_type=doc_type,
            current_page=start_idx,
            total_pages=total_items,
            active_tables=active_tables,
        )

        return {
            "current_page": start_idx,
            "batch_start": start_idx,
            "batch_end": end_idx,
            "rendered_images": rendered_images,
            "has_more": has_more,
            "next_start": next_start,
            "subagent_task_directives": directives,
            "status": "rendered",
            "instruction": (
                f"Batch {start_idx}-{end_idx} ({len(rendered_images)} gambar) berhasil dirender. "
                "Evaluasi visual dan ekstrak ke Markdown."
            ),
        }

    @staticmethod
    def _node_save_and_stitch(state: DocumentBatchState) -> dict[str, Any]:
        """Node 3: Simpan Markdown, stitching parsial/lengkap, dan auto-ingest SQLite."""
        if state.get("status") == "error":
            return {}

        resolved = Path(state["resolved_path"])
        total_items = state.get("total_items", 1)
        raw_markdown = state.get("incoming_markdown", "")
        if not raw_markdown:
            return {
                "status": "error",
                "error": "Markdown masukan kosong pada proses penyimpanan",
            }

        from .multi_page import split_markdown_by_pages, stitch_pages_to_markdown
        from .schemas import DocumentPage
        from .tabular_db import TabularDatabaseManager

        out_base = Path(state["resolved_out"])
        out_base.mkdir(parents=True, exist_ok=True)
        chunks_dir = out_base / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)

        current_page_in = state.get("current_page", state.get("batch_start", 1))

        # Menghapus chunk lama adalah aksi destruktif. Hanya lakukan saat pemanggil
        # secara eksplisit menandai re-ekstraksi penuh; batch biasa harus selalu
        # mempertahankan halaman yang telah disimpan sebelumnya.
        if state.get("reset_existing", False) and current_page_in == 1 and total_items > 1:
            for old_p in chunks_dir.glob("page_*.md"):
                try:
                    num = int(old_p.stem.split("_")[-1])
                    if num > 1:
                        old_p.unlink(missing_ok=True)
                except ValueError:
                    pass

        doc_type = state.get("doc_type", "")
        specs = state.get("specs", "plain")
        is_slide = doc_type == "pptx" or "slide" in specs or "presentation" in specs

        pages_parsed = split_markdown_by_pages(
            raw_markdown,
            default_page_number=current_page_in,
            default_type="slide" if is_slide else "page",
        )

        for p in pages_parsed:
            p_num = int(p["page_number"])
            p_content = str(p["content"])
            p_file = chunks_dir / f"page_{p_num:04d}.md"
            p_file.write_text(p_content, encoding="utf-8")

        all_page_files = sorted(chunks_dir.glob("page_*.md"))
        all_pages_accumulated: list[DocumentPage] = []
        for pf in all_page_files:
            try:
                num = int(pf.stem.split("_")[-1])
                all_pages_accumulated.append(
                    DocumentPage(
                        page_number=num,
                        markdown_content=pf.read_text(encoding="utf-8"),
                    )
                )
            except ValueError:
                continue

        all_pages_accumulated.sort(key=lambda x: x.page_number)
        merged_md = stitch_pages_to_markdown(
            all_pages_accumulated, is_slide=is_slide
        )

        out_md = out_base / f"{resolved.stem}.md"
        out_md.write_text(merged_md, encoding="utf-8")

        got_numbers = [p.page_number for p in all_pages_accumulated]
        missing = [i for i in range(1, total_items + 1) if i not in got_numbers]
        is_complete = len(missing) == 0 and len(got_numbers) >= total_items

        page_structure = [
            {
                "page": p.page_number,
                "char_count": len(p.markdown_content),
                "line_count": len(p.markdown_content.splitlines()),
                "headings": [
                    line.strip()
                    for line in p.markdown_content.splitlines()
                    if line.strip().startswith("#")
                ],
            }
            for p in all_pages_accumulated
        ]

        # Auto-ingest tabel ke SQLite
        db_dir = out_base / "databases"
        db_dir.mkdir(parents=True, exist_ok=True)
        db_file = db_dir / f"{resolved.stem}.sqlite"

        tabular_info: list[dict[str, Any]] = []
        if state.get("ingest_transactional_tables", True):
            try:
                tab_results = extract_and_ingest_tables_from_markdown(
                    markdown_text=merged_md,
                    source_file=str(resolved),
                    db_path=db_file,
                    force_all_tables=False,
                )
                for res in tab_results:
                    if res.status == "success":
                        tabular_info.append({
                            "table_name": res.table_name,
                            "rows_ingested": res.total_rows_ingested,
                            "columns": res.columns,
                            "verified": res.verification_report.is_valid
                            if res.verification_report
                            else False,
                        })
            except Exception as exc:  # noqa: BLE001
                logger.warning("Gagal auto-ingest tabel transaksional ke SQLite: %s", exc)

        active_tables = (
            TabularDatabaseManager(db_file).get_active_tables_summary()
            if db_file.exists()
            else []
        )

        metadata = {
            "source_file": str(resolved),
            "source_extension": resolved.suffix.lower(),
            "specs": [
                s.strip() for s in state.get("specs", "plain").split(",") if s.strip()
            ],
            "extracted_by": "agent",
            "saved_at": datetime.now(UTC).isoformat(),
            "markdown_path": str(out_md),
            "char_count": len(merged_md),
            "line_count": len(merged_md.splitlines()),
            "total_pages_detected": len(pages_parsed),
            "total_pages_in_source": total_items,
            "is_complete": is_complete,
            "pages_saved": got_numbers,
            "missing_pages": missing,
            "page_structure": page_structure,
            "tabular_database_tables": tabular_info,
            "active_database_tables": active_tables,
        }

        out_meta = out_base / f"{resolved.stem}.meta.json"
        out_meta.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        msg = (
            "Seluruh halaman/slide dokumen telah selesai diekstraksi dan disimpan secara lengkap."
            if is_complete
            else f"Halaman/Slide berhasil disimpan ({len(got_numbers)} dari {total_items} halaman/slide tersimpan). Lanjutkan ke nomor: {missing[:5]}..."
        )

        return {
            "merged_markdown": merged_md,
            "is_complete": is_complete,
            "saved_pages": got_numbers,
            "missing_pages": missing,
            "page_structure": page_structure,
            "tabular_tables": tabular_info,
            "active_tables_summary": active_tables,
            "status": "success" if is_complete else "batch_saved",
            "instruction": msg,
        }

    def _node_advance_to_next(self, state: DocumentBatchState) -> dict[str, Any]:
        """Node 4: Jika belum selesai, langsung render halaman berikutnya untuk menghemat 1 round-trip."""
        if state.get("status") == "error" or state.get("is_complete"):
            return {}

        missing = state.get("missing_pages", [])
        if not missing:
            return {"is_complete": True}

        next_page = missing[0]
        advance_state = dict(state)
        advance_state["batch_start"] = next_page
        advance_state["batch_size"] = 1
        advance_state["batch_end"] = next_page

        render_res = self._node_render_batch(cast(Any, advance_state))
        return {
            "current_page": next_page,
            "rendered_images": render_res.get("rendered_images", []),
            "subagent_task_directives": render_res.get("subagent_task_directives", {}),
            "has_more": True,
            "next_start": missing[1] if len(missing) > 1 else None,
            "instruction": (
                f"Halaman/Slide {next_page} berhasil dirender. "
                f"Evaluasi visual gambar dan jalankan peran subagent yang sesuai (Mermaid/Tabular/Hierarchy/Scientific/Slide). "
                f"Kirim hasil Markdown via 'submit_page_and_get_next'."
            ),
        }

    # --- Graph Builders ---

    def _build_render_graph(self) -> CompiledStateGraph:
        builder = StateGraph(cast(Any, DocumentBatchState))
        builder.add_node("resolve_and_inspect", self._node_resolve_and_inspect)
        builder.add_node("render_batch", self._node_render_batch)

        builder.add_edge(START, "resolve_and_inspect")
        builder.add_edge("resolve_and_inspect", "render_batch")
        builder.add_edge("render_batch", END)

        return builder.compile()

    def _build_save_graph(self) -> CompiledStateGraph:
        builder = StateGraph(cast(Any, DocumentBatchState))
        builder.add_node("resolve_and_inspect", self._node_resolve_and_inspect)
        builder.add_node("save_and_stitch", self._node_save_and_stitch)
        builder.add_node("advance_to_next", self._node_advance_to_next)

        builder.add_edge(START, "resolve_and_inspect")
        builder.add_edge("resolve_and_inspect", "save_and_stitch")
        builder.add_edge("save_and_stitch", "advance_to_next")
        builder.add_edge("advance_to_next", END)

        return builder.compile()

    def _build_advance_graph(self) -> CompiledStateGraph:
        builder = StateGraph(cast(Any, DocumentBatchState))
        builder.add_node("resolve_and_inspect", self._node_resolve_and_inspect)
        builder.add_node("save_and_stitch", self._node_save_and_stitch)
        builder.add_node("advance_to_next", self._node_advance_to_next)

        builder.add_edge(START, "resolve_and_inspect")
        builder.add_edge("resolve_and_inspect", "save_and_stitch")
        builder.add_edge("save_and_stitch", "advance_to_next")
        builder.add_edge("advance_to_next", END)

        return builder.compile()


_agent_document_graph_instance: AgentDocumentGraph | None = None


def get_agent_document_graph() -> AgentDocumentGraph:
    """Mengembalikan instance singleton dari AgentDocumentGraph."""
    global _agent_document_graph_instance
    if _agent_document_graph_instance is None:
        _agent_document_graph_instance = AgentDocumentGraph()
    return _agent_document_graph_instance
