"""
Orkestrasi Pipeline Ekstraksi Dokumen VLM -> Markdown Siap Chunking dengan LangGraph.
Mendukung multi-spesifikasi komposit layout dokumen dengan logging transparan.

Alur StateGraph:
    START -> preprocess -> classify -> extract_markdown -> END
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Literal, TypedDict, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .agents import get_agent
from .config import Settings, get_settings
from .extractor import VisionExtractor
from .llm import build_vlm
from .multi_page import extract_document_title
from .preprocess import preprocess_image
from .prompts import normalize_specs
from .schemas import PipelinePageResult

logger = logging.getLogger("app.graph")


class DocumentExtractionState(TypedDict, total=False):
    image_path: str
    preprocessed_path: str
    forced_specs: list[str] | str | None
    forced_doc_type: str | None
    previous_page_context: str | None
    specs: list[str]
    doc_type: str
    has_diagram: bool
    diagram_type: str | None
    has_table: bool
    difficulty: str
    visual_count: int
    table_count: int
    markdown_content: str
    diagram_mermaid_code: str | None
    diagram_summary: str | None
    document_title: str | None
    is_first_page: bool
    final_markdown: str


# --- Adaptive Fast-Path Helpers (0 biaya VLM) --------------------------------

DIAGRAM_OUTPUT_INDICATORS: tuple[str, ...] = (
    "[Diagram/Visual]",
    "[Gambar/Visual]",
    "[Topologi]",
    "[Diagram/Topologi]",
    "```mermaid",
)

_FIGURE_LABEL_RE = re.compile(r"\b(FIGURE|Figure|Bagan|Skema)\s+\d+", re.IGNORECASE)
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?[\s:|-]*-{3,}[\s:|-]*\|?\s*$")


def _has_diagram_indicators(markdown: str) -> bool:
    """Deteksi indikator diagram/visual dari output ekstraksi (tanpa VLM call tambahan)."""
    if any(ind in markdown for ind in DIAGRAM_OUTPUT_INDICATORS):
        return True
    return bool(_FIGURE_LABEL_RE.search(markdown))


def count_visuals(markdown: str) -> int:
    """Hitung jumlah elemen visual (diagram, topologi, gambar, ilustrasi) dalam output."""
    count = 0
    for tag in (
        "[Diagram/Visual]",
        "[Gambar/Visual]",
        "[Topologi]",
        "[Diagram/Topologi]",
    ):
        count += markdown.count(tag)
    # Blok mermaid yang sudah tertulis juga dihitung sebagai 1 visual
    count += markdown.count("```mermaid")
    return count


def count_tables(markdown: str) -> int:
    """Hitung jumlah tabel GFM berdasarkan baris pemisah header (| --- | --- |)."""
    return sum(
        1 for line in markdown.splitlines() if _TABLE_SEPARATOR_RE.match(line)
    )


def _assess_difficulty_from_output(markdown: str) -> str:
    """Heuristic difficulty post-extraction: simple / standard / complex (0 biaya VLM)."""
    if _has_diagram_indicators(markdown):
        return "complex"
    if "|---" in markdown or markdown.count("|") > 8:
        return "standard"
    if len(markdown) > 2000:
        return "standard"
    return "simple"


def _should_skip_judge(markdown: str, difficulty: str, has_diagram: bool) -> bool:
    """Fast-path: skip judge untuk halaman simple yang output-nya bersih & tidak berisiko."""
    if difficulty != "simple":
        return False
    if has_diagram:
        return False
    stripped = markdown.strip()
    if len(stripped) < 20:
        # Output mencurigakan (hampir kosong) -> tetap judge
        return False
    if "[tidak terbaca]" in markdown:
        # Ada area tidak terbaca -> tetap judge untuk attempt recovery
        return False
    # Ada tabel -> integritas data penting, tetap judge
    return "|---" not in markdown


class DocumentExtractionPipeline:
    """Pipeline LangGraph untuk mengekstrak dokumen gambar/scan ke Markdown siap chunking."""

    def __init__(
        self,
        settings: Settings | None = None,
        vlm: BaseChatModel | Any | None = None,
        *,
        thorough: bool = False,
    ) -> None:
        self.settings: Settings = settings or get_settings()
        self.vlm: BaseChatModel = vlm or build_vlm(self.settings)
        self.extractor = VisionExtractor(self.vlm)
        self.thorough: bool = thorough
        self.graph: CompiledStateGraph = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph:
        builder = StateGraph(cast(Any, DocumentExtractionState))

        # Node pipeline
        builder.add_node("preprocess", self._node_preprocess)
        builder.add_node("inspect_and_classify", self._node_inspect_and_classify)
        builder.add_node("extract_markdown", self._node_extract_markdown)
        builder.add_node("summon_diagram_specialist", self._node_summon_diagram_specialist)
        builder.add_node("aggregate_and_judge", self._node_aggregate_and_judge)

        # Edges
        builder.add_edge(START, "preprocess")
        builder.add_edge("preprocess", "inspect_and_classify")
        builder.add_edge("inspect_and_classify", "extract_markdown")
        builder.add_edge("extract_markdown", "summon_diagram_specialist")
        builder.add_edge("summon_diagram_specialist", "aggregate_and_judge")
        builder.add_edge("aggregate_and_judge", END)

        return builder.compile()

    def run(
        self,
        image_path: str,
        *,
        forced_specs: list[str] | str | None = None,
        forced_doc_type: str | None = None,
        previous_page_context: str | None = None,
        is_first_page: bool = False,
    ) -> PipelinePageResult:
        """
        Jalankan pipeline ekstraksi lengkap pada satu gambar halaman dokumen.

        Returns:
            PipelinePageResult terstruktur dan tervalidasi Pydantic.
        """
        initial_state: DocumentExtractionState = {
            "image_path": image_path,
            "forced_specs": forced_specs,
            "forced_doc_type": forced_doc_type,
            "previous_page_context": previous_page_context,
            "is_first_page": is_first_page,
        }

        logger.info("[Pipeline] Memulai ekstraksi: %s (is_first_page=%s)", image_path, is_first_page)
        final_state = cast(dict[str, Any], self.graph.invoke(initial_state))

        final_md = final_state.get("markdown_content", "")
        diff_val = str(final_state.get("difficulty", "standard")).lower()
        clean_difficulty: Literal["simple", "standard", "complex"] = (
            diff_val if diff_val in ("simple", "standard", "complex") else "standard"
        )
        return PipelinePageResult(
            preprocessed_path=final_state.get("preprocessed_path", image_path),
            specs=final_state.get("specs", ["plain"]),
            doc_type=final_state.get("doc_type", "plain"),
            markdown_content=final_md,
            has_diagram=final_state.get("has_diagram", False),
            diagram_mermaid_code=final_state.get("diagram_mermaid_code"),
            difficulty=clean_difficulty,
            visual_count=count_visuals(final_md),
            table_count=count_tables(final_md),
            document_title=final_state.get("document_title"),
        )

    # =========================================================================
    # Node Implementations
    # =========================================================================

    def _node_preprocess(
        self, state: DocumentExtractionState
    ) -> DocumentExtractionState:
        """Tahap 1: Preprocessing gambar."""
        t0 = time.perf_counter()
        image_path = state["image_path"]

        result = preprocess_image(image_path)
        elapsed = (time.perf_counter() - t0) * 1000

        logger.info(
            "[Pipeline:Preprocess] %s -> %s (modified: %s, %dx%d, %.1fms)",
            image_path,
            result.processed_path,
            result.is_modified,
            result.dimensions[0],
            result.dimensions[1],
            elapsed,
        )

        return {
            **state,
            "preprocessed_path": result.processed_path,
        }

    def _node_inspect_and_classify(
        self, state: DocumentExtractionState
    ) -> DocumentExtractionState:
        """Tahap 2: Inspeksi multimodal karakteristik dokumen & deteksi elemen visual/diagram."""
        forced_specs = state.get("forced_specs")
        forced_doc_type = state.get("forced_doc_type")
        img = state.get("preprocessed_path") or state["image_path"]

        # Jika spesifikasi dipaksa secara manual oleh user
        if forced_specs:
            norm_specs = normalize_specs(forced_specs)
            logger.info("[Pipeline:Classify] Menggunakan forced_specs: %s", norm_specs)
            return {
                **state,
                "specs": norm_specs,
                "doc_type": ",".join(norm_specs),
                # Diagram dideteksi post-extraction via output indicators (fast-path),
                # bukan diasumsikan ada hanya karena spec = presentation_slides.
                "has_diagram": False,
                "diagram_type": None,
                "difficulty": "standard",
            }

        if forced_doc_type:
            norm_specs = normalize_specs(forced_doc_type)
            logger.info("[Pipeline:Classify] Menggunakan forced_doc_type: %s", norm_specs)
            return {
                **state,
                "specs": norm_specs,
                "doc_type": ",".join(norm_specs),
                "has_diagram": False,
                "diagram_type": None,
                "difficulty": "standard",
            }

        # Inspeksi otomatis via VLM
        t0 = time.perf_counter()
        is_first = bool(state.get("is_first_page", False))
        insp_res = self.extractor.inspect_page(img, is_first_page=is_first)
        elapsed = (time.perf_counter() - t0) * 1000

        detected_specs = insp_res.get("specs", ["plain"])
        has_diag = bool(insp_res.get("has_diagram", False))
        diag_type = insp_res.get("diagram_type")
        has_tbl = bool(insp_res.get("has_table", False))
        difficulty = str(insp_res.get("difficulty", "standard"))
        doc_title = getattr(insp_res, "document_title", None) or insp_res.get("document_title")

        if doc_title:
            logger.info("[Pipeline:Classify] Judul dokumen terdeteksi: '%s'", doc_title)

        logger.info(
            "[Pipeline:Classify] Layout: %s | Diagram: %s (%s) | Tabel: %s | Difficulty: %s%s (%.1fms)",
            detected_specs,
            has_diag,
            diag_type,
            has_tbl,
            difficulty,
            f" | Judul: '{doc_title}'" if doc_title else "",
            elapsed,
        )

        return {
            **state,
            "specs": detected_specs,
            "doc_type": ",".join(detected_specs),
            "has_diagram": has_diag,
            "diagram_type": diag_type,
            "has_table": has_tbl,
            "difficulty": difficulty,
            "document_title": doc_title or state.get("document_title"),
        }

    def _node_extract_markdown(
        self, state: DocumentExtractionState
    ) -> DocumentExtractionState:
        """Tahap 3: Ekstraksi teks & tata letak Markdown menggunakan Composite Agent."""
        t0 = time.perf_counter()
        img = state.get("preprocessed_path") or state["image_path"]
        specs = state.get("specs", ["plain"])
        prev_context = state.get("previous_page_context")

        agent = get_agent(specs)
        md_text = agent.run(
            img,
            llm=self.vlm,
            previous_page_context=prev_context,
        )

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "[Pipeline:ExtractMarkdown] Ekstraksi teks selesai (%d karakter, %.1fms)",
            len(md_text),
            elapsed,
        )

        # Jika halaman pertama dan judul belum didapat dari inspeksi, ekstrak dari hasil markdown
        current_title = state.get("document_title")
        if state.get("is_first_page") and not current_title:
            extracted_title = extract_document_title(md_text)
            if extracted_title:
                current_title = extracted_title
                logger.info("[Pipeline:ExtractMarkdown] Judul dokumen diekstrak dari Markdown Halaman 1: '%s'", current_title)

        return {
            **state,
            "markdown_content": md_text,
            "document_title": current_title,
        }

    def _node_summon_diagram_specialist(
        self, state: DocumentExtractionState
    ) -> DocumentExtractionState:
        """Tahap 4: Summon Sub-Agent Spesialis Diagram Mermaid HANYA jika diagram terdeteksi."""
        has_diag = state.get("has_diagram", False)
        specs = state.get("specs", [])
        md_content = state.get("markdown_content", "")
        img = state.get("preprocessed_path") or state["image_path"]

        # Deteksi diagram dari output ekstraksi (0 biaya VLM).
        # Prompt presentation_slides menginstruksikan VLM menulis
        # "> **[Diagram/Visual]:** ..." jika ada diagram -> kita manfaatkan itu.
        output_has_diagram = _has_diagram_indicators(md_content)

        if self.thorough:
            should_summon = has_diag or output_has_diagram or "presentation_slides" in specs
        else:
            should_summon = has_diag or output_has_diagram

        if not should_summon:
            logger.info(
                "[Pipeline:SummonSpecialist] Skip diagram specialist (fast mode: tidak ada indikator diagram)"
            )
            return state

        t0 = time.perf_counter()
        logger.info("[Pipeline:SummonSpecialist] Men-summon Sub-Agent Diagram Mermaid...")
        diag_hint = state.get("diagram_type")

        from .diagram import extract_diagram_to_mermaid

        diag_result = extract_diagram_to_mermaid(
            image_path=img,
            llm=self.vlm,
            forced_diagram_type=diag_hint,
        )
        elapsed = (time.perf_counter() - t0) * 1000

        mermaid_code = diag_result.mermaid_code
        diag_summary = diag_result.text_summary

        if mermaid_code:
            logger.info(
                "[Pipeline:SummonSpecialist] Berhasil mengekstrak Diagram Mermaid (%s, %.1fms)",
                diag_result.diagram_type,
                elapsed,
            )
        else:
            logger.info(
                "[Pipeline:SummonSpecialist] Diagram tidak cocok ke Mermaid, fallback ke deskripsi (%.1fms)",
                elapsed,
            )

        return {
            **state,
            "diagram_mermaid_code": mermaid_code,
            "diagram_summary": diag_summary,
        }

    def _node_aggregate_and_judge(
        self, state: DocumentExtractionState
    ) -> DocumentExtractionState:
        """
        Tahap 5: Aggregator & Judge (Koreksi Ulang).
        Menggabungkan teks markdown dengan luaran spesialis diagram, lalu memverifikasi ulang terhadap gambar asli.
        Mode adaptif: halaman 'simple' yang bersih di-skip judge-nya (hemat 1 VLM call).
        """
        t0 = time.perf_counter()
        img = state.get("preprocessed_path") or state["image_path"]
        md_text = state.get("markdown_content", "")
        mermaid_code = state.get("diagram_mermaid_code")
        diag_summary = state.get("diagram_summary")
        specs = state.get("specs", ["plain"])

        # 1. Satukan blok diagram Mermaid ke Markdown jika belum ada
        combined_md = md_text
        if mermaid_code:
            from .diagram import sanitize_mermaid_code, validate_mermaid_syntax

            # Ganti draft yang rusak dengan hasil spesialis yang sudah valid.
            specialist = sanitize_mermaid_code(mermaid_code)
            if specialist and validate_mermaid_syntax(specialist)[0]:
                replaced = False

                def recover_invalid(m: re.Match) -> str:
                    nonlocal replaced
                    draft_code = sanitize_mermaid_code(m.group(1))
                    if not replaced and (not draft_code or not validate_mermaid_syntax(draft_code)[0]):
                        replaced = True
                        return f"```mermaid\n{specialist}\n```"
                    return m.group(0)

                combined_md = re.sub(r"```mermaid\s*([\s\S]*?)\s*```", recover_invalid, combined_md, flags=re.IGNORECASE)
        if mermaid_code and "```mermaid" not in combined_md:
            mermaid_block = f"\n\n```mermaid\n{mermaid_code}\n```"
            if diag_summary:
                mermaid_block += f"\n\n> **[Diagram Summary]:** {diag_summary}"
            combined_md = combined_md + "\n" + mermaid_block

        # 2. Fast-path: skip judge untuk halaman simple yang bersih
        difficulty = state.get("difficulty") or _assess_difficulty_from_output(md_text)
        has_diagram = state.get("has_diagram", False) or _has_diagram_indicators(md_text)

        if not self.thorough and _should_skip_judge(combined_md, difficulty, has_diagram):
            logger.info(
                "[Pipeline:AggregateJudge] Skip judge (fast mode: difficulty=%s, %d karakter bersih)",
                difficulty,
                len(combined_md),
            )
            return {
                **state,
                "difficulty": difficulty,
                "markdown_content": combined_md,
                "final_markdown": combined_md,
            }

        # 3. Lakukan evaluasi koreksi ulang (Judge & Refine)
        final_md = self.extractor.judge_and_refine(
            image_path=img,
            draft_markdown=combined_md,
            specs=specs,
        )

        # 4. Guardrail Pasca-Judge: Sanitasi tabel & validasi ulang blok Mermaid
        from .diagram import sanitize_mermaid_code, validate_mermaid_syntax
        from .tabular_db import sanitize_markdown_tables

        final_md = sanitize_markdown_tables(final_md)
        original_blocks = re.findall(r"```mermaid\s*([\s\S]*?)\s*```", combined_md, re.IGNORECASE)
        final_blocks = re.findall(r"```mermaid\s*([\s\S]*?)\s*```", final_md, re.IGNORECASE)

        def valid_block(code: str) -> bool:
            sanitized = sanitize_mermaid_code(code)
            return bool(sanitized and validate_mermaid_syntax(sanitized)[0])

        if original_blocks and all(valid_block(code) for code in original_blocks) and (
            len(final_blocks) < len(original_blocks) or not all(valid_block(code) for code in final_blocks)
        ):
            final_md = sanitize_markdown_tables(combined_md)

        def _clean_mermaid_in_md(m: re.Match) -> str:
            raw_code = m.group(1)
            sanitized = sanitize_mermaid_code(raw_code)
            if sanitized:
                is_valid, _ = validate_mermaid_syntax(sanitized)
                if is_valid:
                    return f"```mermaid\n{sanitized}\n```"
            return "> **[Diagram/Visual]:** Diagram visual terdeteksi; kode Mermaid tidak valid."

        final_md = re.sub(
            r"```mermaid\s*([\s\S]*?)\s*```",
            _clean_mermaid_in_md,
            final_md,
            flags=re.IGNORECASE,
        )

        elapsed = (time.perf_counter() - t0) * 1000

        logger.info(
            "[Pipeline:AggregateJudge] Koreksi ulang selesai: %d -> %d karakter (%.1fms)",
            len(combined_md),
            len(final_md),
            elapsed,
        )

        return {
            **state,
            "difficulty": difficulty,
            "markdown_content": final_md,
            "final_markdown": final_md,
        }


VisionRAGPipeline = DocumentExtractionPipeline

__all__ = [
    "DocumentExtractionPipeline",
    "DocumentExtractionState",
    "VisionRAGPipeline",
]
