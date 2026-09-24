"""
Orkestrasi Pipeline Ekstraksi Dokumen VLM -> Markdown Siap Chunking dengan LangGraph.
Mendukung multi-spesifikasi komposit layout dokumen dengan logging transparan.

Alur StateGraph:
    START -> preprocess -> inspect/orient -> OCR quality gate -> draft -> specialist -> judge -> END
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .agents import get_agent
from .config import Settings, get_settings
from .extractor import VisionExtractor
from .llm import build_vlm
from .multi_page import extract_document_title
from .ocr import UnlimitedOCRExtractor
from .paddle_ocr import PaddleOCRVLExtractor
from .preprocess import preprocess_image, rotate_image_right_angle
from .prompts import normalize_specs
from .schemas import OCRExtractionResult, OCRRegion, PipelinePageResult
from .text_reflow import reflow_regions

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
    page_number: int | None
    region_output_dir: str | None
    native_text: str | None
    inspection_rotation_degrees: int
    requires_vlm_reading: bool
    force_vlm_reading: bool
    ocr_force_judge: bool
    ocr_result: dict[str, Any]
    ocr_status: str
    ocr_regions: list[dict[str, Any]]
    diagram_mermaid_codes: list[str]
    diagram_summaries: list[str]
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
        ocr_llm: BaseChatModel | Any | None = None,
        *,
        thorough: bool = False,
        ocr_extractor: Any | None = None,
    ) -> None:
        self.settings: Settings = settings or get_settings()
        self.vlm: BaseChatModel = vlm or build_vlm(self.settings)
        self.extractor = VisionExtractor(self.vlm)
        self.ocr_extractor: Any | None = ocr_extractor
        if self.ocr_extractor is None and ocr_llm is not None:
            # Kompatibilitas test/caller lama; runtime normal memakai PaddleOCR-VL.
            self.ocr_extractor = UnlimitedOCRExtractor(
                ocr_llm,
                model_name=self.settings.ocr_model or "injected-ocr",
                prompt=self.settings.ocr_prompt,
                coordinate_size=self.settings.ocr_coordinate_size,
                crop_padding=self.settings.ocr_crop_padding,
                min_trust_score=self.settings.ocr_min_trust_score,
                medium_trust_score=self.settings.ocr_medium_trust_score,
                rotation_retry=self.settings.ocr_rotation_retry,
                blank_ink_ratio=self.settings.ocr_blank_ink_ratio,
                sparse_ink_ratio=self.settings.ocr_sparse_ink_ratio,
            )
        elif (
            self.ocr_extractor is None
            and self.settings.ocr_model
            and self.settings.ocr_backend == "paddleocr_vl"
        ):
            self.ocr_extractor = PaddleOCRVLExtractor(
                base_url=self.settings.ocr_base_url,
                model_name=self.settings.ocr_model,
                api_key=self.settings.ocr_api_key,
                timeout=self.settings.ocr_timeout,
                max_tokens=self.settings.ocr_max_tokens,
                crop_padding=self.settings.ocr_crop_padding,
                min_trust_score=self.settings.ocr_min_trust_score,
                medium_trust_score=self.settings.ocr_medium_trust_score,
                rotation_retry=self.settings.ocr_rotation_retry,
                blank_ink_ratio=self.settings.ocr_blank_ink_ratio,
                sparse_ink_ratio=self.settings.ocr_sparse_ink_ratio,
            )
        elif (
            self.ocr_extractor is None
            and self.settings.ocr_model
            and self.settings.ocr_backend != "disabled"
        ):
            raise ValueError(f"OCR_BACKEND tidak didukung: {self.settings.ocr_backend}")
        self.thorough: bool = thorough
        self.graph: CompiledStateGraph = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph:
        builder = StateGraph(cast(Any, DocumentExtractionState))

        # Node pipeline
        builder.add_node("preprocess", self._node_preprocess)
        builder.add_node("inspect_and_classify", self._node_inspect_and_classify)
        builder.add_node("normalize_orientation", self._node_normalize_orientation)
        builder.add_node("extract_ocr", self._node_extract_ocr)
        builder.add_node("extract_markdown", self._node_extract_markdown)
        builder.add_node("summon_diagram_specialist", self._node_summon_diagram_specialist)
        builder.add_node("aggregate_and_judge", self._node_aggregate_and_judge)

        # Edges
        builder.add_edge(START, "preprocess")
        builder.add_edge("preprocess", "inspect_and_classify")
        builder.add_edge("inspect_and_classify", "normalize_orientation")
        builder.add_edge("normalize_orientation", "extract_ocr")
        builder.add_edge("extract_ocr", "extract_markdown")
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
        page_number: int | None = None,
        region_output_dir: str | Path | None = None,
        native_text: str | None = None,
        force_vlm_reading: bool = False,
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
            "page_number": page_number,
            "region_output_dir": str(region_output_dir) if region_output_dir else None,
            "native_text": native_text,
            "force_vlm_reading": force_vlm_reading,
        }

        logger.info("[Pipeline] Memulai ekstraksi: %s (is_first_page=%s)", image_path, is_first_page)
        final_state = cast(dict[str, Any], self.graph.invoke(initial_state))

        final_md = final_state.get("markdown_content", "")
        ocr_payload = OCRExtractionResult.model_validate(
            final_state.get(
                "ocr_result",
                {"status": "disabled", "model": self.settings.ocr_model},
            )
        )
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
            visual_count=max(
                count_visuals(final_md),
                sum(region.kind == "figure" for region in ocr_payload.regions),
            ),
            table_count=max(
                count_tables(final_md),
                sum(region.kind == "table" for region in ocr_payload.regions),
            ),
            document_title=final_state.get("document_title"),
            ocr_status=final_state.get("ocr_status", ocr_payload.status),
            ocr_model=ocr_payload.model,
            ocr_latency_ms=ocr_payload.latency_ms,
            ocr_regions=ocr_payload.regions,
            region_manifest_path=ocr_payload.manifest_path,
            ocr_quality_score=ocr_payload.quality_score,
            ocr_trust_level=ocr_payload.trust_level,
            ocr_risk_flags=ocr_payload.risk_flags,
            textreflow_applied=ocr_payload.textreflow_applied,
            textreflow_reason=ocr_payload.textreflow_reason,
            rotation_degrees=cast(
                Any,
                final_state.get("inspection_rotation_degrees", 0)
                if ocr_payload.decision == "blank_page"
                else ocr_payload.rotation_degrees,
            ),
            vlm_visual_rescue=bool(final_state.get("requires_vlm_reading", False)),
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
        """Tahap 3: VLM mengklasifikasikan layout dan kompleksitas halaman."""
        forced_specs = state.get("forced_specs")
        forced_doc_type = state.get("forced_doc_type")
        img = state.get("preprocessed_path") or state["image_path"]
        # Tetap inspeksi visual meski layout dipaksa: keputusan user menentukan
        # spesifikasi, sementara VLM menentukan orientasi dan apakah teks kecil
        # perlu dibaca ulang secara independen.
        t0 = time.perf_counter()
        is_first = bool(state.get("is_first_page", False))
        insp_res = self.extractor.inspect_page(img, is_first_page=is_first)
        elapsed = (time.perf_counter() - t0) * 1000

        forced_layout = forced_specs or forced_doc_type
        detected_specs = (
            normalize_specs(forced_layout)
            if forced_layout
            else insp_res.get("specs", ["plain"])
        )
        has_diag = bool(insp_res.get("has_diagram", False))
        diag_type = insp_res.get("diagram_type")
        has_tbl = bool(insp_res.get("has_table", False))
        difficulty = str(insp_res.get("difficulty", "standard"))
        doc_title = getattr(insp_res, "document_title", None) or insp_res.get("document_title")
        requires_vlm_reading = bool(
            insp_res.requires_vlm_reading or state.get("force_vlm_reading", False)
        ) and bool(self.settings.vlm_visual_rescue)

        if doc_title:
            logger.info("[Pipeline:Classify] Judul dokumen terdeteksi: '%s'", doc_title)
        if forced_layout:
            logger.info(
                "[Pipeline:Classify] Menggunakan layout paksa: %s; inspeksi visual tetap aktif.",
                detected_specs,
            )
        if requires_vlm_reading:
            logger.info(
                "[Pipeline:Classify] VLM visual rescue aktif: %s",
                insp_res.reasoning or "teks/layout perlu pembacaan presisi",
            )

        logger.info(
            "[Pipeline:Classify] Layout: %s | Diagram: %s (%s) | Tabel: %s | Difficulty: %s | VLM rescue: %s%s (%.1fms)",
            detected_specs,
            has_diag,
            diag_type,
            has_tbl,
            difficulty,
            requires_vlm_reading,
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
            "inspection_rotation_degrees": insp_res.rotation_degrees,
            "requires_vlm_reading": requires_vlm_reading,
        }

    def _node_normalize_orientation(
        self, state: DocumentExtractionState
    ) -> DocumentExtractionState:
        """Putar halaman berdasarkan inspeksi VLM sebelum dibaca oleh OCR."""
        rotation = int(state.get("inspection_rotation_degrees", 0))
        if rotation not in (90, 180, 270):
            return state
        img = state.get("preprocessed_path") or state["image_path"]
        oriented = rotate_image_right_angle(
            img,
            rotation,
        )
        logger.info("[Pipeline:Orientation] Halaman diputar %d derajat CW", rotation)
        return {**state, "preprocessed_path": oriented}

    def _node_extract_ocr(
        self, state: DocumentExtractionState
    ) -> DocumentExtractionState:
        """OCR primer dengan quality gate; hasil meragukan tidak langsung dipercaya."""
        if self.ocr_extractor is None:
            result = OCRExtractionResult(
                status="disabled",
                decision="disabled",
                model=self.settings.ocr_model,
                error="OCR_MODEL belum dikonfigurasi.",
            )
            logger.info("[Pipeline:OCR] OCR belum dikonfigurasi; siapkan fallback VLM utama")
        else:
            img = state.get("preprocessed_path") or state["image_path"]
            result = self.ocr_extractor.extract_robust(
                img,
                output_dir=state.get("region_output_dir"),
                native_text=state.get("native_text"),
            )

        inspection_rotation = int(state.get("inspection_rotation_degrees", 0))
        result.rotation_degrees = cast(
            Any, (inspection_rotation + result.rotation_degrees) % 360
        )
        if result.oriented_image_path:
            oriented_path = result.oriented_image_path
        else:
            oriented_path = state.get("preprocessed_path") or state["image_path"]
        ocr_has_figure = any(region.kind == "figure" for region in result.regions)
        ocr_has_table = any(region.kind == "table" for region in result.regions)
        return {
            **state,
            "preprocessed_path": oriented_path,
            "has_diagram": bool(state.get("has_diagram")) or ocr_has_figure,
            "has_table": bool(state.get("has_table")) or ocr_has_table,
            "ocr_result": result.model_dump(),
            "ocr_status": result.decision,
            "ocr_regions": [region.model_dump() for region in result.regions],
        }

    def _node_extract_markdown(
        self, state: DocumentExtractionState
    ) -> DocumentExtractionState:
        """Tahap 4: pakai draft OCR; fallback ke VLM utama bila OCR tidak tersedia/gagal."""
        t0 = time.perf_counter()
        img = state.get("preprocessed_path") or state["image_path"]
        specs = state.get("specs", ["plain"])
        prev_context = state.get("previous_page_context")
        ocr_result = OCRExtractionResult.model_validate(
            state.get("ocr_result", {"status": "disabled"})
        )

        if ocr_result.decision == "blank_page":
            md_text = ""
            ocr_status = "blank_page"
            source = "blank-page gate"
        elif (
            ocr_result.status == "success"
            and ocr_result.trust_level == "high"
            and ocr_result.markdown.strip()
            and not state.get("requires_vlm_reading", False)
        ):
            ocr_result.source_markdown = (
                ocr_result.source_markdown or ocr_result.markdown
            )
            page_width = 0
            try:
                from PIL import Image

                with Image.open(img) as source:
                    page_width = source.width
            except OSError:
                logger.warning("[Pipeline:TextReflow] Gagal membaca lebar gambar %s", img)
            reflowed, reflow_reason = reflow_regions(
                ocr_result.regions,
                page_width=page_width,
                specs=specs,
                enabled=self.settings.textreflow_enabled and page_width > 0,
            )
            md_text = reflowed or ocr_result.markdown
            ocr_result.textreflow_applied = reflowed is not None
            ocr_result.textreflow_reason = reflow_reason
            ocr_status = ocr_result.decision
            source = (
                f"OCR ({ocr_result.model}) + TextReflow"
                if reflowed is not None
                else f"OCR ({ocr_result.model})"
            )
        else:
            agent = get_agent(specs)
            md_text = agent.run(
                img,
                llm=self.vlm,
                previous_page_context=prev_context,
                native_text=state.get("native_text"),
            )
            if state.get("requires_vlm_reading", False):
                ocr_status = "vlm_visual_rescue"
                source = "VLM utama independen (visual rescue teks/layout sulit)"
            else:
                ocr_status = "fallback_vlm"
                source = "VLM utama independen (OCR tidak dipercaya/tersedia)"

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "[Pipeline:ExtractMarkdown] Draft dari %s selesai (%d karakter, %.1fms)",
            source,
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
            "ocr_status": ocr_status,
            "ocr_force_judge": ocr_status != "blank_page",
            "ocr_result": ocr_result.model_dump(),
        }

    def _node_summon_diagram_specialist(
        self, state: DocumentExtractionState
    ) -> DocumentExtractionState:
        """Tahap 5: kirim crop figure OCR ke spesialis Mermaid VLM utama."""
        has_diag = state.get("has_diagram", False)
        specs = state.get("specs", [])
        md_content = state.get("markdown_content", "")
        if state.get("ocr_status") == "blank_page":
            return state
        full_image = state.get("preprocessed_path") or state["image_path"]
        ocr_regions = [OCRRegion.model_validate(item) for item in state.get("ocr_regions", [])]
        figure_crops = [
            region.crop_path
            for region in ocr_regions
            if region.kind == "figure" and region.crop_path
        ]

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
        targets = figure_crops or [full_image]
        logger.info(
            "[Pipeline:SummonSpecialist] Memproses %d target diagram%s...",
            len(targets),
            " hasil crop OCR" if figure_crops else " dari halaman penuh",
        )
        diag_hint = state.get("diagram_type")

        from .diagram import extract_diagram_to_mermaid

        mermaid_codes: list[str] = []
        diagram_summaries: list[str] = []
        for target in targets:
            diag_result = extract_diagram_to_mermaid(
                image_path=target,
                llm=self.vlm,
                forced_diagram_type=diag_hint,
            )
            if diag_result.mermaid_code:
                mermaid_codes.append(diag_result.mermaid_code)
            if diag_result.text_summary:
                diagram_summaries.append(diag_result.text_summary)
        elapsed = (time.perf_counter() - t0) * 1000

        if mermaid_codes:
            logger.info(
                "[Pipeline:SummonSpecialist] Berhasil mengekstrak %d Diagram Mermaid (%.1fms)",
                len(mermaid_codes),
                elapsed,
            )
        else:
            logger.info(
                "[Pipeline:SummonSpecialist] Diagram tidak cocok ke Mermaid, fallback ke deskripsi (%.1fms)",
                elapsed,
            )

        return {
            **state,
            "diagram_mermaid_code": mermaid_codes[0] if mermaid_codes else None,
            "diagram_summary": diagram_summaries[0] if diagram_summaries else None,
            "diagram_mermaid_codes": mermaid_codes,
            "diagram_summaries": diagram_summaries,
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
        if state.get("ocr_status") == "blank_page":
            return {
                **state,
                "difficulty": "simple",
                "markdown_content": "",
                "final_markdown": "",
            }
        mermaid_codes = state.get("diagram_mermaid_codes", [])
        if not mermaid_codes and state.get("diagram_mermaid_code"):
            mermaid_codes = [cast(str, state["diagram_mermaid_code"])]
        diagram_summaries = state.get("diagram_summaries", [])
        specs = state.get("specs", ["plain"])

        # 1. Satukan blok diagram Mermaid ke Markdown jika belum ada
        combined_md = md_text
        for index, mermaid_code in enumerate(mermaid_codes):
            from .diagram import sanitize_mermaid_code, validate_mermaid_syntax

            # Ganti draft yang rusak dengan hasil spesialis yang sudah valid.
            specialist = sanitize_mermaid_code(mermaid_code)
            if specialist and validate_mermaid_syntax(specialist)[0]:
                replaced = False

                def recover_invalid(
                    m: re.Match, replacement: str = specialist
                ) -> str:
                    nonlocal replaced
                    draft_code = sanitize_mermaid_code(m.group(1))
                    if not replaced and (not draft_code or not validate_mermaid_syntax(draft_code)[0]):
                        replaced = True
                        return f"```mermaid\n{replacement}\n```"
                    return m.group(0)

                combined_md = re.sub(
                    r"```mermaid\s*([\s\S]*?)\s*```",
                    recover_invalid,
                    combined_md,
                    count=1,
                    flags=re.IGNORECASE,
                )
            fenced = f"```mermaid\n{specialist or mermaid_code}\n```"
            if fenced not in combined_md:
                mermaid_block = f"\n\n{fenced}"
                if index < len(diagram_summaries):
                    mermaid_block += (
                        f"\n\n> **[Diagram Summary]:** {diagram_summaries[index]}"
                    )
                combined_md += mermaid_block

        # 2. Fast-path: skip judge untuk halaman simple yang bersih
        difficulty = state.get("difficulty") or _assess_difficulty_from_output(md_text)
        has_diagram = state.get("has_diagram", False) or _has_diagram_indicators(md_text)

        if (
            not self.thorough
            and not state.get("ocr_force_judge", False)
            and _should_skip_judge(combined_md, difficulty, has_diagram)
        ):
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
            previous_page_context=state.get("previous_page_context"),
            native_text=state.get("native_text"),
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

        final_status = state.get("ocr_status", "disabled")
        if (
            final_status in {"accepted", "retried_rotated"}
            and final_md.strip() != combined_md.strip()
        ):
            final_status = "corrected_by_vlm"

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
            "ocr_status": final_status,
        }


VisionRAGPipeline = DocumentExtractionPipeline

__all__ = [
    "DocumentExtractionPipeline",
    "DocumentExtractionState",
    "VisionRAGPipeline",
]
