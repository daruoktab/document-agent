"""Adapter PaddleOCR-VL untuk layout lokal dan recognizer llama.cpp."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from PIL import Image

from .ocr import _image_signals, assess_ocr_quality
from .preprocess import rotate_image_right_angle
from .schemas import OCRExtractionResult, OCRRegion

logger = logging.getLogger("app.paddle_ocr")


def _plain_mapping(value: Any) -> dict[str, Any]:
    """Ambil mapping serializable dari result object PaddleX."""
    if callable(value):
        value = value()
    if not isinstance(value, dict):
        return {}
    nested = value.get("res")
    return nested if isinstance(nested, dict) else value


def _bbox(value: Any, *, width: int, height: int) -> tuple[int, int, int, int]:
    """Normalisasi bbox Paddle ke koordinat piksel yang valid."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"bbox Paddle tidak valid: {value!r}")
    x1, y1, x2, y2 = (round(float(item)) for item in value)
    left, right = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
    top, bottom = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
    return left, top, right, bottom


def paddle_region_kind(label: str) -> str:
    """Petakan label layout Paddle ke tipe region internal."""
    normalized = label.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"doc_title", "document_title", "title"}:
        return "title"
    if normalized in {
        "paragraph_title",
        "section_title",
        "section_header",
        "heading",
    }:
        return "section"
    if normalized in {"author", "authors"}:
        return "authors"
    if normalized in {"list", "list_item"}:
        return "list"
    if normalized in {
        "caption",
        "figure_title",
        "image_caption",
        "chart_title",
        "table_title",
    }:
        return "caption"
    if normalized in {"footnote", "vision_footnote"}:
        return "footnote"
    if "formula" in normalized:
        return "formula"
    if "table" in normalized:
        return "table"
    if normalized in {"image", "figure", "chart", "seal"}:
        return "figure"
    if normalized in {"header", "page_header"}:
        return "header"
    if normalized in {"footer", "page_footer", "page_number"}:
        return "footer"
    if normalized in {
        "text",
        "paragraph",
        "content",
        "abstract",
        "aside_text",
        "reference",
    }:
        return "text"
    return "unknown"


def parse_paddle_regions(
    payload: dict[str, Any], *, image_size: tuple[int, int]
) -> list[OCRRegion]:
    """Konversi ``parsing_res_list`` Paddle menjadi region internal."""
    width, height = image_size
    raw_regions = payload.get("parsing_res_list", [])
    if not isinstance(raw_regions, list):
        return []

    regions: list[OCRRegion] = []
    for position, item in enumerate(raw_regions, start=1):
        if not isinstance(item, dict):
            continue
        try:
            bbox = _bbox(item.get("block_bbox"), width=width, height=height)
        except (TypeError, ValueError):
            continue
        label = str(item.get("block_label") or "unknown")
        text = str(item.get("block_content") or "").strip()
        raw_order = item.get("block_order")
        reading_order = raw_order if isinstance(raw_order, int) else None
        regions.append(
            OCRRegion(
                index=position,
                label=label,
                kind=cast(Any, paddle_region_kind(label)),
                text=text,
                lines=[line for line in text.splitlines() if line.strip()],
                reading_order=reading_order,
                bbox_model=tuple(float(value) for value in bbox),
                bbox_pixels=bbox,
            )
        )
    return regions


class PaddleOCRVLExtractor:
    """Paddle layout pipeline dengan recognizer GGUF pada llama.cpp."""

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        api_key: str = "not-needed",
        timeout: float = 300,
        max_tokens: int = 8192,
        crop_padding: float = 0.01,
        min_trust_score: float = 0.72,
        medium_trust_score: float = 0.48,
        rotation_retry: bool = True,
        blank_ink_ratio: float = 0.0002,
        sparse_ink_ratio: float = 0.015,
        pipeline: Any | None = None,
        pipeline_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.crop_padding = max(0.0, crop_padding)
        self.min_trust_score = min(1.0, max(0.0, min_trust_score))
        self.medium_trust_score = min(
            self.min_trust_score, max(0.0, medium_trust_score)
        )
        self.rotation_retry = rotation_retry
        self.blank_ink_ratio = max(0.0, blank_ink_ratio)
        self.sparse_ink_ratio = max(self.blank_ink_ratio, sparse_ink_ratio)
        self._pipeline = pipeline
        self._pipeline_factory = pipeline_factory

    def _get_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        factory = self._pipeline_factory
        if factory is None:
            from paddleocr import PaddleOCRVL

            factory = PaddleOCRVL
        self._pipeline = factory(
            pipeline_version="v1.6",
            device="gpu:0",
            vl_rec_backend="llama-cpp-server",
            vl_rec_server_url=self.base_url,
            vl_rec_api_model_name=self.model_name,
            vl_rec_api_key=self.api_key,
            vl_rec_max_concurrency=1,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_layout_detection=True,
            format_block_content=False,
            use_queues=False,
        )
        return self._pipeline

    def extract(
        self,
        image_path: str | Path,
        *,
        output_dir: str | Path | None = None,
        native_text: str | None = None,
    ) -> OCRExtractionResult:
        started = time.perf_counter()
        path = Path(image_path)
        try:
            with Image.open(path) as source:
                image_size = source.size
            results = self._get_pipeline().predict(
                str(path),
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_layout_detection=True,
                format_block_content=False,
                temperature=0.0,
                max_new_tokens=self.max_tokens,
            )
            if not results:
                raise ValueError("PaddleOCR-VL tidak mengembalikan hasil.")
            paddle_result = results[0]
            payload = _plain_mapping(getattr(paddle_result, "json", paddle_result))
            markdown_payload = _plain_mapping(getattr(paddle_result, "markdown", {}))
            markdown = str(
                markdown_payload.get("markdown_texts")
                or markdown_payload.get("text")
                or payload.get("markdown_texts")
                or ""
            ).strip()
            if not markdown:
                raise ValueError("Markdown PaddleOCR-VL kosong.")
            regions = parse_paddle_regions(payload, image_size=image_size)
            raw_response = json.dumps(payload, ensure_ascii=False, default=str)
            quality = assess_ocr_quality(
                image_path=path,
                markdown=markdown,
                raw_response=raw_response,
                regions=regions,
                native_text=native_text,
                min_trust_score=self.min_trust_score,
                medium_trust_score=self.medium_trust_score,
                blank_ink_ratio=self.blank_ink_ratio,
                sparse_ink_ratio=self.sparse_ink_ratio,
            )
            result = OCRExtractionResult(
                status="success",
                markdown=markdown,
                source_markdown=markdown,
                raw_response=raw_response,
                model=self.model_name,
                latency_ms=(time.perf_counter() - started) * 1000,
                regions=regions,
                decision="accepted" if quality.trust_level == "high" else "low_trust",
                quality_score=quality.score,
                trust_level=quality.trust_level,
                risk_flags=quality.risk_flags,
                native_text_similarity=quality.native_text_similarity,
                oriented_image_path=str(path.resolve()),
            )
            if output_dir is not None:
                self._persist(path, result, Path(output_dir))
            return result
        except Exception as exc:
            logger.exception("[PaddleOCR-VL] Ekstraksi gagal untuk %s", path)
            return OCRExtractionResult(
                status="error",
                decision="error",
                model=self.model_name,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=str(exc),
                oriented_image_path=str(path.resolve()),
            )

    def extract_robust(
        self,
        image_path: str | Path,
        *,
        output_dir: str | Path | None = None,
        native_text: str | None = None,
    ) -> OCRExtractionResult:
        started = time.perf_counter()
        path = Path(image_path)
        ink_ratio, _, _ = _image_signals(path)
        if ink_ratio <= self.blank_ink_ratio and not (native_text or "").strip():
            result = OCRExtractionResult(
                status="success",
                decision="blank_page",
                model=self.model_name,
                trust_level="high",
                quality_score=1.0,
                oriented_image_path=str(path.resolve()),
            )
            if output_dir is not None:
                self._persist(path, result, Path(output_dir))
            return result

        first = self.extract(path, native_text=native_text)
        candidates: list[tuple[int, Path, OCRExtractionResult]] = [(0, path, first)]
        if self.rotation_retry and first.trust_level != "high":
            for degrees in (90, 180, 270):
                rotated = Path(rotate_image_right_angle(path, degrees))
                candidate = self.extract(rotated, native_text=native_text)
                candidates.append((degrees, rotated, candidate))
                if candidate.status == "success" and candidate.trust_level == "high":
                    break

        rotation, best_path, best = max(
            candidates,
            key=lambda item: (
                item[2].quality_score,
                len(item[2].markdown),
                -item[0],
            ),
        )
        for degrees, candidate_path, _ in candidates:
            if degrees and candidate_path != best_path:
                candidate_path.unlink(missing_ok=True)
        best.rotation_degrees = cast(Any, rotation)
        best.candidate_scores = {
            str(degrees): result.quality_score for degrees, _, result in candidates
        }
        best.latency_ms = (time.perf_counter() - started) * 1000
        if best.status == "success" and best.decision != "blank_page":
            best.decision = (
                "retried_rotated"
                if best.trust_level == "high" and rotation
                else "accepted"
                if best.trust_level == "high"
                else "low_trust"
            )
            if rotation and output_dir is not None:
                temporary_best_path = best_path
                best_path = Path(
                    rotate_image_right_angle(path, rotation, output_dir=output_dir)
                )
                temporary_best_path.unlink(missing_ok=True)
            best.oriented_image_path = str(best_path.resolve())
        if output_dir is not None:
            self._persist(best_path, best, Path(output_dir))
        return best

    def _persist(
        self, image_path: Path, result: OCRExtractionResult, output_dir: Path
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            pad_x = round(image.width * self.crop_padding)
            pad_y = round(image.height * self.crop_padding)
            for region in result.regions:
                if region.kind not in {"table", "figure"}:
                    continue
                left, top, right, bottom = region.bbox_pixels
                crop_box = (
                    max(0, left - pad_x),
                    max(0, top - pad_y),
                    min(image.width, right + pad_x),
                    min(image.height, bottom + pad_y),
                )
                if crop_box[2] - crop_box[0] < 8 or crop_box[3] - crop_box[1] < 8:
                    continue
                crop_path = output_dir / f"region_{region.index:03d}_{region.kind}.png"
                image.crop(crop_box).save(crop_path, format="PNG")
                region.crop_path = str(crop_path.resolve())

        manifest_path = output_dir / "manifest.json"
        result.manifest_path = str(manifest_path.resolve())
        manifest_path.write_text(
            json.dumps(result.model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


__all__ = ["PaddleOCRVLExtractor", "paddle_region_kind", "parse_paddle_regions"]
