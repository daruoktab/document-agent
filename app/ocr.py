"""Adapter Unlimited-OCR dengan grounding, quality gate, dan recovery rotasi."""

from __future__ import annotations

import json
import logging
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from PIL import Image

from .llm import image_data_uri
from .preprocess import rotate_image_right_angle
from .schemas import OCRExtractionResult, OCRQualityAssessment, OCRRegion

logger = logging.getLogger("app.ocr")

GROUNDING_RE = re.compile(
    r"<\|det\|>\s*(?P<label>[^\[\r\n]*?)\s*"
    r"\[\s*(?P<x1>-?\d+(?:\.\d+)?)\s*,\s*(?P<y1>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<x2>-?\d+(?:\.\d+)?)\s*,\s*(?P<y2>-?\d+(?:\.\d+)?)\s*\]"
    r"\s*<\|/det\|>",
    re.IGNORECASE,
)


def _response_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return str(content or "")


def _region_kind(
    label: str, text: str = ""
) -> Literal["text", "table", "figure", "unknown"]:
    normalized = label.strip().lower()
    normalized_text = text.strip().lower()
    if any(word in normalized for word in ("table", "tabel")) or (
        normalized_text.count("|") >= 4
        or "<table" in normalized_text
        or "</table>" in normalized_text
    ):
        return "table"
    if any(
        word in normalized
        for word in ("figure", "image", "diagram", "chart", "visual", "gambar")
    ):
        return "figure"
    if any(
        marker in normalized_text
        for marker in ("[diagram/visual]", "[gambar/visual]", "```mermaid")
    ):
        return "figure"
    if any(word in normalized for word in ("text", "title", "heading", "paragraph")):
        return "text"
    return "unknown"


def _clean_markdown(raw: str) -> str:
    cleaned = GROUNDING_RE.sub("", raw).strip()
    if cleaned.startswith("```markdown") and cleaned.endswith("```"):
        cleaned = cleaned[len("```markdown") : -3].strip()
    elif cleaned.startswith("```md") and cleaned.endswith("```"):
        cleaned = cleaned[len("```md") : -3].strip()
    return cleaned


def _normalized_text(value: str) -> str:
    return " ".join(re.findall(r"\w+", value.lower(), flags=re.UNICODE))


def _image_signals(image_path: str | Path) -> tuple[float, float, float]:
    """Hitung kepadatan tinta dan konsentrasi proyeksi baris/kolom secara murah."""
    with Image.open(image_path) as source:
        image = source.convert("L")
        image.thumbnail((800, 800))
        width, height = image.size
        pixels = list(image.tobytes())

    if not pixels or width <= 0 or height <= 0:
        return 0.0, 0.0, 0.0
    row_counts = [0] * height
    col_counts = [0] * width
    ink_count = 0
    for offset, pixel in enumerate(pixels):
        if pixel < 220:
            y, x = divmod(offset, width)
            row_counts[y] += 1
            col_counts[x] += 1
            ink_count += 1
    if ink_count == 0:
        return 0.0, 0.0, 0.0
    denominator = float(ink_count * ink_count)
    horizontal_energy = sum(value * value for value in row_counts) / denominator
    vertical_energy = sum(value * value for value in col_counts) / denominator
    return ink_count / len(pixels), horizontal_energy, vertical_energy


def assess_ocr_quality(
    *,
    image_path: str | Path,
    markdown: str,
    raw_response: str,
    regions: list[OCRRegion],
    native_text: str | None = None,
    min_trust_score: float = 0.72,
    medium_trust_score: float = 0.48,
    blank_ink_ratio: float = 0.0002,
    sparse_ink_ratio: float = 0.015,
) -> OCRQualityAssessment:
    """Nilai OCR dari bentuk output, bukti grounding, citra, dan text-layer PDF."""
    ink_ratio, horizontal_energy, vertical_energy = _image_signals(image_path)
    is_blank = ink_ratio <= blank_ink_ratio
    is_sparse = not is_blank and ink_ratio <= sparse_ink_ratio
    right_angle_suspected = (
        not is_blank
        and vertical_energy > max(horizontal_energy * 1.35, horizontal_energy + 0.001)
    )
    risk_flags: list[str] = []
    score = 0.46

    visible = markdown.strip()
    if len(visible) >= 12:
        score += 0.08
    elif len(visible) < 4:
        score -= 0.30
        risk_flags.append("too_short")

    printable_ratio = (
        sum(char.isprintable() or char in "\n\t" for char in visible) / len(visible)
        if visible
        else 0.0
    )
    if printable_ratio >= 0.98:
        score += 0.08
    elif printable_ratio < 0.90:
        score -= 0.20
        risk_flags.append("non_printable_output")

    alnum_ratio = (
        sum(char.isalnum() for char in visible) / len(visible) if visible else 0.0
    )
    if alnum_ratio >= 0.35:
        score += 0.08
    elif alnum_ratio < 0.15:
        score -= 0.18
        risk_flags.append("low_alphanumeric_ratio")

    if regions:
        score += 0.15
    else:
        score -= 0.10
        risk_flags.append("missing_grounding")

    lines = [line.strip() for line in visible.splitlines() if line.strip()]
    if len(lines) >= 4 and len(set(lines)) / len(lines) < 0.5:
        score -= 0.22
        risk_flags.append("repetitive_output")
    if "<|det|>" in visible or "<|/det|>" in visible:
        score -= 0.20
        risk_flags.append("malformed_grounding")

    if is_blank:
        risk_flags.append("blank_image")
    elif is_sparse:
        risk_flags.append("sparse_page")
        if len(visible) > max(160, int(ink_ratio * 100_000)):
            score -= 0.22
            risk_flags.append("sparse_output_mismatch")
    if right_angle_suspected:
        risk_flags.append("right_angle_suspected")

    similarity: float | None = None
    native_normalized = _normalized_text(native_text or "")
    ocr_normalized = _normalized_text(visible)
    if len(native_normalized) >= 12:
        similarity = SequenceMatcher(None, native_normalized, ocr_normalized).ratio()
        if similarity >= 0.70:
            score += 0.18
        elif similarity < 0.30:
            score -= 0.30
            risk_flags.append("native_text_mismatch")

    score = min(1.0, max(0.0, score))
    if score >= min_trust_score:
        trust_level: Literal["high", "medium", "low"] = "high"
    elif score >= medium_trust_score:
        trust_level = "medium"
    else:
        trust_level = "low"
    return OCRQualityAssessment(
        score=score,
        trust_level=trust_level,
        risk_flags=list(dict.fromkeys(risk_flags)),
        ink_ratio=ink_ratio,
        horizontal_energy=horizontal_energy,
        vertical_energy=vertical_energy,
        native_text_similarity=similarity,
        is_blank=is_blank,
        is_sparse=is_sparse,
        right_angle_suspected=right_angle_suspected,
    )


def parse_grounding_regions(
    raw: str,
    *,
    image_size: tuple[int, int],
    coordinate_size: int = 1024,
) -> list[OCRRegion]:
    """Parse token grounding dan petakan koordinat model ke piksel gambar."""
    matches = list(GROUNDING_RE.finditer(raw))
    if not matches:
        return []

    width, height = image_size
    coordinate_size = max(1, coordinate_size)
    regions: list[OCRRegion] = []
    for offset, match in enumerate(matches):
        index = offset + 1
        next_start = matches[offset + 1].start() if offset + 1 < len(matches) else len(raw)
        region_text = raw[match.end() : next_start].strip()
        coords = tuple(float(match.group(key)) for key in ("x1", "y1", "x2", "y2"))
        x1, y1, x2, y2 = coords
        px = (
            round(x1 / coordinate_size * width),
            round(y1 / coordinate_size * height),
            round(x2 / coordinate_size * width),
            round(y2 / coordinate_size * height),
        )
        left = min(max(px[0], 0), width)
        top = min(max(px[1], 0), height)
        right = min(max(px[2], 0), width)
        bottom = min(max(px[3], 0), height)
        if right <= left or bottom <= top:
            logger.warning("[OCR] Abaikan bbox invalid #%d: %s", index, coords)
            continue
        label = match.group("label").strip() or "unknown"
        regions.append(
            OCRRegion(
                index=index,
                label=label,
                kind=_region_kind(label, region_text),
                text=region_text,
                bbox_model=coords,
                bbox_pixels=(left, top, right, bottom),
            )
        )
    return regions


class UnlimitedOCRExtractor:
    """Panggil OCR, ukur kualitasnya, dan coba orientasi alternatif bila perlu."""

    def __init__(
        self,
        llm: BaseChatModel,
        *,
        model_name: str,
        prompt: str,
        coordinate_size: int = 1024,
        crop_padding: float = 0.01,
        min_trust_score: float = 0.72,
        medium_trust_score: float = 0.48,
        rotation_retry: bool = True,
        blank_ink_ratio: float = 0.0002,
        sparse_ink_ratio: float = 0.015,
    ) -> None:
        self.llm = llm
        self.model_name = model_name
        self.prompt = prompt
        self.coordinate_size = max(1, coordinate_size)
        self.crop_padding = max(0.0, crop_padding)
        self.min_trust_score = min(1.0, max(0.0, min_trust_score))
        self.medium_trust_score = min(
            self.min_trust_score, max(0.0, medium_trust_score)
        )
        self.rotation_retry = rotation_retry
        self.blank_ink_ratio = max(0.0, blank_ink_ratio)
        self.sparse_ink_ratio = max(self.blank_ink_ratio, sparse_ink_ratio)

    def extract(
        self,
        image_path: str | Path,
        *,
        output_dir: str | Path | None = None,
        native_text: str | None = None,
    ) -> OCRExtractionResult:
        path = Path(image_path)
        started = time.perf_counter()
        try:
            with Image.open(path) as source:
                image_size = source.size
            content: list[dict[str, Any]] = [
                {"type": "text", "text": self.prompt},
                {"type": "image_url", "image_url": {"url": image_data_uri(path)}},
            ]
            response = self.llm.invoke([HumanMessage(content=cast(Any, content))])
            raw = _response_text(response.content).strip()
            if not raw:
                raise ValueError("Model OCR mengembalikan respons kosong.")

            regions = parse_grounding_regions(
                raw, image_size=image_size, coordinate_size=self.coordinate_size
            )
            markdown = _clean_markdown(raw)
            if not markdown:
                raise ValueError("Markdown OCR kosong setelah token grounding dibersihkan.")
            quality = assess_ocr_quality(
                image_path=path,
                markdown=markdown,
                raw_response=raw,
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
                raw_response=raw,
                model=self.model_name,
                latency_ms=(time.perf_counter() - started) * 1000,
                regions=regions,
                decision="accepted" if quality.trust_level == "high" else "low_trust",
                quality_score=quality.score,
                trust_level=quality.trust_level,
                risk_flags=quality.risk_flags,
                oriented_image_path=str(path.resolve()),
                native_text_similarity=quality.native_text_similarity,
                candidate_scores={"0": quality.score},
            )
            if output_dir is not None:
                self._persist_regions(path, result, Path(output_dir))
            logger.info(
                "[OCR] %s: %d karakter, %d region, trust=%s %.2f, %.1fms",
                result.decision,
                len(result.markdown),
                len(result.regions),
                result.trust_level,
                result.quality_score,
                result.latency_ms,
            )
            return result
        except Exception as exc:  # noqa: BLE001
            logger.warning("[OCR] Gagal, pipeline akan fallback ke VLM utama: %s", exc)
            return OCRExtractionResult(
                status="error",
                decision="error",
                model=self.model_name,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=str(exc),
            )

    def extract_robust(
        self,
        image_path: str | Path,
        *,
        output_dir: str | Path | None = None,
        native_text: str | None = None,
    ) -> OCRExtractionResult:
        """OCR dengan blank gate dan pemilihan kandidat rotasi berdasarkan kualitas."""
        path = Path(image_path)
        ink_ratio, _, _ = _image_signals(path)
        if ink_ratio <= self.blank_ink_ratio and not _normalized_text(native_text or ""):
            blank_result = OCRExtractionResult(
                status="success",
                decision="blank_page",
                model=self.model_name,
                quality_score=1.0,
                trust_level="high",
                risk_flags=["blank_page"],
                oriented_image_path=str(path.resolve()),
                candidate_scores={"0": 1.0},
            )
            if output_dir is not None:
                self._persist_regions(path, blank_result, Path(output_dir))
            return blank_result

        started = time.perf_counter()
        candidates: list[tuple[int, Path, OCRExtractionResult]] = []
        first = self.extract(path, native_text=native_text)
        candidates.append((0, path, first))

        should_retry = self.rotation_retry and (
            first.status != "success"
            or first.trust_level != "high"
            or "right_angle_suspected" in first.risk_flags
        )
        if should_retry:
            for degrees in (90, 270, 180):
                rotated = Path(rotate_image_right_angle(path, degrees))
                candidate = self.extract(rotated, native_text=native_text)
                candidates.append((degrees, rotated, candidate))
                if candidate.status == "success" and candidate.trust_level == "high":
                    break

        best_rotation, best_path, best = max(
            candidates,
            key=lambda item: (
                item[2].quality_score if item[2].status == "success" else -1.0,
                -int(item[0] != 0),
            ),
        )
        scores = {str(degrees): result.quality_score for degrees, _, result in candidates}
        for degrees, candidate_path, _ in candidates:
            if degrees and candidate_path != best_path:
                candidate_path.unlink(missing_ok=True)
        best.rotation_degrees = cast(Any, best_rotation)
        best.candidate_scores = scores
        best.latency_ms = (time.perf_counter() - started) * 1000

        if best.status == "success":
            if best.trust_level == "high":
                best.decision = "retried_rotated" if best_rotation else "accepted"
            else:
                best.decision = "low_trust"
            if best_rotation and output_dir:
                temporary_best_path = best_path
                best_path = Path(
                    rotate_image_right_angle(path, best_rotation, output_dir=output_dir)
                )
                temporary_best_path.unlink(missing_ok=True)
            best.oriented_image_path = str(best_path.resolve())
            if output_dir is not None:
                self._persist_regions(best_path, best, Path(output_dir))

        logger.info(
            "[OCR:QualityGate] decision=%s rotation=%d trust=%s score=%.2f risks=%s",
            best.decision,
            best_rotation,
            best.trust_level,
            best.quality_score,
            best.risk_flags,
        )
        return best

    def _persist_regions(
        self,
        image_path: Path,
        result: OCRExtractionResult,
        output_dir: Path,
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


__all__ = [
    "GROUNDING_RE",
    "UnlimitedOCRExtractor",
    "assess_ocr_quality",
    "parse_grounding_regions",
]
