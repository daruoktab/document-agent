"""Integrasi aman TextReflow untuk halaman teks satu atau dua kolom."""

from __future__ import annotations

import re
from collections.abc import Sequence

from textreflow import assemble, validate_layout

from .schemas import OCRRegion

SUPPORTED_SPECS = {"plain", "markdown_hierarchy", "bilingual_journal"}
EXCLUDED_SPECS = {"presentation_slides", "chat_transcript", "signature_form"}
BLOCKED_KINDS = {"table", "figure", "formula", "unknown"}

_TEXTREFLOW_LABELS = {
    "title": "Title",
    "section": "Section-header",
    "authors": "Authors",
    "text": "Text",
    "list": "List-item",
    "caption": "Caption",
    "footnote": "Footnote",
    "header": "Page-header",
    "footer": "Page-footer",
}


def _render_markdown(blocks: list[dict[str, object]]) -> str:
    parts: list[str] = []
    for block in blocks:
        kind = str(block.get("type", "paragraph"))
        text = str(block.get("text") or "").strip()
        if not text:
            continue
        if kind == "title":
            parts.append(f"# {text.lstrip('# ').strip()}")
        elif kind == "section":
            parts.append(f"## {text.lstrip('# ').strip()}")
        elif kind == "list":
            parts.append(
                "\n".join(
                    line if re.match(r"^\s*[-*+]\s+", line) else f"- {line}"
                    for line in text.splitlines()
                    if line.strip()
                )
            )
        elif kind == "caption":
            parts.append(f"*{text}*")
        elif kind == "footnote":
            parts.append(f"> {text}")
        else:
            parts.append(text)
    return "\n\n".join(parts).strip()


def _content_size(value: str) -> int:
    return sum(char.isalnum() for char in value)


def reflow_regions(
    regions: Sequence[OCRRegion],
    *,
    page_width: int,
    specs: Sequence[str],
    enabled: bool,
) -> tuple[str | None, str]:
    """Kembalikan Markdown hasil reflow atau alasan fallback deterministik."""
    if not enabled:
        return None, "disabled"
    active_specs = set(specs)
    if active_specs & EXCLUDED_SPECS or not active_specs & SUPPORTED_SPECS:
        return None, "ineligible_spec"
    if not regions:
        return None, "missing_regions"
    if any(region.kind in BLOCKED_KINDS for region in regions):
        return None, "non_text_region"

    layout_regions: list[dict[str, object]] = []
    source_text: list[str] = []
    for region in regions:
        label = _TEXTREFLOW_LABELS.get(region.kind)
        if label is None:
            return None, f"unsupported_region:{region.kind}"
        lines = region.lines or [line for line in region.text.splitlines() if line.strip()]
        layout_regions.append(
            {
                "label": label,
                "bbox": list(region.bbox_pixels),
                "lines": lines,
            }
        )
        if region.kind not in {"header", "footer"}:
            source_text.extend(lines)

    layout = {
        "pages": [
            {
                "page": 0,
                "width": page_width,
                "regions": layout_regions,
            }
        ]
    }
    problems = validate_layout(layout, strict=False)
    if problems:
        return None, f"invalid_layout:{problems[0]}"
    blocks, _ = assemble(layout)
    markdown = _render_markdown(blocks)
    source_size = _content_size("\n".join(source_text))
    output_size = _content_size(markdown)
    if not markdown or source_size == 0:
        return None, "empty_output"
    ratio = output_size / source_size
    if ratio < 0.85 or ratio > 1.15:
        return None, "content_guard"
    return markdown, "applied"


__all__ = ["reflow_regions"]
