"""
Vision VLM & Document Text Extractor Package.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "AGENT_REGISTRY",
    "DocumentExtractionAgent",
    "DocumentExtractionPipeline",
    "DocumentExtractionState",
    "Settings",
    "TabularDatabaseManager",
    "VisionExtractor",
    "batch_extract_documents",
    "build_deep_agent",
    "classify_diagram_convertibility",
    "cross_verify_dual_track",
    "extract_and_ingest_tables_from_markdown",
    "extract_diagram_to_mermaid",
    "extract_pdf_markdown_mupdf",
    "format_page_delimiter",
    "get_agent",
    "get_settings",
    "pdf_page_count",
    "pdf_to_images",
    "preprocess_image",
    "preview_markdown_chunks",
    "process_presentation",
    "process_presentation_vision",
    "query_sqlite",
    "run_deep_reasoning_agent",
    "sanitize_mermaid_code",
    "scan_document_directories",
    "split_markdown_by_pages",
    "stitch_pages_to_markdown",
]

_MODULE_MAP: dict[str, str] = {
    "AGENT_REGISTRY": ".agents",
    "DocumentExtractionAgent": ".agents",
    "get_agent": ".agents",
    "batch_extract_documents": ".batch",
    "scan_document_directories": ".batch",
    "Settings": ".config",
    "get_settings": ".config",
    "build_deep_agent": ".deep_agent",
    "run_deep_reasoning_agent": ".deep_agent",
    "classify_diagram_convertibility": ".diagram",
    "extract_diagram_to_mermaid": ".diagram",
    "sanitize_mermaid_code": ".diagram",
    "VisionExtractor": ".extractor",
    "DocumentExtractionPipeline": ".graph",
    "DocumentExtractionState": ".graph",
    "format_page_delimiter": ".multi_page",
    "preview_markdown_chunks": ".multi_page",
    "split_markdown_by_pages": ".multi_page",
    "stitch_pages_to_markdown": ".multi_page",
    "extract_pdf_markdown_mupdf": ".pdf",
    "pdf_page_count": ".pdf",
    "pdf_to_images": ".pdf",
    "process_presentation": ".ppt",
    "process_presentation_vision": ".ppt",
    "preprocess_image": ".preprocess",
    "TabularDatabaseManager": ".tabular_db",
    "cross_verify_dual_track": ".tabular_db",
    "extract_and_ingest_tables_from_markdown": ".tabular_db",
    "query_sqlite": ".tabular_db",
}


def __getattr__(name: str) -> Any:
    if name in _MODULE_MAP:
        mod = importlib.import_module(_MODULE_MAP[name], __name__)
        val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
