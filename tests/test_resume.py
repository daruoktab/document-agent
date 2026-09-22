"""Regression tests for page/slide checkpoint recovery."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.pdf import process_multipage_pdf
from app.ppt import process_presentation_vision
from main import build_parser


def _write_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _pdf_checkpoint_payload(pdf_path: Path) -> dict[str, object]:
    return {
        "version": 1,
        "source_file": str(pdf_path),
        "total_pages": 2,
        "document_title": "Recovered document",
        "pages": {
            "1": {
                "markdown": "# Page 1",
                "specs": ["plain"],
                "visual_count": 0,
                "table_count": 0,
                "tabular_event": None,
            }
        },
    }


class _Pipeline:
    def __init__(self) -> None:
        self.pages: list[int] = []

    def run(self, _image_path: str, *, page_number: int | None = None, **_kwargs: object) -> dict[str, object]:
        assert page_number is not None
        self.pages.append(page_number)
        return {
            "markdown_content": f"# Page {page_number}",
            "specs": ["plain"],
            "visual_count": 0,
            "table_count": 0,
            "document_title": "Recovered document" if page_number == 1 else None,
        }


class TestCheckpointResume(unittest.TestCase):
    def test_cli_exposes_resume_switch(self) -> None:
        args = build_parser().parse_args(["document.pdf", "--resume"])
        self.assertTrue(args.resume)

    def test_pdf_resume_only_processes_missing_pages_and_removes_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_path = root / "document.pdf"
            pdf_path.write_bytes(b"placeholder")
            output_path = root / "document.md"
            checkpoint_path = root / "logs" / "document_checkpoint.json"
            _write_checkpoint(checkpoint_path, _pdf_checkpoint_payload(pdf_path))
            page_images = [root / "page_0001.png", root / "page_0002.png"]
            pipeline = _Pipeline()

            with (
                patch("app.pdf.pdf_page_count", return_value=2),
                patch("app.pdf.extract_pdf_native_text_by_page", return_value={}),
                patch(
                    "app.pdf.pdf_to_images",
                    side_effect=lambda *_args, pages, **_kwargs: [
                        page_images[index] for index in pages
                    ],
                ) as render_pages,
            ):
                result = process_multipage_pdf(
                    pdf_path,
                    pipeline=pipeline,
                    output_markdown_path=output_path,
                    auto_tabular_db=False,
                    resume=True,
                )

            self.assertEqual(pipeline.pages, [2])
            self.assertEqual(render_pages.call_args.kwargs["pages"], [1])
            self.assertIn("# Page 1", result.full_markdown)
            self.assertIn("# Page 2", result.full_markdown)
            self.assertFalse(checkpoint_path.exists())

    def test_presentation_resume_only_processes_missing_slides(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pptx_path = root / "deck.pptx"
            pptx_path.write_bytes(b"placeholder")
            output_path = root / "deck.md"
            checkpoint_path = root / "logs" / "deck_checkpoint.json"
            _write_checkpoint(
                checkpoint_path,
                {
                    "version": 1,
                    "source_file": str(pptx_path),
                    "total_pages": 2,
                    "document_title": "Recovered deck",
                    "pages": {
                        "1": {
                            "markdown": "# Slide 1",
                            "visual_count": 0,
                            "table_count": 0,
                        }
                    },
                },
            )
            slide_images = [root / "slide_0001.jpg", root / "slide_0002.jpg"]
            pipeline = _Pipeline()

            with (
                patch(
                    "app.ppt.render_presentation_slides_to_images",
                    return_value=slide_images,
                ),
                patch("app.ppt.process_page_tabular_agent"),
                patch("app.ppt.prune_document_pages"),
                patch("app.ppt.cross_verify_dual_track"),
            ):
                markdown = process_presentation_vision(
                    pptx_path=pptx_path,
                    pipeline=pipeline,
                    output_markdown_path=output_path,
                    resume=True,
                )

            self.assertEqual(pipeline.pages, [2])
            self.assertIn("# Slide 1", markdown)
            self.assertIn("# Page 2", markdown)
            self.assertFalse(checkpoint_path.exists())


if __name__ == "__main__":
    unittest.main()
