import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

from PIL import Image

from app.extractor import VisionExtractor
from app.graph import DocumentExtractionPipeline
from app.schemas import DiagramExtractionResult


class TestAgentOrchestrator(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.img_file = self.test_dir / "test_img.png"
        img = Image.new("RGB", (400, 300), color=(255, 255, 255))
        img.save(str(self.img_file))

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_extractor_inspect_page(self):
        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = json.dumps(
            {
                "specs": ["presentation_slides"],
                "has_diagram": True,
                "diagram_type": "flowchart",
                "has_table": False,
                "rotation_degrees": 90,
                "requires_vlm_reading": True,
            }
        )
        mock_llm.invoke.return_value = mock_resp

        extractor = VisionExtractor(mock_llm)
        res = extractor.inspect_page(str(self.img_file))

        self.assertEqual(res["specs"], ["presentation_slides"])
        self.assertTrue(res["has_diagram"])
        self.assertEqual(res["diagram_type"], "flowchart")
        self.assertEqual(res.rotation_degrees, 90)
        self.assertTrue(res.requires_vlm_reading)

    def test_extractor_judge_and_refine(self):
        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = (
            "# Judul Terkoreksi\n\n- Poin 1\n- Poin 2\n\n"
            "```mermaid\nflowchart TD\n  A --> B\n```"
        )
        mock_llm.invoke.return_value = mock_resp

        extractor = VisionExtractor(mock_llm)
        draft = "# Judul Awal\n- Poin 1"
        refined = extractor.judge_and_refine(str(self.img_file), draft)

        self.assertIn("Judul Terkoreksi", refined)
        self.assertIn("```mermaid", refined)

    def test_judge_preserves_diagrams_when_replaced_with_description(self):
        diagram = '```mermaid\nflowchart TD\n A["Start"] --> B["End"]\n```'
        mock_llm = MagicMock()
        extractor = VisionExtractor(mock_llm)
        for remaining in (0, 1):
            with self.subTest(remaining=remaining):
                draft = "# Dua Diagram\n\n" + diagram + "\n\n" + diagram
                mock_llm.invoke.return_value.content = (
                    "# Dua Diagram\n\n> **[Diagram/Visual]:** Alur proses.\n\n"
                    + (diagram if remaining else "")
                )
                self.assertEqual(extractor.judge_and_refine(str(self.img_file), draft), draft)

    def test_judge_does_not_preserve_non_flowchart_mermaid(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = (
            "# Interaksi\n\n"
            "> **[Diagram/Visual]:** Pengguna mengirim permintaan ke layanan."
        )
        extractor = VisionExtractor(mock_llm)
        draft = (
            "# Interaksi\n\n"
            "```mermaid\nsequenceDiagram\nUser->>API: Request\n```"
        )

        refined = extractor.judge_and_refine(str(self.img_file), draft)

        self.assertNotIn("sequenceDiagram", refined)
        self.assertNotIn("```mermaid", refined)
        self.assertIn("Pengguna mengirim permintaan", refined)

    def test_pipeline_orchestration_with_diagram_and_judge(self):
        mock_llm = MagicMock()

        # Step 1: inspect_page
        resp_inspect = MagicMock()
        resp_inspect.content = json.dumps(
            {
                "specs": ["presentation_slides"],
                "has_diagram": True,
                "diagram_type": "flowchart",
                "has_table": False,
            }
        )

        # Step 2: extract_markdown (text)
        resp_text = MagicMock()
        resp_text.content = "# Slide 1: Arsitektur Sistem\n- Komponen A terhubung ke Komponen B"

        # Step 3: specialist tetap memvalidasi hint flowchart sebelum ekstraksi.
        resp_diag_classify = MagicMock()
        resp_diag_classify.content = json.dumps(
            {
                "is_convertible": True,
                "diagram_type": "flowchart",
                "recommended_format": "mermaid",
                "mermaid_type": "flowchart",
                "confidence": 0.98,
                "reasoning": "Alur komponen memiliki panah yang jelas",
            }
        )

        # Step 4: extract diagram to Mermaid.
        resp_diag_extract = MagicMock()
        resp_diag_extract.content = """```mermaid
flowchart TD
  A[Komponen A] --> B[Komponen B]
```
Diagram alur komponen sistem."""

        # Step 5: judge_and_refine
        resp_judge = MagicMock()
        resp_judge.content = (
            "# Slide 1: Arsitektur Sistem\n\n"
            "- Komponen A terhubung ke Komponen B\n\n"
            "```mermaid\nflowchart TD\n  A[Komponen A] --> B[Komponen B]\n```\n\n"
            "> **[Diagram Summary]:** Diagram alur komponen sistem."
        )

        mock_llm.invoke.side_effect = [
            resp_inspect,
            resp_text,
            resp_diag_classify,
            resp_diag_extract,
            resp_judge,
        ]

        pipeline = DocumentExtractionPipeline(vlm=mock_llm)
        result = pipeline.run(str(self.img_file))

        self.assertTrue(result["has_diagram"])
        self.assertIn("```mermaid", result["markdown_content"])
        self.assertIn("Komponen A", result["markdown_content"])
        self.assertIn("Slide 1: Arsitektur Sistem", result["markdown_content"])

    def test_non_flowchart_description_is_added_to_final_markdown(self):
        pipeline = object.__new__(DocumentExtractionPipeline)
        pipeline.thorough = False
        pipeline.vlm = MagicMock()
        pipeline.extractor = cast(
            Any,
            SimpleNamespace(
                judge_and_refine=lambda **kwargs: kwargs["draft_markdown"]
            ),
        )
        state = {
            "image_path": str(self.img_file),
            "markdown_content": "# Diagram Interaksi",
            "has_diagram": True,
            "diagram_type": "sequence_diagram",
            "ocr_status": "disabled",
            "ocr_regions": [],
            "specs": ["presentation_slides"],
        }
        description = "Pengguna mengirim permintaan ke API, lalu API mengembalikan respons."
        result = DiagramExtractionResult(
            status="unsuitable",
            is_mermaid=False,
            diagram_type="sequence_diagram",
            text_description=description,
            text_summary=description,
        )

        with patch("app.diagram.extract_diagram_to_mermaid", return_value=result):
            specialist_state = pipeline._node_summon_diagram_specialist(
                cast(Any, state)
            )
        final_state = pipeline._node_aggregate_and_judge(specialist_state)

        self.assertNotIn("```mermaid", final_state["markdown_content"])
        self.assertIn("> **[Diagram/Visual]:** Pengguna", final_state["markdown_content"])


if __name__ == "__main__":
    unittest.main()
