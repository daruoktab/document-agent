import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from PIL import Image

from app.extractor import VisionExtractor
from app.graph import DocumentExtractionPipeline


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

        # Step 3: extract diagram to mermaid (forced_diagram_type='flowchart' langsung mengekstrak)
        resp_diag_extract = MagicMock()
        resp_diag_extract.content = """```mermaid
flowchart TD
  A[Komponen A] --> B[Komponen B]
```
Diagram alur komponen sistem."""

        # Step 4: judge_and_refine
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
            resp_diag_extract,
            resp_judge,
        ]

        pipeline = DocumentExtractionPipeline(vlm=mock_llm)
        result = pipeline.run(str(self.img_file))

        self.assertTrue(result["has_diagram"])
        self.assertIn("```mermaid", result["markdown_content"])
        self.assertIn("Komponen A", result["markdown_content"])
        self.assertIn("Slide 1: Arsitektur Sistem", result["markdown_content"])


if __name__ == "__main__":
    unittest.main()
