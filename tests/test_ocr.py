import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image, ImageDraw

from app.config import Settings
from app.graph import DocumentExtractionPipeline, DocumentExtractionState
from app.ocr import UnlimitedOCRExtractor, assess_ocr_quality, parse_grounding_regions
from app.pdf import extract_pdf_native_text_by_page
from app.preprocess import rotate_image_right_angle
from app.schemas import OCRExtractionResult


class TestOCRGrounding(unittest.TestCase):
    def test_visual_rescue_keeps_more_complete_trusted_ocr_table(self) -> None:
        headers = "| POSTING DATE | TRANSACTION DESCRIPTION | BALANCE |\n| --- | --- | --- |\n"
        ocr_markdown = headers + "\n".join(
            f"| 202402{i:02d} | Transfer {i} | {i} |" for i in range(1, 30)
        )
        vlm_markdown = headers + "\n".join(
            f"| 202402{i:02d} | Transfer {i} | {i} |" for i in range(1, 9)
        )
        main_llm = MagicMock()
        pipeline = DocumentExtractionPipeline(Settings(ocr_model="paddleocr-vl-1.6"), vlm=main_llm)
        state: DocumentExtractionState = {
            "image_path": "page.png",
            "specs": ["plain"],
            "requires_vlm_reading": True,
            "ocr_result": OCRExtractionResult(
                status="success",
                markdown=ocr_markdown,
                model="paddleocr-vl-1.6",
                trust_level="high",
            ).model_dump(),
        }
        with patch("app.graph.get_agent") as get_agent:
            get_agent.return_value.run.return_value = vlm_markdown
            selected = pipeline._node_extract_markdown(state)
        final = pipeline._node_aggregate_and_judge(selected)

        self.assertEqual(selected["ocr_status"], "accepted")
        self.assertTrue(selected["preserve_ocr_table"])
        self.assertIn("20240229", final["markdown_content"])
        self.assertNotIn("20240229", vlm_markdown)
        main_llm.invoke.assert_not_called()

    def test_parse_regions_maps_and_clamps_coordinates(self) -> None:
        raw = (
            "<|det|>table [100, 200, 900, 800]<|/det|>A | B\n"
            "<|det|>figure [-10, 0, 1100, 1024]<|/det|>Diagram"
        )

        regions = parse_grounding_regions(raw, image_size=(2048, 1024))

        self.assertEqual(len(regions), 2)
        self.assertEqual(regions[0].kind, "table")
        self.assertEqual(regions[0].bbox_pixels, (200, 200, 1800, 800))
        self.assertEqual(regions[1].kind, "figure")
        self.assertEqual(regions[1].bbox_pixels, (0, 0, 2048, 1024))

    def test_extractor_persists_only_visual_and_table_crops(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "page.png"
            Image.new("RGB", (1024, 1024), "white").save(image_path)

            response = MagicMock()
            response.content = (
                "<|det|>title [10, 10, 500, 80]<|/det|># Invoice\n"
                "<|det|>table [100, 100, 900, 500]<|/det|>| A | B |\n"
                "<|det|>figure [200, 550, 800, 950]<|/det|>Flow diagram"
            )
            llm = MagicMock()
            llm.invoke.return_value = response
            output_dir = root / "regions"

            result = UnlimitedOCRExtractor(
                llm,
                model_name="unlimited-ocr",
                prompt="<|grounding|>Convert the document to markdown.",
            ).extract(image_path, output_dir=output_dir)

            self.assertEqual(result.status, "success")
            self.assertEqual(len(result.regions), 3)
            self.assertIsNone(result.regions[0].crop_path)
            self.assertTrue(Path(result.regions[1].crop_path or "").is_file())
            self.assertTrue(Path(result.regions[2].crop_path or "").is_file())
            manifest = json.loads((output_dir / "manifest.json").read_text("utf-8"))
            self.assertEqual(manifest["model"], "unlimited-ocr")

    def test_extractor_returns_error_for_empty_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "page.png"
            Image.new("RGB", (32, 32), "white").save(image_path)
            response = MagicMock(content="")
            llm = MagicMock()
            llm.invoke.return_value = response

            result = UnlimitedOCRExtractor(
                llm,
                model_name="unlimited-ocr",
                prompt="Free OCR.",
            ).extract(image_path)

            self.assertEqual(result.status, "error")
            self.assertIn("respons kosong", result.error or "")

    def test_pipeline_uses_ocr_draft_before_vlm_judge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "page.png"
            image = Image.new("RGB", (256, 128), "white")
            draw = ImageDraw.Draw(image)
            for y in (20, 45, 70, 95):
                draw.rectangle((20, y, 230, y + 5), fill="black")
            image.save(image_path)

            ocr_response = MagicMock(
                content=(
                    "<|det|>text [50, 100, 900, 850]<|/det|>"
                    "# Judul\n\nIsi dari OCR yang cukup panjang."
                )
            )
            ocr_llm = MagicMock()
            ocr_llm.invoke.return_value = ocr_response

            inspect_response = MagicMock(
                content=json.dumps(
                    {
                        "specs": ["plain"],
                        "has_diagram": False,
                        "has_table": False,
                        "difficulty": "simple",
                        "requires_vlm_reading": False,
                    }
                )
            )
            judge_response = MagicMock(content="# Judul\n\nIsi final tervalidasi.")
            main_llm = MagicMock()
            main_llm.invoke.side_effect = [inspect_response, judge_response]

            pipeline = DocumentExtractionPipeline(
                Settings(ocr_model="unlimited-ocr"),
                vlm=main_llm,
                ocr_llm=ocr_llm,
            )
            result = pipeline.run(
                str(image_path),
                forced_specs="plain",
                region_output_dir=Path(tmp) / "regions",
            )

            self.assertEqual(result.ocr_status, "corrected_by_vlm")
            self.assertEqual(result.markdown_content, judge_response.content)
            self.assertEqual(ocr_llm.invoke.call_count, 1)
            self.assertEqual(main_llm.invoke.call_count, 2)

    def test_small_text_visual_rescue_uses_vlm_even_when_ocr_is_trusted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "small_text.png"
            image = Image.new("RGB", (256, 128), "white")
            ImageDraw.Draw(image).rectangle((20, 50, 230, 65), fill="black")
            image.save(image_path)

            ocr_llm = MagicMock()
            ocr_llm.invoke.return_value = MagicMock(
                content=(
                    "<|det|>text [50, 100, 900, 850]<|/det|>"
                    "Draft OCR yang dipercaya tetapi bukan sumber final."
                )
            )
            main_llm = MagicMock()
            main_llm.invoke.side_effect = [
                MagicMock(
                    content=json.dumps(
                        {
                            "specs": ["plain"],
                            "has_diagram": False,
                            "has_table": False,
                            "difficulty": "complex",
                            "requires_vlm_reading": True,
                            "reasoning": "Font sangat kecil dan teks miring.",
                        }
                    )
                ),
                MagicMock(content="# Dibaca VLM\n\nTeks kecil dan miring terbaca."),
                MagicMock(content="# Dibaca VLM\n\nTeks kecil dan miring final."),
            ]
            pipeline = DocumentExtractionPipeline(
                Settings(ocr_model="unlimited-ocr"),
                vlm=main_llm,
                ocr_llm=ocr_llm,
            )

            result = pipeline.run(str(image_path), forced_specs="plain")

            self.assertEqual(result.ocr_status, "vlm_visual_rescue")
            self.assertTrue(result.vlm_visual_rescue)
            self.assertIn("final", result.markdown_content)
            self.assertEqual(ocr_llm.invoke.call_count, 1)
            self.assertEqual(main_llm.invoke.call_count, 3)

    def test_blank_page_skips_ocr_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "blank.png"
            Image.new("RGB", (200, 200), "white").save(image_path)
            llm = MagicMock()

            result = UnlimitedOCRExtractor(
                llm,
                model_name="unlimited-ocr",
                prompt="Free OCR.",
            ).extract_robust(image_path)

            self.assertEqual(result.decision, "blank_page")
            self.assertEqual(result.markdown, "")
            llm.invoke.assert_not_called()

    def test_rotation_retry_selects_trusted_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "rotated.png"
            image = Image.new("RGB", (160, 260), "white")
            draw = ImageDraw.Draw(image)
            for x in (25, 55, 85, 115):
                draw.rectangle((x, 20, x + 5, 240), fill="black")
            image.save(image_path)

            llm = MagicMock()
            llm.invoke.side_effect = [
                MagicMock(content="????"),
                MagicMock(
                    content=(
                        "<|det|>text [80, 100, 940, 850]<|/det|>"
                        "Dokumen sudah terbaca dengan orientasi yang benar."
                    )
                ),
            ]
            result = UnlimitedOCRExtractor(
                llm,
                model_name="unlimited-ocr",
                prompt="<|grounding|>Convert the document to markdown.",
            ).extract_robust(image_path)

            self.assertEqual(result.decision, "retried_rotated")
            self.assertEqual(result.rotation_degrees, 90)
            self.assertEqual(result.trust_level, "high")
            self.assertEqual(llm.invoke.call_count, 2)

    def test_native_text_mismatch_marks_plausible_hallucination_risky(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "page.png"
            image = Image.new("RGB", (300, 150), "white")
            ImageDraw.Draw(image).rectangle((20, 50, 280, 65), fill="black")
            image.save(image_path)
            raw = (
                "<|det|>text [50, 200, 950, 500]<|/det|>"
                "Invoice total USD 99,999 for fictional customer."
            )
            regions = parse_grounding_regions(raw, image_size=image.size)

            quality = assess_ocr_quality(
                image_path=image_path,
                markdown="Invoice total USD 99,999 for fictional customer.",
                raw_response=raw,
                regions=regions,
                native_text="END OF DOCUMENT BaliTower Metro Ethernet",
            )

            self.assertIn("native_text_mismatch", quality.risk_flags)
            self.assertNotEqual(quality.trust_level, "high")

    def test_low_trust_ocr_uses_independent_vlm_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "page.png"
            image = Image.new("RGB", (300, 150), "white")
            ImageDraw.Draw(image).rectangle((20, 50, 280, 65), fill="black")
            image.save(image_path)

            ocr_llm = MagicMock()
            ocr_llm.invoke.return_value = MagicMock(
                content="Invoice total USD 99,999 for fictional customer."
            )
            main_llm = MagicMock()
            main_llm.invoke.side_effect = [
                MagicMock(
                    content=json.dumps(
                        {
                            "specs": ["plain"],
                            "has_diagram": False,
                            "has_table": False,
                            "difficulty": "simple",
                            "requires_vlm_reading": False,
                        }
                    )
                ),
                MagicMock(content="# END OF DOCUMENT"),
                MagicMock(content="# END OF DOCUMENT\n\nBaliTower Metro Ethernet"),
            ]
            pipeline = DocumentExtractionPipeline(
                Settings(ocr_model="unlimited-ocr", ocr_rotation_retry=False),
                vlm=main_llm,
                ocr_llm=ocr_llm,
            )

            result = pipeline.run(
                str(image_path),
                forced_specs="plain",
                native_text="END OF DOCUMENT BaliTower Metro Ethernet",
            )

            self.assertEqual(result.ocr_status, "fallback_vlm")
            self.assertIn("END OF DOCUMENT", result.markdown_content)
            self.assertIn("native_text_mismatch", result.ocr_risk_flags)
            self.assertEqual(ocr_llm.invoke.call_count, 1)
            self.assertEqual(main_llm.invoke.call_count, 3)

    def test_right_angle_rotation_swaps_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "page.png"
            Image.new("RGB", (80, 40), "white").save(image_path)
            rotated_path = rotate_image_right_angle(image_path, 90)
            with Image.open(rotated_path) as rotated:
                self.assertEqual(rotated.size, (40, 80))

    def test_pdf_native_text_is_available_as_independent_evidence(self) -> None:
        import pymupdf

        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "native.pdf"
            document = pymupdf.open()
            page = document.new_page()
            page.insert_text((72, 72), "END OF DOCUMENT")
            document.save(pdf_path)
            document.close()

            native = extract_pdf_native_text_by_page(pdf_path)

            self.assertIn("END OF DOCUMENT", native[1])


if __name__ == "__main__":
    unittest.main()
