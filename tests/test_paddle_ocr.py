import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from app.paddle_ocr import (
    PaddleOCRVLExtractor,
    paddle_region_kind,
    parse_paddle_regions,
)


class _FakePaddleResult:
    def __init__(self, payload: dict, markdown: str) -> None:
        self.json = {"res": payload}
        self.markdown = {"markdown_texts": markdown}


class _FakePaddlePipeline:
    def __init__(self, result: _FakePaddleResult) -> None:
        self.result = result
        self.calls: list[tuple[str, dict]] = []

    def predict(self, image_path: str, **kwargs):
        self.calls.append((image_path, kwargs))
        return [self.result]


class TestPaddleOCRVLExtractor(unittest.TestCase):
    def test_maps_semantic_region_labels(self) -> None:
        self.assertEqual(paddle_region_kind("doc_title"), "title")
        self.assertEqual(paddle_region_kind("paragraph_title"), "section")
        self.assertEqual(paddle_region_kind("figure_title"), "caption")
        self.assertEqual(paddle_region_kind("display_formula"), "formula")
        self.assertEqual(paddle_region_kind("page_number"), "footer")

    def test_parse_regions_keeps_bbox_content_and_order(self) -> None:
        regions = parse_paddle_regions(
            {
                "parsing_res_list": [
                    {
                        "block_label": "text",
                        "block_content": "Baris pertama\nBaris kedua",
                        "block_bbox": [-10, 20, 1200, 800],
                        "block_order": 3,
                    }
                ]
            },
            image_size=(1000, 600),
        )

        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].bbox_pixels, (0, 20, 1000, 600))
        self.assertEqual(regions[0].reading_order, 3)
        self.assertEqual(regions[0].lines, ["Baris pertama", "Baris kedua"])

    def test_extract_normalizes_result_and_persists_visual_crops(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "page.png"
            image = Image.new("RGB", (400, 300), "white")
            draw = ImageDraw.Draw(image)
            for y in (30, 60, 90, 120):
                draw.rectangle((20, y, 370, y + 5), fill="black")
            image.save(image_path)
            payload = {
                "parsing_res_list": [
                    {
                        "block_label": "doc_title",
                        "block_content": "Laporan",
                        "block_bbox": [20, 20, 380, 55],
                        "block_order": 1,
                    },
                    {
                        "block_label": "table",
                        "block_content": "| A | B |\n|---|---|\n| 1 | 2 |",
                        "block_bbox": [20, 70, 380, 180],
                        "block_order": None,
                    },
                    {
                        "block_label": "image",
                        "block_content": "",
                        "block_bbox": [40, 190, 360, 290],
                        "block_order": None,
                    },
                ]
            }
            pipeline = _FakePaddlePipeline(
                _FakePaddleResult(payload, "# Laporan\n\n| A | B |\n|---|---|\n| 1 | 2 |")
            )
            output_dir = root / "regions"
            extractor = PaddleOCRVLExtractor(
                base_url="http://127.0.0.1:8081/v1",
                model_name="paddleocr-vl-1.6",
                pipeline=pipeline,
                rotation_retry=False,
            )

            result = extractor.extract_robust(image_path, output_dir=output_dir)

            self.assertEqual(result.status, "success")
            self.assertEqual(result.model, "paddleocr-vl-1.6")
            self.assertEqual([region.kind for region in result.regions], ["title", "table", "figure"])
            self.assertTrue(Path(result.regions[1].crop_path or "").is_file())
            self.assertTrue(Path(result.regions[2].crop_path or "").is_file())
            manifest = json.loads((output_dir / "manifest.json").read_text("utf-8"))
            self.assertEqual(manifest["source_markdown"], result.markdown)
            self.assertEqual(len(pipeline.calls), 1)

    def test_empty_pipeline_result_becomes_recoverable_error(self) -> None:
        class EmptyPipeline:
            def predict(self, image_path: str, **kwargs):
                return []

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "page.png"
            Image.new("RGB", (80, 80), "black").save(image_path)
            result = PaddleOCRVLExtractor(
                base_url="http://127.0.0.1:8081/v1",
                model_name="paddleocr-vl-1.6",
                pipeline=EmptyPipeline(),
                rotation_retry=False,
            ).extract_robust(image_path)

            self.assertEqual(result.status, "error")
            self.assertEqual(result.decision, "error")

    def test_image_placeholders_are_removed_without_losing_visual_regions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "page.png"
            image = Image.new("RGB", (400, 300), "white")
            ImageDraw.Draw(image).rectangle((20, 20, 350, 150), fill="black")
            image.save(image_path)
            payload = {
                "parsing_res_list": [
                    {
                        "block_label": "image",
                        "block_content": "",
                        "block_bbox": [20, 20, 350, 150],
                    }
                ]
            }
            markdown = (
                '## Perangkat\n\n<div style="text-align: center;">'
                '<img src="imgs/img_in_image_box_20_20_350_150.jpg" alt="Image" />'
                "</div>\n\n7210 SAS-Sx 10/100GE\n\n"
                "![Image](imgs/another_placeholder.jpg)\n\n"
                "> **[Diagram/Visual]:** Panel depan perangkat Nokia."
            )
            extractor = PaddleOCRVLExtractor(
                base_url="http://127.0.0.1:8081/v1",
                model_name="paddleocr-vl-1.6",
                pipeline=_FakePaddlePipeline(_FakePaddleResult(payload, markdown)),
                rotation_retry=False,
            )

            result = extractor.extract_robust(
                image_path, output_dir=root / "regions"
            )

            self.assertEqual(result.status, "success")
            self.assertEqual(
                result.markdown,
                "## Perangkat\n\n7210 SAS-Sx 10/100GE\n\n"
                "> **[Diagram/Visual]:** Panel depan perangkat Nokia.",
            )
            self.assertTrue(Path(result.regions[0].crop_path or "").is_file())
            manifest = json.loads((root / "regions" / "manifest.json").read_text("utf-8"))
            self.assertEqual(manifest["source_markdown"], result.markdown)


if __name__ == "__main__":
    unittest.main()
