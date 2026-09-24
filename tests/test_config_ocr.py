import os
import unittest
from unittest.mock import MagicMock, patch

from app.config import Settings
from app.graph import DocumentExtractionPipeline


class TestOCRSettings(unittest.TestCase):
    def test_server_defaults_keep_ocr_unconfigured(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()

        self.assertEqual(settings.base_url, "http://127.0.0.1:8080/v1")
        self.assertEqual(
            settings.vlm_model,
            "gemma-4-12b-vlm",
        )
        self.assertEqual(settings.ocr_backend, "paddleocr_vl")
        self.assertEqual(settings.ocr_model, "")
        self.assertEqual(settings.ocr_base_url, "http://127.0.0.1:8081/v1")
        self.assertEqual(settings.ocr_temperature, 0.0)
        self.assertEqual(settings.ocr_max_tokens, 4096)
        self.assertEqual(
            settings.ocr_prompt,
            "<|grounding|>Convert the document to markdown.",
        )
        self.assertEqual(settings.ocr_min_trust_score, 0.72)
        self.assertEqual(settings.ocr_medium_trust_score, 0.48)
        self.assertTrue(settings.ocr_rotation_retry)
        self.assertTrue(settings.textreflow_enabled)
        self.assertTrue(settings.excel_native_survey)
        self.assertTrue(settings.excel_region_rendering)
        self.assertEqual(settings.excel_base_dpi, 300)
        self.assertEqual(settings.excel_max_dpi, 450)
        self.assertEqual(settings.excel_small_font_points, 8.0)
        self.assertTrue(settings.vlm_visual_rescue)

    def test_ocr_endpoint_and_model_are_independent_from_vlm(self) -> None:
        with patch.dict(
            os.environ,
            {
                "BASE_URL": "http://localhost:8080/v1",
                "VLM_MODEL": "main-vlm",
                "OCR_MODEL": "unlimited-ocr",
                "OCR_BASE_URL": "http://localhost:8081/v1",
            },
            clear=True,
        ):
            settings = Settings()

        self.assertEqual(settings.vlm_model, "main-vlm")
        self.assertEqual(settings.ocr_model, "unlimited-ocr")
        self.assertEqual(settings.vlm_base_url, "http://localhost:8080/v1")
        self.assertEqual(settings.ocr_base_url, "http://localhost:8081/v1")
        self.assertNotEqual(settings.vlm_base_url, settings.ocr_base_url)

    def test_legacy_llm_base_url_remains_a_vlm_fallback(self) -> None:
        with patch.dict(
            os.environ,
            {"LLM_BASE_URL": "http://legacy-server:9000/v1"},
            clear=True,
        ):
            settings = Settings()

        self.assertEqual(settings.vlm_base_url, "http://legacy-server:9000/v1")
        self.assertEqual(settings.ocr_base_url, "http://127.0.0.1:8081/v1")

    def test_pipeline_selects_paddle_backend_without_eager_model_load(self) -> None:
        settings = Settings(
            ocr_backend="paddleocr_vl",
            ocr_model="paddleocr-vl-1.6",
            ocr_base_url="http://localhost:8081/v1",
        )
        with patch("app.graph.PaddleOCRVLExtractor") as extractor_cls:
            extractor = MagicMock()
            extractor_cls.return_value = extractor

            pipeline = DocumentExtractionPipeline(settings, vlm=MagicMock())

        self.assertIs(pipeline.ocr_extractor, extractor)
        extractor_cls.assert_called_once_with(
            base_url="http://localhost:8081/v1",
            model_name="paddleocr-vl-1.6",
            api_key=settings.ocr_api_key,
            timeout=settings.ocr_timeout,
            max_tokens=settings.ocr_max_tokens,
            crop_padding=settings.ocr_crop_padding,
            min_trust_score=settings.ocr_min_trust_score,
            medium_trust_score=settings.ocr_medium_trust_score,
            rotation_retry=settings.ocr_rotation_retry,
            blank_ink_ratio=settings.ocr_blank_ink_ratio,
            sparse_ink_ratio=settings.ocr_sparse_ink_ratio,
        )


if __name__ == "__main__":
    unittest.main()
