import os
import unittest
from unittest.mock import patch

from app.config import Settings


class TestOCRSettings(unittest.TestCase):
    def test_server_defaults_keep_ocr_unconfigured(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()

        self.assertEqual(settings.base_url, "http://127.0.0.1:8080/v1")
        self.assertEqual(
            settings.vlm_model,
            "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf",
        )
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


if __name__ == "__main__":
    unittest.main()
