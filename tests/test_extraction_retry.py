"""Bounded recovery from responses that exhaust generation without an answer."""

import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from app.extractor import EmptyLLMResponseError, VisionExtractor


class TestExtractionRetry(unittest.TestCase):
    def extract(self, responses):
        llm = MagicMock()
        llm.invoke.side_effect = responses
        with patch("app.learning_store.LearningStore") as store:
            store.return_value.active_run.return_value = None
            extractor = VisionExtractor(llm)
        return llm, extractor

    @patch("app.extractor.image_data_uri", return_value="data:image/png;base64,AA==")
    def test_empty_length_retries_without_thinking(self, _image):
        llm, extractor = self.extract([
            AIMessage(content="", response_metadata={"finish_reason": "length"}),
            AIMessage(content="| A |\n| --- |\n| 42 |"),
        ])
        self.assertIn("42", extractor.extract_markdown("missing.png"))
        self.assertEqual(llm.invoke.call_count, 2)
        self.assertEqual(llm.invoke.call_args.kwargs, {
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        })

    @patch("app.extractor.image_data_uri", return_value="data:image/png;base64,AA==")
    def test_retry_is_bounded_and_empty_still_fails(self, _image):
        empty = AIMessage(content="", response_metadata={"finish_reason": "length"})
        llm, extractor = self.extract([empty, empty])
        with self.assertRaises(EmptyLLMResponseError):
            extractor.extract_markdown("missing.png")
        self.assertEqual(llm.invoke.call_count, 2)

    @patch("app.extractor.image_data_uri", return_value="data:image/png;base64,AA==")
    def test_empty_stop_does_not_retry(self, _image):
        llm, extractor = self.extract([
            AIMessage(content="", response_metadata={"finish_reason": "stop"}),
        ])
        with self.assertRaises(EmptyLLMResponseError):
            extractor.extract_markdown("missing.png")
        self.assertEqual(llm.invoke.call_count, 1)
