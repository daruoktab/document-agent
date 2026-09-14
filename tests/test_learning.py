"""Learning regression tests use synthetic documents and no external model calls."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import dspy
from dspy.utils import DummyLM
from PIL import Image

from app.dspy_learning import (
    PageProgram,
    build_learning_lm,
    is_improvement,
    model_fingerprint,
    optimize,
    predict_page,
    score_markdown,
)
from app.learning_store import LearningStore, split_examples

GOLD = "Laporan\n\n| Barang | Nilai |\n| --- | --- |\n| Buku | 100 |"
WRONG = "Laporan\n\n| Barang | Nilai |\n| --- | --- |\n| Buku | 10 |"


class LearningFakeLM(DummyLM):
    """Respond to both GEPA reflection and extraction through the real DSPy adapter."""

    def __init__(self):
        super().__init__([])
        self.saw_image = False

    def forward(self, prompt=None, messages=None, **kwargs):
        messages = messages or []
        self.saw_image |= any(
            isinstance(m.get("content"), list)
            and any(part.get("type") == "image_url" for part in m["content"])
            for m in messages
        )
        system = str(messages[0]["content"]) if messages else ""
        if "improved_instruction" in system:
            self.answers = iter(
                [
                    {
                        "improved_instruction": "LEARNED: Baca teks dan angka tepat sesuai gambar."
                    }
                ]
            )
        else:
            self.answers = iter([{"markdown": GOLD if "LEARNED:" in system else WRONG}])
        return super().forward(prompt=prompt, messages=messages, **kwargs)


class TestLearning(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = LearningStore(self.root / "learning")
        self.env = patch.dict(os.environ, {"LEARNING_DATA_DIR": str(self.store.root)})
        self.env.start()
        self.image = self.root / "page.png"
        Image.new("RGB", (16, 16), "white").save(self.image)

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def save(self, **kwargs):
        data: dict[str, Any] = {
            "document_hash": "doc",
            "page": 1,
            "image_path": self.image,
            "original": WRONG,
            "corrected": GOLD,
            "context": "Baca halaman ini.",
            "version": "baseline",
            "ready": True,
            "document_name": "Laporan",
        }
        data.update(kwargs)
        return self.store.save_correction(**data)

    def seed(self):
        for i in range(6):
            path = self.root / f"{i}.png"
            Image.new("RGB", (16, 16), (i * 30, 0, 0)).save(path)
            self.save(document_hash=f"doc{i}", image_path=path)

    def candidate(self, **kwargs):
        data = {
            "status": "completed",
            "eligible": True,
            "baseline_version": "baseline",
            "model_fingerprint": model_fingerprint(),
            "instructions": "LEARNED: Read faithfully.",
            "example_ids": [e["id"] for e in self.store.examples()],
        }
        data.update(kwargs)
        self.store.save_run(data)
        return data

    def test_correction_revisions_and_withdrawal_preserve_original(self):
        first = self.save()
        self.save(corrected="Draf baru", ready=False)
        self.assertEqual(self.store.examples(), [])
        latest = self.store.latest_correction("doc", 1)
        assert latest is not None
        self.assertFalse(latest["ready"])
        with self.store.connection() as conn:
            import json

            old = json.loads(
                conn.execute(
                    "SELECT data FROM corrections WHERE id=?", (first,)
                ).fetchone()[0]
            )
        self.assertEqual(old["original"], WRONG)
        self.assertEqual(old["corrected"], GOLD)

    def test_images_survive_deletion_of_source(self):
        original = self.image.read_bytes()
        self.save()
        self.image.unlink()
        self.assertEqual(self.store.examples()[0]["image"], original)

    def test_dataset_split_groups_copies_and_pages(self):
        self.seed()
        self.save(document_hash="copy", image_path=self.root / "0.png")
        self.save(document_hash="doc0", page=2, image_path=self.image)
        splits = split_examples(self.store.examples())
        locations = {
            e["document_hash"]: name for name, subset in splits.items() for e in subset
        }
        self.assertEqual(locations["doc0"], locations["copy"])
        for a, b in (
            ("train", "test"),
            ("train", "validation"),
            ("validation", "test"),
        ):
            self.assertFalse(
                {e["image_hash"] for e in splits[a]}
                & {e["image_hash"] for e in splits[b]}
            )

    def test_partitions_remain_fixed_after_new_uploads(self):
        self.seed()
        before = self.store.dataset_split(self.store.examples())
        self.save(document_hash="additional", image_path=self.image)
        after = self.store.dataset_split(self.store.examples())
        for split, subset in before.items():
            self.assertTrue(
                {e["id"] for e in subset} <= {e["id"] for e in after[split]}
            )

    def test_duplicate_bridge_between_frozen_partitions_rejected(self):
        self.seed()
        splits = self.store.dataset_split(self.store.examples())
        self.save(document_hash="additional", image_path=self.image)
        test_doc = splits["test"][0]["document_hash"]
        training_page = self.root / f"{splits['train'][0]['document_hash'][-1]}.png"
        self.save(document_hash=test_doc, page=2, image_path=training_page)
        with self.assertRaisesRegex(ValueError, "dataset berbeda"):
            self.store.dataset_split(self.store.examples())

    def test_metrics_detect_numeric_row_and_column_changes(self):
        self.assertEqual(score_markdown(GOLD, GOLD)["score"], 1)
        result = score_markdown(GOLD, WRONG)
        self.assertLess(result["table"], 1)
        self.assertIn("100", result["feedback"])
        self.assertLess(
            score_markdown(GOLD, GOLD.replace("| Buku | 100 |", ""))["table"], 1
        )
        self.assertLess(
            score_markdown(GOLD, GOLD.replace("| Buku | 100 |", "| 100 | Buku |"))[
                "table"
            ],
            1,
        )
        self.assertLess(score_markdown(GOLD, GOLD.replace("Laporan", ""))["text"], 1)
        self.assertEqual(score_markdown(GOLD, "")["score"], 0)
        self.assertFalse(
            is_improvement(
                {"score": 0.7, "text": 0.8, "table": 0.6},
                {"score": 0.75, "text": 0.7, "table": 0.8},
            )
        )

    def test_activation_gate_staleness_and_rollback(self):
        self.save()
        failed = self.candidate(eligible=False)
        with self.assertRaises(ValueError):
            self.store.activate(failed["id"], model_fingerprint())
        candidate = self.candidate()
        with self.assertRaises(ValueError):
            self.store.activate(candidate["id"], "different-model")
        self.store.activate(candidate["id"], model_fingerprint())
        self.assertEqual(self.store.active(), candidate["id"])
        with self.assertRaises(ValueError):
            self.store.activate(candidate["id"], model_fingerprint())
        self.assertEqual(self.store.rollback(), "baseline")
        newer = self.candidate()
        self.save(ready=False)
        with self.assertRaisesRegex(ValueError, "Koreksi berubah"):
            self.store.activate(newer["id"], model_fingerprint())

    def test_dspy_multimodal_payload_and_config(self):
        from app.config import Settings

        lm = LearningFakeLM()
        text = predict_page(
            PageProgram("LEARNED: Read"),
            lm,
            dspy.Image.from_path(str(self.image)),
            "read",
        )
        self.assertEqual(text, GOLD)
        self.assertTrue(lm.saw_image)
        configured = build_learning_lm(
            Settings(
                vlm_model="test-model",
                vlm_base_url="http://localhost:9999/v1",
                vlm_api_key="test-key",
            )
        )
        self.assertEqual(configured.model, "openai/test-model")
        self.assertEqual(configured.kwargs["api_base"], "http://localhost:9999/v1")
        self.assertFalse(configured.cache)

    def test_optimizer_failure_does_not_publish(self):
        self.seed()
        with (
            patch(
                "app.dspy_learning.build_learning_lm",
                side_effect=RuntimeError("provider unavailable"),
            ),
            self.assertRaises(RuntimeError),
        ):
            optimize(self.store)
        self.assertEqual(self.store.active(), "baseline")
        self.assertEqual(self.store.runs()[0]["status"], "failed")

    def test_real_gepa_cycle_and_runtime_activation(self):
        from app.extractor import VisionExtractor

        self.seed()
        fake = LearningFakeLM()
        legacy = MagicMock()
        legacy.invoke.return_value = SimpleNamespace(content=WRONG)
        with (
            patch("app.dspy_learning.build_learning_lm", return_value=fake),
            patch("app.llm.build_vlm", return_value=legacy),
        ):
            result = optimize(self.store, max_metric_calls=12)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["eligible"])
        self.assertEqual(self.store.active(), "baseline")
        self.store.activate(result["id"], model_fingerprint())
        with patch(
            "app.dspy_learning.build_learning_lm", return_value=LearningFakeLM()
        ):
            extractor = VisionExtractor(legacy)
        self.assertEqual(extractor.extract_markdown(str(self.image)), GOLD)
        observation = self.store.observation(self.image)
        assert observation is not None
        self.assertEqual(observation["version"], result["id"])
        self.store.rollback()
        self.assertEqual(
            VisionExtractor(legacy).extract_markdown(str(self.image)), WRONG
        )

    def test_streamlit_correction_form(self):
        from streamlit.testing.v1 import AppTest

        def correction_app(path: str):
            from pathlib import Path

            from app.learning_ui import render_correction_form

            render_correction_form(
                stem="demo",
                page=1,
                image_path=Path(path),
                original="Hasil awal",
                images=[Path(path)],
                source_path=None,
            )

        app = AppTest.from_function(correction_app, args=(str(self.image),)).run()
        self.assertEqual(len(app.exception), 0)
        app.text_area[0].set_value(GOLD)
        app.checkbox[0].check()
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(self.store.examples()[0]["corrected"], GOLD)
        app.checkbox[0].uncheck()
        app.button[0].click().run()
        self.assertEqual(self.store.examples(), [])


if __name__ == "__main__":
    unittest.main()
