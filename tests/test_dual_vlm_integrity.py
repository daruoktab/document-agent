"""Audit and paragraph editing tests with no live model requests."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from app.config import Settings
from app.extractor import VisionExtractor
from app.graph import DocumentExtractionPipeline
from app.language_refiner import refine_audited_markdown
from app.model_runtime import ModelRuntime


@pytest.mark.parametrize(
    "before,after",
    [
        ("Surat disetujui oleh Budi.", "Surat disetujui oleh Andi."),
        ("Budi: 100 rupiah. Sari: 200 rupiah.", "Budi: 200 rupiah. Sari: 100 rupiah."),
        (
            "Surat disetujui oleh Budi. Pelaksanaan wajib menunggu konfirmasi tertulis.",
            "Selesai.",
        ),
        ("Saldo: -100 rupiah.", "Saldo: 100 rupiah."),
        ("- 42", "42"),
        ("Hasil operasi: 2 ** 3.", "Hasil operasi: 2 3."),
    ],
)
def test_legacy_full_rewrites_cannot_silently_change_facts(before, after):
    llm = MagicMock()
    llm.invoke.return_value = SimpleNamespace(content=after)
    final, status = refine_audited_markdown(
        llm=llm, draft_markdown=before, audited_markdown=before
    )
    assert final == before
    assert status.startswith("fallback_")


@pytest.mark.parametrize(
    "verdict",
    [
        {"equivalent": False, "uncertain": False, "facts_preserved": False},
        {"equivalent": True, "uncertain": True, "facts_preserved": True},
        {"equivalent": "true", "uncertain": False, "facts_preserved": True},
    ],
)
def test_structured_rewrites_require_clear_visual_verification(tmp_path, verdict):
    before, after = (
        "Surat disetujui oleh Budi.",
        "Budi telah menyetujui surat tersebut.",
    )
    image = tmp_path / "page.png"
    Image.new("RGB", (10, 10)).save(image)
    llm, verifier = MagicMock(), MagicMock()
    llm.invoke.return_value.content = json.dumps(
        {"edits": [{"paragraph": 0, "before": before, "after": after}]}
    )
    verifier.invoke.return_value.content = json.dumps(verdict)
    final, status = refine_audited_markdown(
        llm=llm,
        draft_markdown=before,
        audited_markdown=before,
        verifier=verifier,
        image_path=str(image),
    )
    assert (final, status) == (before, "fallback_invalid")


def test_valid_sentence_rewrite_is_verified_against_image(tmp_path):
    before, after = (
        "Budi sudah melakukan persetujuan surat.",
        "Budi telah menyetujui surat.",
    )
    image = tmp_path / "page.png"
    Image.new("RGB", (10, 10)).save(image)
    llm, verifier = MagicMock(), MagicMock()
    llm.invoke.return_value.content = json.dumps(
        {"edits": [{"paragraph": 0, "before": before, "after": after}]}
    )
    verifier.invoke.return_value.content = (
        '{"equivalent": true, "uncertain": false, "facts_preserved": true}'
    )
    assert refine_audited_markdown(
        llm=llm,
        draft_markdown=before,
        audited_markdown=before,
        verifier=verifier,
        image_path=str(image),
    ) == (after, "accepted")
    assert "image_url" in str(verifier.invoke.call_args.args[0])
    assert "image_url" not in str(llm.invoke.call_args.args[0])


def test_language_verifier_outage_retains_audited_text(tmp_path):
    before = "Budi sudah melakukan persetujuan surat."
    image = tmp_path / "page.png"
    Image.new("RGB", (10, 10)).save(image)
    llm, verifier = MagicMock(), MagicMock()
    llm.invoke.return_value.content = json.dumps(
        {
            "edits": [
                {
                    "paragraph": 0,
                    "before": before,
                    "after": "Budi telah menyetujui surat.",
                }
            ]
        }
    )
    verifier.invoke.side_effect = TimeoutError("offline")
    final, status = refine_audited_markdown(
        llm=llm,
        draft_markdown=before,
        audited_markdown=before,
        verifier=verifier,
        image_path=str(image),
    )
    assert final == before
    assert status == "fallback_unverified"


def test_single_vlm_skips_extra_language_stage():
    model = MagicMock()
    llm = ModelRuntime(model).routed("agent")
    assert refine_audited_markdown(
        llm=llm, draft_markdown="text", audited_markdown="text"
    ) == (
        "text",
        "skipped_single_vlm",
    )
    model.invoke.assert_not_called()


def test_failed_audit_never_enables_language_refinement(tmp_path):
    image = tmp_path / "page.png"
    Image.new("RGB", (10, 10)).save(image)
    visual, agent = MagicMock(), MagicMock()
    visual.invoke.side_effect = TimeoutError("offline")
    agent.invoke.side_effect = TimeoutError("offline too")
    with patch("app.learning_store.LearningStore") as store:
        store.return_value.active_run.return_value = None
        pipeline = DocumentExtractionPipeline(
            Settings(ocr_model="", language_vlm_model="agent"),
            vlm=visual,
            language_vlm=agent,
            thorough=True,
        )
        state = pipeline._node_aggregate_and_judge(
            {"image_path": str(image), "markdown_content": "Dokumen asli."}
        )
        assert state["visual_audit_applied"] is False
        assert state["visual_audit_status"] == "fallback_error"
        assert (
            pipeline._node_refine_language(state)["language_refine_status"] == "skipped"
        )


@pytest.mark.parametrize(
    "edits,accepted",
    [
        ([{"line": 3, "column": 1, "before": "10", "after": "11"}], True),
        ([{"line": 3, "column": 1, "before": "wrong", "after": "11"}], False),
        ([{"line": 3, "column": 1, "before": "10", "after": "11|extra"}], False),
    ],
)
def test_long_table_audit_preserves_all_rows(tmp_path, edits, accepted):
    image = tmp_path / "page.png"
    Image.new("RGB", (10, 10)).save(image)
    draft = "# Data\n| Nama | Nilai |\n| --- | --- |\n" + "\n".join(
        f"| A{i} | 10 |" for i in range(30)
    )
    model = MagicMock()
    model.invoke.return_value.content = json.dumps(
        {"prose_valid": True, "edits": edits}
    )
    with patch("app.learning_store.LearningStore") as store:
        store.return_value.active_run.return_value = None
        result = VisionExtractor(model).audit_markdown(
            str(image), draft, preserve_tables=True
        )
    assert (result.action == "accepted") is accepted
    assert len(result.final_markdown.splitlines()) == len(draft.splitlines())
    assert "| A29 | 10 |" in result.final_markdown
    assert ("| A0 | 11 |" in result.final_markdown) is accepted


def test_learning_fallback_does_not_use_primary_program_again(tmp_path):
    image = tmp_path / "page.png"
    Image.new("RGB", (10, 10)).save(image)
    visual, agent = MagicMock(), MagicMock()
    agent.invoke.return_value.content = "Dokumen fallback."
    runtime = ModelRuntime(visual, agent)
    with patch("app.learning_store.LearningStore") as store:
        store.return_value.active_run.return_value = None
        extractor = VisionExtractor(runtime.routed("vision"))
    extractor.learning_program = object()
    extractor.learning_lm = object()
    with patch(
        "app.dspy_learning.predict_page", side_effect=TimeoutError("offline")
    ) as predict:
        assert extractor.extract_markdown(str(image)) == "Dokumen fallback."
        assert extractor.extract_markdown(str(image)) == "Dokumen fallback."
    predict.assert_called_once()
    visual.invoke.assert_not_called()
    assert agent.invoke.call_count == 2


def test_learning_program_is_not_applied_to_a_different_injected_model():
    from langchain_openai import ChatOpenAI

    from app.dspy_learning import model_fingerprint

    settings = Settings(vlm_model="configured", vlm_base_url="http://unused/v1")
    other = ChatOpenAI(
        model="different", api_key="test", base_url=settings.vlm_base_url
    )
    with patch("app.learning_store.LearningStore") as store:
        store.return_value.active_run.return_value = {
            "id": "active",
            "instructions": "optimized",
            "model_fingerprint": model_fingerprint(settings),
        }
        extractor = VisionExtractor(other, settings=settings)
    assert extractor.learning_program is None
