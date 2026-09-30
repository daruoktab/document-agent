"""Routing model visual dan bahasa pada pipeline dokumen."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from app.config import Settings
from app.diagram import extract_diagram_to_mermaid
from app.language_refiner import refine_audited_markdown
from app.llm import build_language_vlm, build_vlm
from app.schemas import PageTabularEvent


def _response(content: str) -> SimpleNamespace:
    return SimpleNamespace(content=content, response_metadata={})


def test_named_role_environment_keys_override_legacy_names() -> None:
    environment = {
        "VLM_VISION_FOCUS_MODEL": "vision-new",
        "VLM_MODEL": "vision-old",
        "VLM_VISION_FOCUS_BASE_URL": "http://vision-new/v1",
        "LLM_BASE_URL": "http://vision-old/v1",
        "VLM_VISION_FOCUS_API_KEY": "vision-new-key",
        "LLM_API_KEY": "vision-old-key",
        "VLM_VISION_FOCUS_TEMPERATURE": "0.2",
        "VLM_VISION_FOCUS_TIMEOUT": "45",
        "VLM_VISION_FOCUS_MAX_TOKENS": "2048",
        "VLM_VISION_FOCUS_ENABLE_THINKING": "false",
        "VLM_ENABLE_THINKING": "true",
        "VLM_AGENT_FOCUS_MODEL": "agent-new",
        "LANGUAGE_VLM_MODEL": "agent-old",
        "VLM_AGENT_FOCUS_BASE_URL": "http://agent-new/v1",
        "VLM_AGENT_FOCUS_API_KEY": "agent-new-key",
        "VLM_AGENT_FOCUS_TEMPERATURE": "0.3",
        "VLM_AGENT_FOCUS_TIMEOUT": "90",
        "VLM_AGENT_FOCUS_MAX_TOKENS": "8192",
        "VLM_AGENT_FOCUS_ENABLE_THINKING": "false",
        "LANGUAGE_VLM_ENABLE_THINKING": "true",
    }
    with patch.dict(os.environ, environment, clear=True):
        settings = Settings()

    assert settings.vlm_model == "vision-new"
    assert settings.base_url == settings.vlm_base_url == "http://vision-new/v1"
    assert settings.vlm_api_key == "vision-new-key"
    assert settings.vlm_temperature == 0.2
    assert settings.vlm_timeout == 45
    assert settings.vlm_max_tokens == 2048
    assert settings.vlm_enable_thinking is False
    assert settings.language_vlm_model == "agent-new"
    assert settings.language_vlm_base_url == "http://agent-new/v1"
    assert settings.language_vlm_api_key == "agent-new-key"
    assert settings.language_vlm_temperature == 0.3
    assert settings.language_vlm_timeout == 90
    assert settings.language_vlm_max_tokens == 8192
    assert settings.language_vlm_enable_thinking is False


def test_legacy_role_environment_keys_remain_supported() -> None:
    with patch.dict(
        os.environ,
        {
            "LLM_BASE_URL": "http://legacy/v1",
            "LLM_API_KEY": "legacy-key",
            "VLM_MODEL": "legacy-vision",
            "LANGUAGE_VLM_MODEL": "legacy-agent",
            "LANGUAGE_VLM_ENABLE_THINKING": "true",
        },
        clear=True,
    ):
        settings = Settings()

    assert settings.vlm_model == "legacy-vision"
    assert settings.vlm_base_url == "http://legacy/v1"
    assert settings.vlm_api_key == "legacy-key"
    assert settings.language_vlm_model == "legacy-agent"
    assert settings.language_vlm_enable_thinking is True


def test_visual_model_builder_uses_vision_focus_role() -> None:
    settings = Settings(vlm_model="vision-model", vlm_base_url="http://vision/v1")
    with patch("app.llm.build_chat_model") as builder:
        build_vlm(settings)

    kwargs = builder.call_args.kwargs
    assert kwargs["model"] == "vision-model"
    assert kwargs["base_url"] == "http://vision/v1"
    assert kwargs["callbacks"][0].role == "vlm-vision-focus"


def test_language_vlm_builder_uses_independent_endpoint_and_no_template_flag() -> None:
    settings = Settings(
        language_vlm_model="language-model",
        language_vlm_base_url="http://localhost:9999/v1",
        language_vlm_api_key="language-key",
        language_vlm_enable_thinking=None,
    )
    with patch("app.llm.build_chat_model") as builder:
        build_language_vlm(settings)

    kwargs = builder.call_args.kwargs
    assert kwargs["model"] == "language-model"
    assert kwargs["base_url"] == "http://localhost:9999/v1"
    assert kwargs["api_key"] == "language-key"
    assert kwargs["enable_thinking"] is None
    assert kwargs["callbacks"][0].role == "vlm-agent-focus"

    inherited = Settings(
        vlm_base_url="http://localhost:8080/v1",
        vlm_api_key="visual-key",
        language_vlm_model="language-model",
        language_vlm_base_url="",
        language_vlm_api_key="",
    )
    with patch("app.llm.build_chat_model") as builder:
        build_language_vlm(inherited)
    assert builder.call_args.kwargs["base_url"] == inherited.vlm_base_url
    assert builder.call_args.kwargs["api_key"] == inherited.vlm_api_key


def test_agent_model_builder_requires_model_name() -> None:
    with pytest.raises(ValueError, match="VLM_AGENT_FOCUS_MODEL"):
        build_language_vlm(Settings(language_vlm_model=""))


def test_pipeline_keeps_images_on_gemma_and_refines_text_on_language_vlm(tmp_path) -> None:
    from app.graph import DocumentExtractionPipeline

    image = tmp_path / "page.png"
    Image.new("RGB", (300, 160), "white").save(image)
    visual = MagicMock()
    visual.invoke.side_effect = [
        _response(json.dumps({"specs": ["plain"], "has_diagram": False,
                              "has_table": False, "difficulty": "simple",
                              "rotation_degrees": 0, "requires_vlm_reading": False})),
        _response("# Judul\n\nIsi halaman 42."),
        _response("# Judul\n\nIsi halaman 42."),
    ]
    language = MagicMock()
    language.invoke.return_value = _response("## Judul\n\nIsi halaman 42.\n")
    settings = Settings(ocr_model="", language_vlm_model="language-model")

    with patch("app.learning_store.LearningStore") as store:
        store.return_value.active_run.return_value = None
        pipeline = DocumentExtractionPipeline(
            settings, vlm=visual, language_vlm=language
        )
        result = pipeline.run(str(image))

    assert result.markdown_content == "## Judul\n\nIsi halaman 42."
    assert visual.invoke.call_count == 3
    for call in visual.invoke.call_args_list:
        assert "image_url" in str(call.args[0])
    language.invoke.assert_called_once()
    assert "image_url" not in str(language.invoke.call_args.args[0])


def test_pipeline_without_language_model_keeps_optional_stage_disabled() -> None:
    from app.graph import DocumentExtractionPipeline, DocumentExtractionState

    visual = MagicMock()
    with (
        patch("app.graph.build_language_vlm") as build_language,
        patch("app.learning_store.LearningStore") as store,
    ):
        store.return_value.active_run.return_value = None
        pipeline = DocumentExtractionPipeline(
            Settings(ocr_model="", language_vlm_model=""), vlm=visual
        )

    assert pipeline.language_vlm is None
    build_language.assert_not_called()
    state: DocumentExtractionState = {
        "markdown_content": "# Sudah diaudit",
        "final_markdown": "# Sudah diaudit",
        "visual_audit_applied": True,
    }
    result = pipeline._node_refine_language(state)
    assert result["final_markdown"] == "# Sudah diaudit"
    assert result["language_refine_status"] == "skipped"


def test_pipeline_never_refines_text_before_visual_audit() -> None:
    from app.graph import DocumentExtractionPipeline, DocumentExtractionState

    language = MagicMock()
    with patch("app.learning_store.LearningStore") as store:
        store.return_value.active_run.return_value = None
        pipeline = DocumentExtractionPipeline(
            Settings(ocr_model="", language_vlm_model="agent-model"),
            vlm=MagicMock(),
            language_vlm=language,
        )

    state: DocumentExtractionState = {
        "markdown_content": "# Draft",
        "final_markdown": "# Draft",
        "visual_audit_applied": False,
    }
    result = pipeline._node_refine_language(state)
    assert result["language_refine_status"] == "skipped"
    language.invoke.assert_not_called()


def test_language_refiner_preserves_visual_audit_when_data_changes() -> None:
    audited = (
        "# Rekap 2026\n\n| Akun | Nilai |\n| --- | ---: |\n| A | 42 |\n\n"
        '```mermaid\nflowchart TD\n A["Mulai"] --> B["Selesai"]\n```'
    )
    language = MagicMock()
    language.invoke.return_value = _response(
        audited.replace("| A | 42 |", "| A | 43 |")
    )

    final, status = refine_audited_markdown(
        llm=language, draft_markdown=audited, audited_markdown=audited
    )
    assert status == "fallback_invalid"
    assert final == audited


def test_language_refiner_accepts_text_formatting_without_changing_facts() -> None:
    audited = (
        "# Rekap 2026\n\n| Akun | Nilai |\n| --- | ---: |\n| A | 42 |\n\n"
        '```mermaid\nflowchart TD\n A["Mulai"] --> B["Selesai"]\n```'
    )
    revised = audited.replace("# Rekap 2026", "## Rekap 2026")
    language = MagicMock()
    language.invoke.return_value = _response(revised)

    final, status = refine_audited_markdown(
        llm=language, draft_markdown=audited, audited_markdown=audited
    )

    assert (final, status) == (revised, "accepted")
    prompt = language.invoke.call_args.args[0]
    assert "HASIL AUDIT VISUAL" in prompt
    assert "image_url" not in prompt


@pytest.mark.parametrize(
    "revision",
    [
        '# Rekap 2026\n\n```mermaid\nflowchart TD\n A["Mulai"] --> C["Selesai"]\n```',
        "# Rekap 2026\n\nNilai berubah menjadi 43.",
        "Berikut adalah Markdown: # Rekap 2026",
    ],
)
def test_language_refiner_rejects_unsafe_revisions(revision: str) -> None:
    audited = (
        "# Rekap 2026\n\nNilai 42.\n\n"
        '```mermaid\nflowchart TD\n A["Mulai"] --> B["Selesai"]\n```'
    )
    language = MagicMock()
    language.invoke.return_value = _response(revision)

    final, status = refine_audited_markdown(
        llm=language, draft_markdown=audited, audited_markdown=audited
    )
    assert (final, status) == (audited, "fallback_invalid")


def test_language_refiner_skips_empty_visual_audit() -> None:
    language = MagicMock()
    assert refine_audited_markdown(
        llm=language, draft_markdown="draft", audited_markdown="  "
    ) == ("  ", "skipped_empty")
    language.invoke.assert_not_called()


def test_language_refiner_falls_back_when_agent_is_unavailable() -> None:
    audited = "# Rekap 2026\n\nNilai 42."
    language = MagicMock()
    language.invoke.side_effect = RuntimeError("server unavailable")
    final, status = refine_audited_markdown(
        llm=language, draft_markdown=audited, audited_markdown=audited
    )
    assert status == "fallback_error"
    assert final == audited


def test_mermaid_code_repair_uses_language_vlm_without_image(tmp_path) -> None:
    image = tmp_path / "diagram.png"
    Image.new("RGB", (300, 160), "white").save(image)
    visual = MagicMock()
    visual.invoke.side_effect = [
        _response(json.dumps({"is_convertible": True, "diagram_type": "flowchart",
                              "recommended_format": "mermaid"})),
        _response('```mermaid\nflowchart TD\n A["Mulai"] -->\n```'),
    ]
    language = MagicMock()
    language.invoke.return_value = _response(
        '```mermaid\nflowchart TD\n A["Mulai"] --> B["Selesai"]\n```'
    )
    with (
        patch("app.diagram.validate_mermaid_syntax", side_effect=[
            (False, "missing node"), (True, None)
        ]),
        patch("app.diagram.render_mermaid_to_png", return_value=(
            False, None, "mmdc executable not found"
        )),
    ):
        result = extract_diagram_to_mermaid(image, visual, language_llm=language)

    assert result.is_mermaid
    assert visual.invoke.call_count == 2
    language.invoke.assert_called_once()
    assert isinstance(language.invoke.call_args.args[0], str)
    assert "image_url" not in language.invoke.call_args.args[0]


def test_mermaid_repair_failure_does_not_publish_invalid_code(tmp_path) -> None:
    image = tmp_path / "diagram.png"
    Image.new("RGB", (300, 160), "white").save(image)
    visual = MagicMock()
    visual.invoke.side_effect = [
        _response(json.dumps({"is_convertible": True, "diagram_type": "flowchart",
                              "recommended_format": "mermaid", "reasoning": "Alur dua langkah."})),
        _response('```mermaid\nflowchart TD\n A["Mulai"] -->\n```'),
    ]
    language = MagicMock()
    language.invoke.side_effect = RuntimeError("server unavailable")
    with patch("app.diagram.validate_mermaid_syntax", return_value=(
        False, "missing node"
    )):
        result = extract_diagram_to_mermaid(image, visual, language_llm=language)

    assert result.status == "unsuitable"
    assert result.mermaid_code is None
    assert "tidak lolos validasi" in (result.text_summary or "")


def test_deep_agent_uses_language_vlm_for_master_and_subagents() -> None:
    from app.deep_agent import build_deep_agent

    visual = MagicMock()
    language = MagicMock()
    sentinel = object()
    with (
        patch("app.deep_agent.build_vlm", return_value=visual),
        patch("app.deep_agent.build_language_vlm", return_value=language),
        patch("app.deep_agent.DocumentExtractionPipeline") as pipeline_builder,
        patch("app.deep_agent.VisionExtractor"),
        patch("app.deep_agent.create_deep_agent", return_value=sentinel) as create,
    ):
        result = build_deep_agent(Settings(language_vlm_model="language-model"))

    assert result is sentinel
    assert create.call_args.kwargs["model"].runtime.models["agent"] is language
    assert create.call_args.kwargs["model"].role == "agent"
    assert len(create.call_args.kwargs["subagents"]) == 8
    assert all("model" not in agent for agent in create.call_args.kwargs["subagents"])
    assert pipeline_builder.call_args.kwargs["language_vlm"].runtime.models["agent"] is language


def test_deep_agent_falls_back_to_visual_model_without_agent_focus() -> None:
    from app.deep_agent import build_deep_agent

    visual = MagicMock()
    with (
        patch("app.deep_agent.build_vlm", return_value=visual),
        patch("app.deep_agent.build_language_vlm") as build_language,
        patch("app.deep_agent.DocumentExtractionPipeline"),
        patch("app.deep_agent.VisionExtractor"),
        patch("app.deep_agent.create_deep_agent", return_value=object()) as create,
    ):
        build_deep_agent(Settings(language_vlm_model=""))

    build_language.assert_not_called()
    assert create.call_args.kwargs["model"].runtime.models["vision"] is visual


def test_deep_agent_image_tools_stay_on_vision_focus(tmp_path) -> None:
    from app.deep_agent import build_deep_agent

    image = tmp_path / "page.png"
    Image.new("RGB", (100, 100), "white").save(image)
    visual = MagicMock()
    language = MagicMock()
    visual_extractor = MagicMock()
    visual_extractor.audit_markdown.return_value = SimpleNamespace(
        final_markdown="# Audited", action="accepted"
    )
    page_agent = MagicMock()
    page_agent.run.return_value = "# Extracted"
    pipeline = MagicMock()
    pipeline.ocr_extractor = None

    with (
        patch("app.deep_agent.build_vlm", return_value=visual),
        patch("app.deep_agent.build_language_vlm", return_value=language),
        patch("app.deep_agent.DocumentExtractionPipeline", return_value=pipeline),
        patch("app.deep_agent.VisionExtractor", return_value=visual_extractor),
        patch("app.deep_agent.get_agent", return_value=page_agent),
        patch("app.deep_agent.preprocess_image", return_value=SimpleNamespace(
            processed_path=str(image)
        )),
        patch("app.deep_agent.refine_audited_markdown", return_value=(
            "## Final", "accepted"
        )) as refine,
        patch("app.deep_agent.create_deep_agent") as create,
    ):
        build_deep_agent(Settings(language_vlm_model="agent-model"))
        tools_by_name = {
            item.name: item for item in create.call_args.kwargs["tools"]
        }
        extracted = tools_by_name["extract_to_markdown"].invoke({
            "image_path": str(image), "specs": "plain"
        })
        judged = tools_by_name["judge_and_refine_markdown"].invoke({
            "image_path": str(image),
            "draft_markdown": "# Draft",
            "specs": "plain",
        })

    assert extracted == "# Extracted"
    assert page_agent.run.call_args.kwargs["llm"].runtime.models["vision"] is visual
    visual_extractor.audit_markdown.assert_called_once()
    assert refine.call_args.kwargs["llm"].runtime.models["agent"] is language
    assert judged == "## Final"
    language.invoke.assert_not_called()


@pytest.mark.parametrize("agent_enabled", [False, True])
def test_ppt_tabular_reflection_prefers_agent_focus(tmp_path, agent_enabled: bool) -> None:
    from app.ppt import process_presentation_vision

    source = tmp_path / "slides.pptx"
    source.write_bytes(b"placeholder")
    visual = MagicMock()
    language = MagicMock() if agent_enabled else None
    pipeline = MagicMock()
    pipeline.vlm = visual
    pipeline.language_vlm = language
    pipeline.run.return_value = {"markdown_content": "# Slide 1", "specs": ["plain"]}

    with (
        patch(
            "app.ppt.render_presentation_slides_to_images",
            return_value=[tmp_path / "slide_0001.jpg"],
        ),
        patch("app.ppt.process_page_tabular_agent") as tabular,
        patch("app.ppt.prune_document_pages"),
        patch("app.ppt.cross_verify_dual_track"),
    ):
        process_presentation_vision(
            pptx_path=source,
            pipeline=pipeline,
            output_markdown_path=tmp_path / "slides.md",
        )

    assert tabular.call_args.kwargs["llm"] is (language or visual)


@pytest.mark.parametrize("route", ["visual", "agent", "explicit"])
def test_pdf_tabular_reflection_model_priority(tmp_path, route: str) -> None:
    from app.pdf import process_multipage_pdf

    source = tmp_path / "document.pdf"
    source.write_bytes(b"placeholder")
    visual = MagicMock()
    language = MagicMock() if route != "visual" else None
    explicit = MagicMock() if route == "explicit" else None
    pipeline = MagicMock()
    pipeline.vlm = visual
    pipeline.language_vlm = language
    pipeline.run.return_value = {"markdown_content": "# Page 1", "specs": ["plain"]}
    tabular_event = PageTabularEvent(page_number=1)

    with (
        patch("app.pdf.pdf_page_count", return_value=1),
        patch("app.pdf.extract_pdf_native_text_by_page", return_value={}),
        patch("app.pdf.pdf_to_images", return_value=[tmp_path / "page_0001.png"]),
        patch("app.pdf.process_page_tabular_agent", return_value=(tabular_event, None)) as tabular,
    ):
        process_multipage_pdf(
            source,
            pipeline=pipeline,
            llm=explicit,
            auto_tabular_db=True,
            db_path=tmp_path / "document.sqlite",
            output_markdown_path=tmp_path / "document.md",
        )

    assert tabular.call_args.kwargs["llm"] is (explicit or language or visual)


def test_mcp_diagram_tool_routes_visual_and_text_models(tmp_path) -> None:
    from app.mcp_server import extract_diagram_to_mermaid as mcp_extract_diagram

    image = tmp_path / "diagram.png"
    Image.new("RGB", (100, 100), "white").save(image)
    visual = MagicMock()
    language = MagicMock()
    diagram_result = MagicMock()
    diagram_result.model_dump.return_value = {"status": "success"}
    with (
        patch("app.mcp_server.get_settings", return_value=Settings(
            language_vlm_model="language-model"
        )),
        patch("app.mcp_server.build_vlm", return_value=visual),
        patch("app.mcp_server.build_language_vlm", return_value=language),
        patch("app.mcp_server.preprocess_image", return_value=SimpleNamespace(
            processed_path=str(image)
        )),
        patch("app.mcp_server.run_extract_diagram", return_value=diagram_result) as run,
    ):
        payload = mcp_extract_diagram(str(image))

    assert json.loads(payload)["status"] == "success"
    assert run.call_args.kwargs["llm"].runtime.models["vision"] is visual
    assert run.call_args.kwargs["language_llm"].runtime.models["agent"] is language


def test_mcp_diagram_tool_without_agent_focus_uses_vision_only(tmp_path) -> None:
    from app.mcp_server import extract_diagram_to_mermaid as mcp_extract_diagram

    image = tmp_path / "diagram.png"
    Image.new("RGB", (100, 100), "white").save(image)
    visual = MagicMock()
    diagram_result = MagicMock()
    diagram_result.model_dump.return_value = {"status": "success"}
    with (
        patch("app.mcp_server.get_settings", return_value=Settings(
            language_vlm_model=""
        )),
        patch("app.mcp_server.build_vlm", return_value=visual),
        patch("app.mcp_server.build_language_vlm") as build_language,
        patch("app.mcp_server.preprocess_image", return_value=SimpleNamespace(
            processed_path=str(image)
        )),
        patch("app.mcp_server.run_extract_diagram", return_value=diagram_result) as run,
    ):
        payload = mcp_extract_diagram(str(image))

    assert json.loads(payload)["status"] == "success"
    build_language.assert_not_called()
    assert run.call_args.kwargs["llm"].runtime.models["vision"] is visual
    assert run.call_args.kwargs["language_llm"] is None
