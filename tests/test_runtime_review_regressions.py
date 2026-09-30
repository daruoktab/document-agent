"""Queue, MCP, configuration, OCR, and storage regressions (offline)."""

import io
import sqlite3
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException, UploadFile
from openpyxl import Workbook
from PIL import Image

from app import api, config
from app.agent_graph import AgentDocumentGraph
from app.batch import batch_extract_documents
from app.config import Settings
from app.excel import persist_excel_native_data, survey_excel_workbook
from app.ocr import UnlimitedOCRExtractor
from app.paddle_ocr import PaddleOCRVLExtractor
from app.schemas import OCRExtractionResult


@pytest.mark.parametrize("terminal", ["completed", "failed"])
def test_api_waits_for_queued_jobs(tmp_path, terminal):
    manager = MagicMock()
    manager.start_job.return_value = SimpleNamespace(job_id="1", status="queued")
    manager.get_job.side_effect = [
        SimpleNamespace(job_id="1", status="running"),
        SimpleNamespace(job_id="1", status=terminal),
    ]
    with (
        patch.object(api, "OUTPUT_DIR", tmp_path),
        patch.object(api.JobManager, "get_instance", return_value=manager),
        patch.object(api.time, "sleep"),
        patch.object(api, "build_result_zip", return_value=b"zip"),
    ):
        upload = UploadFile(filename="page.pdf", file=io.BytesIO(b"pdf"))
        if terminal == "completed":
            assert api.ingest(upload).body == b"zip"
        else:
            with pytest.raises(HTTPException) as exc:
                api.ingest(upload)
            assert exc.value.status_code == 500
    assert manager.get_job.call_count == 2


def test_advance_propagates_next_page_render_failure():
    graph = AgentDocumentGraph()
    with patch.object(
        graph,
        "_node_render_batch",
        return_value={"status": "error", "error": "renderer failed"},
    ):
        result = graph._node_advance_to_next(
            {"missing_pages": [2], "saved_pages": [1], "is_complete": False}
        )
    assert result["status"] == "error"
    assert result["is_complete"] is False
    assert result["error"] == "renderer failed"


@pytest.mark.parametrize("claimed_complete", [False, True])
def test_mcp_cannot_claim_completion_with_missing_pages(tmp_path, claimed_complete):
    from app.mcp_agent_server import submit_page_and_get_next

    graph = MagicMock()
    graph.advance_graph.invoke.return_value = {
        "status": "batch_saved",
        "is_complete": claimed_complete,
        "total_items": 2,
        "saved_pages": [1],
        "missing_pages": [2],
        "rendered_images": [],
    }
    with patch("app.agent_graph.get_agent_document_graph", return_value=graph):
        result = submit_page_and_get_next(
            "source.pdf", 1, "page one", output_dir=str(tmp_path)
        )
    assert isinstance(result, str) and result.startswith("ERROR:")


def test_process_alias_overrides_dotenv_new_name(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "VLM_VISION_FOCUS_MODEL=dotenv-model\nVLM_VISION_FOCUS_TIMEOUT=300\nOCR_TIMEOUT=123\n"
    )
    monkeypatch.setenv("VLM_MODEL", "service-model")
    monkeypatch.setenv("VLM_TIMEOUT", "45")
    config._load_local_dotenv(env)
    settings = Settings()
    assert settings.vlm_model == "service-model"
    assert settings.vlm_timeout == 45
    assert settings.ocr_timeout == 123
    assert "VLM_VISION_FOCUS_MODEL" not in config.os.environ


def test_auxiliary_application_settings_still_read_dotenv(tmp_path, monkeypatch):
    from app.job_tracker import get_max_concurrent_extractions
    from app.learning_store import learning_root
    from app.ppt import _find_libreoffice_binary

    binary = tmp_path / "soffice.exe"
    binary.touch()
    root = tmp_path / "learning"
    env = tmp_path / ".env"
    env.write_text(f"MAX_CONCURRENT_EXTRACTIONS=3\nLEARNING_DATA_DIR={root}\nLIBREOFFICE_BIN={binary}\n")
    monkeypatch.setattr(config, "_LOCAL_ENV_PATH", env)
    config._load_local_dotenv()
    assert get_max_concurrent_extractions() == 3
    assert learning_root() == root
    assert _find_libreoffice_binary() == str(binary)


def test_dotenv_reload_keeps_previous_snapshot_visible_until_read_completes(tmp_path):
    env = tmp_path / ".env"
    env.write_text("VLM_VISION_FOCUS_MODEL=old-model\n")
    config._load_local_dotenv(env)
    env.write_text("VLM_VISION_FOCUS_MODEL=new-model\n")
    original_read = Path.read_text

    def read_with_concurrent_settings(path, **kwargs):
        assert Settings().vlm_model == "old-model"
        return original_read(path, **kwargs)

    with patch.object(Path, "read_text", read_with_concurrent_settings):
        config._load_local_dotenv(env)
    assert Settings().vlm_model == "new-model"


@pytest.mark.parametrize("backend", ["paddle", "legacy"])
def test_ocr_outage_does_not_retry_rotations(tmp_path, backend):
    image = tmp_path / "page.png"
    Image.new("RGB", (100, 100), "black").save(image)
    extractor = (
        PaddleOCRVLExtractor(base_url="unused", model_name="ocr")
        if backend == "paddle"
        else UnlimitedOCRExtractor(MagicMock(), model_name="ocr", prompt="read")
    )
    with patch.object(
        extractor,
        "extract",
        return_value=OCRExtractionResult(status="error", decision="error"),
    ) as call:
        result = extractor.extract_robust(image)
    call.assert_called_once()
    assert result.status == "error"


def test_paddle_configured_timeout_replaces_vendor_default():
    client = MagicMock()
    original_sdk = client._client

    class Recognizer:
        _genai_client = client

        @property
        def genai_client(self):
            return self._genai_client

    recognizer = Recognizer()
    pipeline = SimpleNamespace(
        paddlex_pipeline=SimpleNamespace(vl_rec_model=recognizer)
    )
    extractor = PaddleOCRVLExtractor(
        base_url="unused", model_name="ocr", timeout=17, pipeline=pipeline
    )
    extractor._get_pipeline()
    recognizer.genai_client.create_chat_completion([], timeout=600, return_future=True)
    assert client.create_chat_completion.call_args.kwargs["timeout"] == 17
    assert client.create_chat_completion.call_args.kwargs["return_future"] is True
    original_sdk.with_options.assert_called_once_with(timeout=17, max_retries=0)
    extractor._get_pipeline()
    client._client.with_options.assert_not_called()  # SDK clone must not be cloned again


def _workbook(path: Path, header: str):
    book = Workbook()
    sheet = book.active
    sheet.title = "Data"
    sheet.append(["Nama", header])
    for index in range(7):
        sheet.append([f"Item {index}", index + 1])
    book.save(path)
    book.close()


@pytest.mark.parametrize("other_source", [False, True])
def test_excel_schema_change_removes_stale_source_rows(tmp_path, other_source):
    source = tmp_path / "book.xlsx"
    target = tmp_path / "book.sqlite"
    _workbook(source, "Jumlah")
    first = persist_excel_native_data(survey_excel_workbook(source), target)
    assert len(first) == 1
    if other_source:
        other = tmp_path / "other.xlsx"
        _workbook(other, "Jumlah")
        persist_excel_native_data(survey_excel_workbook(other), target)
    _workbook(source, "Nilai")
    second = persist_excel_native_data(survey_excel_workbook(source), target)
    assert second != first
    with closing(sqlite3.connect(target)) as connection:
        old_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name = ?", (first[0],)
        ).fetchone()
        assert bool(old_exists) is other_source
        if other_source:
            assert (
                connection.execute(
                    f'SELECT count(*) FROM "{first[0]}" WHERE source_file = ?',
                    (str(source.resolve()),),
                ).fetchone()[0]
                == 0
            )
            assert (
                connection.execute(f'SELECT count(*) FROM "{first[0]}"').fetchone()[0]
                == 7
            )
        assert (
            connection.execute(f'SELECT count(*) FROM "{second[0]}"').fetchone()[0] == 7
        )


def test_batch_same_names_use_distinct_stable_output_paths(tmp_path):
    sources = [tmp_path / "a" / "report.pdf", tmp_path / "b" / "report.pdf"]
    for source in sources:
        source.parent.mkdir()
        source.write_bytes(b"placeholder")
    with (
        patch("app.batch.DocumentExtractionPipeline"),
        patch(
            "app.batch.process_multipage_pdf",
            return_value=SimpleNamespace(markdown_content="text"),
        ) as process,
    ):
        batch_extract_documents(
            sources, output_dir=tmp_path / "out", settings=Settings()
        )
        outputs = [
            call.kwargs["output_markdown_path"] for call in process.call_args_list
        ]
    assert len(outputs) == 2 and outputs[0] != outputs[1]


def test_excel_refresh_rolls_back_when_an_insert_fails(tmp_path):
    source, target = tmp_path / "book.xlsx", tmp_path / "book.sqlite"
    _workbook(source, "Jumlah")
    survey = survey_excel_workbook(source)
    tables = persist_excel_native_data(survey, target)
    with closing(sqlite3.connect(target)) as connection, connection:
        connection.execute(
            f'CREATE TRIGGER fail_refresh BEFORE INSERT ON "{tables[0]}" '
            "BEGIN SELECT RAISE(ABORT, 'test failure'); END"
        )
    with pytest.raises(sqlite3.IntegrityError):
        persist_excel_native_data(survey, target)
    with closing(sqlite3.connect(target)) as connection:
        assert (
            connection.execute(f'SELECT count(*) FROM "{tables[0]}"').fetchone()[0] == 7
        )
        assert connection.execute("SELECT count(*) FROM excel_cells").fetchone()[0] > 0
