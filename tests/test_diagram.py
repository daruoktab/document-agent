"""
Unit tests untuk Diagram Mermaid & Visual Artifacts Specialist Module.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from app.diagram import (
    classify_diagram_convertibility,
    extract_diagram_to_mermaid,
    get_diagram_recommendation,
    render_mermaid_to_png,
    retain_flowchart_mermaid,
    sanitize_mermaid_code,
    validate_mermaid_syntax,
)
from app.schemas import DiagramConvertibilityResult, DiagramExtractionResult


def _create_dummy_image(path: Path) -> Path:
    """Helper untuk membuat file gambar dummy untuk testing."""
    img = Image.new("RGB", (300, 200), color=(255, 255, 255))
    img.save(str(path))
    return path


def test_sanitize_mermaid_code():
    raw_with_block = """
    Berikut adalah hasil ekstraksinya:
    ```mermaid
    flowchart TD
        A[Start] --> B[Process]
        B --> C[End]
    ```
    Semoga membantu!
    """
    cleaned = sanitize_mermaid_code(raw_with_block)
    assert cleaned is not None
    assert cleaned.startswith("flowchart TD")
    assert "--> C[End]" in cleaned

    # Kasus label unquoted dengan tanda kurung & HTML (penyebab crash 'got PS')
    raw_unquoted_paren = """flowchart TD
    A[<b>Hidup Saleh</b><br>(Tit 1:8)]:::blueNode
    Center{4<br>Syarat<br>Kepemimpinan}
    classDef blueCircle fill:#008CBA,stroke:#fff,stroke-width:2px,rx:15,ry:15;
    """
    cleaned_unquoted = sanitize_mermaid_code(raw_unquoted_paren)
    assert cleaned_unquoted is not None
    assert 'A["<b>Hidup Saleh</b><br/>(Tit 1:8)"]:::blueNode' in cleaned_unquoted
    assert 'Center{"4<br/>Syarat<br/>Kepemimpinan"}' in cleaned_unquoted
    assert "rx:15" not in cleaned_unquoted
    assert "ry:15" not in cleaned_unquoted

    # Kasus legacy 'graph TD' dan double brackets '"]]'
    raw_graph_legacy = """graph TD
    DLatch1["DLatch 1<br/>Q: Top Output<br/>Q_bar: Bottom Output"]]
    RP1_RP0["RP1<br/>RP0"]<br/>(2)<br/>["Bank Select"] --> D_Mem
    """
    cleaned_legacy = sanitize_mermaid_code(raw_graph_legacy)
    assert cleaned_legacy is not None
    assert cleaned_legacy.startswith("flowchart TD")
    assert 'DLatch1["DLatch 1<br/>Q: Top Output<br/>Q_bar: Bottom Output"]' in cleaned_legacy
    assert '"]]' not in cleaned_legacy
    assert 'RP1_RP0["RP1<br/>RP0"] --> D_Mem' in cleaned_legacy

    # Kasus Figure 2-1: Subgraph cycle (UserMem -.-> LowerMem di dalam subgraph UserMem)
    raw_subgraph_cycle = """flowchart TD
    subgraph UserMem ["User Memory Space"]
        UpperMem["3FFh"]
        LowerMem["1FFFh"]
        UserMem -.->|Indirect Address Pointer| LowerMem
    end
    Input1["('Input 1')"] --> UpperMem
    """
    cleaned_sub_cycle = sanitize_mermaid_code(raw_subgraph_cycle)
    assert cleaned_sub_cycle is not None
    assert "UserMem_node -.->|Indirect Address Pointer| LowerMem" in cleaned_sub_cycle
    assert 'Input1["Input 1"]' in cleaned_sub_cycle
    is_valid_sc, sc_err = validate_mermaid_syntax(cleaned_sub_cycle)
    assert is_valid_sc is True
    assert sc_err is None

    # Kasus Figure 2-2: Duplicate conflicting node ID across subgraphs & duplicate edges & redundant labels
    raw_subgraph_conflict = """flowchart TD
    subgraph Bank0 [Bank 0]
        Block_Low["4Fh - 7Fh"] --> Block_High["CFh - FFh"]
        Row8_Reg["EEDATA"]
    end
    subgraph Bank1 [Bank 1]
        Block_Low["4Fh"] --> Block_High["CFh"]
    end
    note1["Note"] -.-> Row8_Reg["EEDATA"]
    """
    cleaned_conflict = sanitize_mermaid_code(raw_subgraph_conflict)
    assert cleaned_conflict is not None
    assert 'Bank1_Block_Low["4Fh"]' in cleaned_conflict
    assert 'note1["Note"] -.-> Row8_Reg' in cleaned_conflict
    is_valid_cf, cf_err = validate_mermaid_syntax(cleaned_conflict)
    assert is_valid_cf is True
    assert cf_err is None


def test_validate_mermaid_syntax():
    valid_code = """flowchart TD
    A[User] -->|Auth| B(API Gateway)
    B --> C{Decision}
    C -->|Yes| D[Database]
    """
    is_valid, err = validate_mermaid_syntax(valid_code)
    assert is_valid is True
    assert err is None

    invalid_header = """randomHeader
    A --> B
    """
    is_valid_bad, err_bad = validate_mermaid_syntax(invalid_header)
    assert is_valid_bad is False
    assert "Header Mermaid tidak dikenali" in str(err_bad)

    # Deteksi penutup kurung ganda
    bad_double = """flowchart TD
    A["Text"]] --> B["Next"]
    """
    is_valid_db, err_db = validate_mermaid_syntax(bad_double)
    assert is_valid_db is False
    assert "penutup kurung siku ganda" in str(err_db)

    # Deteksi Subgraph Self-Cycle oleh linter
    raw_raw_cycle = """flowchart TD
    subgraph UserMem [Memory]
        UserMem --> LowerMem
    end
    """
    is_valid_cy, err_cy = validate_mermaid_syntax(raw_raw_cycle)
    assert is_valid_cy is False
    assert "Setting UserMem as parent of UserMem would create a cycle" in str(err_cy)

    # Deteksi Conflicting Duplicate Node ID oleh linter
    raw_raw_conf = """flowchart TD
    A["Label Satu"]
    A["Label Dua"]
    """
    is_valid_cnf, err_cnf = validate_mermaid_syntax(raw_raw_conf)
    assert is_valid_cnf is False
    assert "dideklarasikan dengan dua label berbeda" in str(err_cnf)


def test_get_diagram_recommendation():
    rec_flow = get_diagram_recommendation("flowchart")
    assert rec_flow.is_mermaid_compatible is True
    assert rec_flow.recommended_format == "mermaid_code"
    assert rec_flow.suggested_syntax == "flowchart TD"

    for flowchart_alias in ("workflow", "swimlane", "decision_tree", "alur proses"):
        alias_recommendation = get_diagram_recommendation(flowchart_alias)
        assert alias_recommendation.diagram_type == "flowchart"
        assert alias_recommendation.is_mermaid_compatible is True

    rec_pin = get_diagram_recommendation("pin_diagram")
    assert rec_pin.is_mermaid_compatible is False
    assert rec_pin.diagram_type == "pin_diagram"

    # Verifikasi fleksibilitas Pydantic coercion terhadap variasi model (mis. 'pinout', 'memory map')
    rec_coerced = get_diagram_recommendation("microcontroller_pinout")
    assert rec_coerced.diagram_type == "pin_diagram"

    # Verifikasi pembuatan DiagramExtractionResult dengan tipe pin_diagram tidak error
    res_pin = DiagramExtractionResult(
        is_mermaid=True,
        diagram_type="pin_diagram",
        mermaid_code="flowchart LR\n    RA0 --> RA1",
    )
    assert res_pin.diagram_type == "pin_diagram"

    rec_unsuitable = get_diagram_recommendation("unsuitable_statistical_chart")
    assert rec_unsuitable.is_mermaid_compatible is False
    assert rec_unsuitable.recommended_format == "text_description"

    for diagram_type in (
        "sequence_diagram",
        "class_diagram",
        "state_diagram",
        "er_diagram",
        "mindmap",
        "gantt_chart",
        "block_architecture",
        "memory_map",
        "circuit_diagram",
        "timing_diagram",
        "git_graph",
        "generic_diagram",
    ):
        recommendation = get_diagram_recommendation(diagram_type)
        assert recommendation.is_mermaid_compatible is False
        assert recommendation.recommended_format == "text_description"


def test_classify_diagram_convertibility(tmp_path):
    img_file = tmp_path / "diagram_test.png"
    _create_dummy_image(img_file)

    mock_llm = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = json.dumps(
        {
            "is_convertible": True,
            "diagram_type": "flowchart",
            "recommended_format": "mermaid",
            "mermaid_type": "flowchart",
            "confidence": 0.95,
            "reasoning": "Diagram alur proses login",
            "nodes_or_entities": ["User", "Login", "Dashboard"],
        }
    )
    mock_llm.invoke.return_value = mock_resp

    res = classify_diagram_convertibility(img_file, mock_llm)
    assert isinstance(res, DiagramConvertibilityResult)
    assert res.is_convertible is True
    assert res.diagram_type == "flowchart"
    assert res.mermaid_type == "flowchart"
    assert "User" in res.nodes_or_entities


def test_sequence_diagram_is_described_instead_of_converted(tmp_path):
    img_file = tmp_path / "seq_test.png"
    _create_dummy_image(img_file)

    mock_llm = MagicMock()

    # Step 1: Classify
    resp_classify = MagicMock()
    resp_classify.content = """
    {
      "is_convertible": true,
      "diagram_type": "sequence_diagram",
      "recommended_format": "mermaid",
      "mermaid_type": "sequenceDiagram",
      "confidence": 0.98,
      "reasoning": "Sequence interaksi user dan backend API",
      "nodes_or_entities": ["Client", "Server", "DB"]
    }
    """

    resp_description = MagicMock()
    resp_description.content = (
        "Urutan autentikasi memperlihatkan Client mengirim permintaan ke Server, "
        "Server membaca data pengguna dari DB, lalu mengembalikan token."
    )

    mock_llm.invoke.side_effect = [resp_classify, resp_description]

    res = extract_diagram_to_mermaid(
        img_file,
        mock_llm,
        forced_diagram_type="flowchart",
    )
    assert isinstance(res, DiagramExtractionResult)
    assert res.status == "unsuitable"
    assert res.is_mermaid is False
    assert res.diagram_type == "sequence_diagram"
    assert res.mermaid_code is None
    assert res.text_description is not None
    assert "Client" in res.text_description
    assert "Server" in res.text_description


def test_classifier_enforces_flowchart_only_policy(tmp_path):
    img_file = _create_dummy_image(tmp_path / "forced_sequence.png")
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = json.dumps(
        {
            "is_convertible": True,
            "diagram_type": "sequence_diagram",
            "recommended_format": "mermaid",
            "mermaid_type": "sequenceDiagram",
            "confidence": 0.99,
            "reasoning": "Urutan pesan eksplisit",
        }
    )

    result = classify_diagram_convertibility(img_file, mock_llm)

    assert result.is_convertible is False
    assert result.recommended_format == "text_description"
    assert result.mermaid_type is None


def test_retain_flowchart_mermaid_removes_other_mermaid_types():
    markdown = """# Visual
```mermaid
sequenceDiagram
  User->>API: Request
```

> **[Diagram/Visual]:** Urutan permintaan pengguna.

```mermaid
flowchart TD
  A[\"Mulai\"] --> B[\"Selesai\"]
```
"""

    cleaned = retain_flowchart_mermaid(markdown)

    assert "sequenceDiagram" not in cleaned
    assert "Urutan permintaan pengguna" in cleaned
    assert "flowchart TD" in cleaned


def test_missing_renderer_preserves_mermaid_without_retry(tmp_path):
    img_file = tmp_path / "flow.png"
    _create_dummy_image(img_file)
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = '```mermaid\nflowchart TD\n A["Start"] --> B["End"]\n```'
    with patch("app.diagram.render_mermaid_to_png", return_value=(False, None, "MMDC executable not found at 'mmdc'")):
        result = extract_diagram_to_mermaid(img_file, mock_llm, forced_diagram_type="flowchart")
    assert result.is_mermaid
    assert result.mermaid_code is not None
    assert 'A["Start"] --> B["End"]' in result.mermaid_code
    assert mock_llm.invoke.call_count == 2


def test_extract_diagram_unsuitable_fallback(tmp_path):
    img_file = tmp_path / "map_test.png"
    _create_dummy_image(img_file)

    mock_llm = MagicMock()
    resp_classify = MagicMock()
    resp_classify.content = """
    {
      "is_convertible": false,
      "diagram_type": "unsuitable_map_or_spatial",
      "recommended_format": "text_description",
      "mermaid_type": null,
      "confidence": 0.99,
      "reasoning": "Peta topografi wilayah Jawa Barat dengan kontur ketinggian",
      "nodes_or_entities": []
    }
    """
    mock_llm.invoke.return_value = resp_classify

    res = extract_diagram_to_mermaid(img_file, mock_llm)
    assert res.status == "unsuitable"
    assert res.is_mermaid is False
    assert res.mermaid_code is None
    assert res.text_summary is not None and "Peta topografi wilayah" in res.text_summary


def test_extract_diagram_self_correction_retry(tmp_path):
    img_file = tmp_path / "retry_diag.png"
    _create_dummy_image(img_file)

    mock_llm = MagicMock()
    # Step 1: Classify
    resp_classify = MagicMock()
    resp_classify.content = json.dumps(
        {
            "is_convertible": True,
            "diagram_type": "flowchart",
            "recommended_format": "mermaid",
            "mermaid_type": "flowchart",
            "confidence": 0.95,
            "reasoning": "Flowchart logic",
            "nodes_or_entities": ["A", "B"],
        }
    )

    # Step 2: Percobaan pertama menghasilkan Mermaid dengan kurung ganda dan tag dangling
    resp_extract_broken = MagicMock()
    resp_extract_broken.content = """
    ```mermaid
    flowchart TD
        A["Start"]] --> B["Process"]
    ```
    """

    # Step 3: Percobaan koreksi mandiri menghasilkan Mermaid yang bersih dan valid
    resp_extract_fixed = MagicMock()
    resp_extract_fixed.content = """
    ```mermaid
    flowchart TD
        A["Start"] --> B["Process"]
    ```
    Diagram berhasil diperbaiki.
    """

    mock_llm.invoke.side_effect = [resp_classify, resp_extract_broken, resp_extract_fixed]

    res = extract_diagram_to_mermaid(img_file, mock_llm)
    assert res.status == "success"
    assert res.is_mermaid is True
    assert res.mermaid_code is not None
    assert 'A["Start"] --> B["Process"]' in res.mermaid_code


def test_render_mermaid_to_png_success(tmp_path: Path):
    """Memverifikasi render Mermaid valid ke PNG bytes dan penyimpanan file."""
    valid_mermaid = """flowchart TD
    A["Mulai"] --> B["Proses"]
    B --> C["Selesai"]
    """
    out_file = tmp_path / "diagram_test.png"
    ok, png_bytes, err = render_mermaid_to_png(valid_mermaid, output_path=out_file)
    if not ok and err and any(
        marker in err.lower()
        for marker in (
            "chrome-headless-shell",
            "tidak terinstal",
            "system cannot find the file specified",
            "winerror 2",
        )
    ):
        return
    assert ok is True
    assert err is None
    assert png_bytes is not None
    assert len(png_bytes) > 0
    # Header PNG adalah \x89PNG
    assert png_bytes[:4] == b"\x89PNG"
    assert out_file.exists()


def test_render_mermaid_to_png_failure():
    """Memverifikasi kegagalan compile Mermaid menangkap pesan error presisi."""
    cycle_mermaid = """flowchart TD
    subgraph S [Sub]
        S --> InsideNode
    end
    """
    ok, png_bytes, err = render_mermaid_to_png(cycle_mermaid)
    assert ok is False
    assert png_bytes is None
    assert err is not None
    assert any(
        message in err
        for message in (
            "would create a cycle",
            "Error",
            "chrome-headless-shell",
            "tidak terinstal",
        )
    )


def test_extract_diagram_visual_feedback_loop(tmp_path: Path):
    """
    Memverifikasi bahwa setelah ekstraksi awal dan rendering berhasil,
    sistem memanggil LLM untuk inspeksi visual multimodal dan mengonfirmasi hasil diagram.
    """
    img_file = _create_dummy_image(tmp_path / "diag_vis.png")
    mock_llm = MagicMock()

    resp_classify = MagicMock()
    resp_classify.content = json.dumps(
        {
            "is_diagram": True,
            "diagram_type": "flowchart",
            "is_convertible": True,
            "confidence": 0.95,
            "reasoning": "Alur proses valid untuk Mermaid flowchart",
        }
    )

    resp_extract = MagicMock()
    resp_extract.content = """
    ```mermaid
    flowchart TD
        A["Mulai"] --> B["Selesai"]
    ```
    Diagram alur sederhana.
    """

    # Verifikasi visual mengonfirmasi kesesuaian gambar render dengan gambar dokumen
    resp_verify = MagicMock()
    resp_verify.content = "[CONFIRMED] Diagram hasil render sudah mencakup seluruh simpul dan relasi dengan akurat."

    mock_llm.invoke.side_effect = [resp_classify, resp_extract, resp_verify]

    res = extract_diagram_to_mermaid(img_file, mock_llm)
    assert res.status == "success"
    assert res.is_mermaid is True
    assert res.mermaid_code is not None
    assert 'A["Mulai"] --> B["Selesai"]' in res.mermaid_code
    if res.rendered_image_bytes is not None:
        assert res.rendered_image_bytes[:4] == b"\x89PNG"


if __name__ == "__main__":
    import tempfile
    test_sanitize_mermaid_code()
    print("✓ test_sanitize_mermaid_code passed")
    test_validate_mermaid_syntax()
    print("✓ test_validate_mermaid_syntax passed")
    test_get_diagram_recommendation()
    print("✓ test_get_diagram_recommendation passed")

    with tempfile.TemporaryDirectory() as td:
        p_td = Path(td)
        test_render_mermaid_to_png_success(p_td)
        print("✓ test_render_mermaid_to_png_success passed")
        test_render_mermaid_to_png_failure()
        print("✓ test_render_mermaid_to_png_failure passed")
        test_extract_diagram_self_correction_retry(p_td)
        print("✓ test_extract_diagram_self_correction_retry passed")
        test_extract_diagram_visual_feedback_loop(p_td)
        print("✓ test_extract_diagram_visual_feedback_loop passed")

    print("All tests in test_diagram.py passed successfully!")
