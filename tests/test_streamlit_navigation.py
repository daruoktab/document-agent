"""Exercise opening a batch and its document without running extraction."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from streamlit.testing.v1 import AppTest


class TestDocumentNavigation(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)
        document = self.output / "sample"
        document.mkdir()
        (document / "sample.md").write_text(
            "# Sample\n\n```mermaid\ngraph TD\nA-->B\n```\n"
            "```mermaid\ngraph TD\nB-->C\n```\n",
            encoding="utf-8",
        )

    def script_header(self) -> str:
        return f"""
from pathlib import Path
from unittest.mock import patch
import streamlit as st
from app import streamlit_logic as ui
output = Path({str(self.output)!r})
manager = ui.JobManager()
"""

    def test_selected_batch_opens_directly_and_document_button_navigates(self) -> None:
        script = self.script_header() + """
batch = {
    'id': 'batch1', 'name': 'Sample batch', 'created_at': '2026-09-23T00:00:00Z',
    'documents': [{'stem': 'sample', 'source_name': 'sample.pdf'}],
}
with patch.object(ui.JobManager, 'get_instance', return_value=manager):
    if st.session_state.get('workspace_page') == 'Dokumen':
        ui.render_document_workspace(st.session_state['selected_stem'], output)
    else:
        ui.render_history_workspace([], [batch], manager, output)
"""
        app = AppTest.from_string(script, default_timeout=20)
        app.session_state["selected_batch_id"] = "batch1"
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.subheader[0].value, "Rincian batch terpilih")
        self.assertFalse(any(x.key == "batch_search" for x in app.text_input))

        app.button(key="history_back_to_list").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any(x.key == "batch_search" for x in app.text_input))
        app.button(key="history_batch_batch1").click().run()
        app.button(key="batch_open_sample").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["workspace_page"], "Dokumen")
        self.assertIn("Hasil dokumen", app.subheader[0].value)
        app.button(key="back_to_batch_sample").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["workspace_page"], "Histori")
        self.assertEqual(app.session_state["selected_batch_id"], "batch1")
        self.assertEqual(app.subheader[0].value, "Rincian batch terpilih")

    def test_opening_document_defers_zip_and_full_document_diagrams(self) -> None:
        script = self.script_header() + """
st.session_state.setdefault('zip_calls', 0)
st.session_state.setdefault('diagram_calls', 0)
def build_zip(*args):
    st.session_state['zip_calls'] += 1
    return b'zip-test-data'
def render_diagram(*args, **kwargs):
    st.session_state['diagram_calls'] += 1
with patch.object(ui.JobManager, 'get_instance', return_value=manager), \\
     patch.object(ui, 'build_document_zip', side_effect=build_zip), \\
     patch.object(ui, 'render_mermaid_html', side_effect=render_diagram):
    ui.render_completed_document_view('sample', output)
"""
        app = AppTest.from_string(script, default_timeout=20).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["zip_calls"], 0)
        self.assertEqual(app.session_state["diagram_calls"], 0)
        app.button(key="prepare_document_sample").click().run()
        self.assertEqual(app.session_state["zip_calls"], 1)
        app.run()
        self.assertEqual(app.session_state["zip_calls"], 1)
        app.checkbox(key="show_diagram_sample").check().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["diagram_calls"], 1)


if __name__ == "__main__":
    unittest.main()
