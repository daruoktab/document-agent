"""
Registry profil prompt ekstraksi dokumen -> Markdown siap chunking.

PENTING: File ini BUKAN autonomous agent (tidak ada tool-calling,
tidak ada orkestrasi, tidak ada LLM yang memutuskan apa pun).
`DocumentExtractionAgent` hanyalah profil prompt deterministik:

    1 spec (mis. 'chat_transcript')
        -> 1 system prompt spesifik
        -> 1 panggilan VisionExtractor.extract_markdown()
        -> selesai

Dipakai oleh:
  - `app/graph.py` (pipeline LangGraph default) pada node extract_markdown
  - `app/deep_agent.py` sebagai tool 'extract_to_markdown' untuk sub-agent

Untuk autonomous orchestrator (Master + 7 Sub-Agent yang memilih tool sendiri),
lihat `app/deep_agent.py`.

Mendukung multi-spesifikasi komposit layout dokumen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.language_models.chat_models import BaseChatModel

from .extractor import VisionExtractor
from .prompts import (
    SPEC_METADATA,
    SYSTEM_DOCUMENT_EXTRACTOR,
    get_vision_system_prompt,
    normalize_specs,
)


@dataclass
class DocumentExtractionAgent:
    """Agent ekstraksi untuk satu atau kombinasi spesifikasi layout dokumen."""

    name: str
    description: str
    specs: list[str] = field(default_factory=lambda: ["plain"])
    system_prompt: str | None = None

    def __post_init__(self) -> None:
        self.specs = normalize_specs(self.specs)
        if self.system_prompt is None:
            self.system_prompt = get_vision_system_prompt(self.specs)

    @property
    def doc_type(self) -> str:
        """Alias spesifikasi utama untuk kompatibilitas."""
        return self.specs[0] if self.specs else "plain"

    def build(self, llm: BaseChatModel) -> VisionExtractor:
        """Bangun instance VisionExtractor yang dikonfigurasi dengan system prompt agent."""
        return VisionExtractor(
            llm=llm,
            system_prompt=self.system_prompt or SYSTEM_DOCUMENT_EXTRACTOR,
        )

    def run(
        self,
        image_path: str,
        llm: BaseChatModel,
        *,
        previous_page_context: str | None = None,
        native_text: str | None = None,
    ) -> str:
        """Jalankan ekstraksi Markdown komposit pada gambar input."""
        extractor = self.build(llm)
        return extractor.extract_markdown(
            image_path=image_path,
            specs=self.specs,
            previous_page_context=previous_page_context,
            native_text=native_text,
        )


AGENT_REGISTRY: dict[str, DocumentExtractionAgent] = {
    name: DocumentExtractionAgent(
        name=name,
        description=meta["description"],
        specs=[name],
    )
    for name, meta in SPEC_METADATA.items()
}


def get_agent(specs: list[str] | str | None = None) -> DocumentExtractionAgent:
    """
    Ambil atau bangun agent berdasarkan satu atau kombinasi nama spesifikasi.
    Mendukung format 'journal,hierarchy' atau ['bilingual_journal', 'markdown_hierarchy'].
    """
    normalized = normalize_specs(specs)

    # Jika spesifikasi tunggal dan terdaftar
    if len(normalized) == 1 and normalized[0] in AGENT_REGISTRY:
        return AGENT_REGISTRY[normalized[0]]

    # Jika kombinasi multi-spesifikasi
    combo_name = "+".join(normalized)
    desc = " + ".join(
        SPEC_METADATA[s]["description"] if s in SPEC_METADATA else s for s in normalized
    )
    return DocumentExtractionAgent(
        name=combo_name,
        description=f"Komposit: {desc}",
        specs=normalized,
    )
