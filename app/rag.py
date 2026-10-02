"""
MODUL ARSITEKTUR & INGESTI RAG MULTIMODAL (QWEN-VL READY).

Modul ini bertanggung jawab atas alur pemrosesan hulu Retrieval-Augmented Generation (RAG):
  1. Chunking Multimodal: Memotong teks Markdown secara hierarkis (H1-H3) sekaligus mengaitkan
     setiap potongan dengan referensi citra visual halaman (image path, image metadata, page number).
  2. Embedding Multimodal (Qwen-VL Ready): Arsitektur dan adapter embedding untuk teks dan citra
     visual halaman menggunakan model Vision-Language (Qwen-VL-Chat / Qwen2-VL), dilengkapi
     offline fallback deterministik berdimensi n.
  3. Dokumen RAG & Metadata Vektor: Skema penyimpanan dua dimensi metadata (konten teks + referensi citra)
     ke persistent vector database terisolasi per dokumen.
  4. Pencarian / Retrieval Semantik Multimodal: Menyediakan fungsi temu-kembali terstruktur yang
     mengembalikan teks relevan beserta nomor halaman dan path citra visual untuk dikonsumsi oleh
     chatbot eksternal maupun Deep Agent.

BATASAN & RUANG LINGKUP SISTEM:
  - Repositori ini HANYA berfokus pada INGESTION / INJECTION & EMBEDDING.
  - Alur percakapan chatbot, antarmuka chat, dan memori percakapan RAG berada di sistem chatbot utama
    (yang akan memanggil API/endpoint retrieval dari repo ini).
  - Deep Agent di repo ini hanya bertindak sebagai pencari konteks (tool retrieval dokumen) untuk
    kebutuhan penalaran ekstraksi, BUKAN chatbot percakapan interaktif.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings

logger = logging.getLogger("app.rag")


# ==============================================================================
# 1. Model Data Chunking & Dokumen RAG Multimodal
# ==============================================================================


@dataclass
class ChunkItem:
    """
    Representasi satu potongan (chunk) teks Markdown siap indeks,
    dengan dukungan multimodal (teks + referensi citra visual halaman).
    """

    chunk_id: int
    char_count: int
    token_estimate: int
    preview: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    start_char: int = 0
    end_char: int = 0
    page_number: int | None = None
    image_path: str | None = None
    image_metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    @property
    def is_multimodal(self) -> bool:
        """True jika chunk ini memiliki referensi citra visual halaman."""
        return self.image_path is not None and bool(str(self.image_path).strip())

    def to_dict(self) -> dict[str, Any]:
        """Konversi objek chunk ke dictionary standar serialisasi JSON."""
        return asdict(self)


@dataclass
class ChunkingPreview:
    """Hasil simulasi pembagian dokumen Markdown menjadi chunk-chunk multimodal terstruktur."""

    source_file: str
    total_characters: int
    total_chunks: int
    chunk_size: int
    chunk_overlap: int
    avg_chunk_size: float
    chunks: list[ChunkItem] = field(default_factory=list)
    multimodal_chunks_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Konversi pratinjau chunking ke dictionary JSON."""
        return {
            "source_file": self.source_file,
            "total_characters": self.total_characters,
            "total_chunks": self.total_chunks,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "avg_chunk_size": self.avg_chunk_size,
            "multimodal_chunks_count": self.multimodal_chunks_count,
            "chunks": [c.to_dict() for c in self.chunks],
        }


@dataclass
class RAGDocument:
    """Struktur dokumen siap indeks ke Vector Database (teks + citra visual + metadata)."""

    doc_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    page_number: int | None = None
    image_path: str | None = None
    image_metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalResult:
    """
    Hasil temu kembali dokumen/chunk dari Vector DB atau Reranker.
    Menyediakan teks relevan sekaligus referensi citra visual dan nomor halaman
    agar sistem chatbot eksternal dapat menampilkan cuplikan teks dan gambar halaman asal.
    """

    chunk_id: int | str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    page_number: int | None = None
    image_path: str | None = None
    image_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ==============================================================================
# 2. Engine Pemotongan Markdown Multimodal (Multimodal Chunking Engine)
# ==============================================================================

PAGE_DELIMITER_RE = re.compile(
    r"<!--\s*(?:PAGE|SLIDE|HALAMAN)\s*[:_-]?\s*(\d+)\s*-->|<!--\s*(?:Page|Slide|Halaman)\s*(\d+)\s*-->",
    re.IGNORECASE,
)


def _parse_page_key(key: Any) -> int | None:
    """Parse integer page number from int, string number, or label (e.g. 'page_1')."""
    if isinstance(key, int):
        return key
    s = str(key).strip()
    if s.isdigit():
        return int(s)
    m = re.search(r"\d+", s)
    if m:
        return int(m.group(0))
    return None


def _find_page_number_for_text(
    text: str, page_spans: Any = None
) -> int | None:
    """
    Tentukan nomor halaman berdasarkan kecocokan penanda dalam teks.
    """
    match = PAGE_DELIMITER_RE.search(text)
    if match:
        raw_num = match.group(1) or match.group(2)
        if raw_num and raw_num.isdigit():
            return int(raw_num)
    return None


def preview_markdown_chunks(
    markdown_content: str,
    *,
    source_file: str = "document",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    page_images: (
        Mapping[int, str | Path]
        | Mapping[str, str | Path]
        | Sequence[str | Path]
        | None
    ) = None,
    doc_stem: str | None = None,
) -> ChunkingPreview:
    """
    Simulasikan pemecahan dokumen Markdown menjadi chunk-chunk hierarkis multimodal.
    Mengaitkan nomor halaman dan path citra visual ke tiap chunk jika tersedia.

    Args:
        markdown_content: Konten teks dokumen berformat Markdown.
        source_file: Nama berkas dokumen asal.
        chunk_size: Target ukuran karakter per chunk.
        chunk_overlap: Ukuran overlap karakter antar chunk.
        page_images: Pemetaan opsional nomor halaman -> path citra visual halaman,
                     atau list path citra berurutan (indeks 0 = halaman 1).
        doc_stem: Identifier dokumen untuk resolusi otomatis citra halaman dari output/{stem}/pages/.
    """
    if not markdown_content or not markdown_content.strip():
        return ChunkingPreview(
            source_file=source_file,
            total_characters=0,
            total_chunks=0,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            avg_chunk_size=0.0,
            chunks=[],
            multimodal_chunks_count=0,
        )

    try:
        from langchain_text_splitters import (
            MarkdownHeaderTextSplitter,
            RecursiveCharacterTextSplitter,
        )
    except ImportError as exc:
        raise ImportError(
            "langchain_text_splitters diperlukan untuk menjalankan fungsi chunking. "
            "Instal dengan: uv pip install langchain-text-splitters"
        ) from exc

    # Persiapkan mapping citra per halaman
    image_lookup: dict[int, str] = {}
    if isinstance(page_images, Mapping):
        for k, v in page_images.items():
            parsed_k = _parse_page_key(k)
            if parsed_k is not None:
                image_lookup[parsed_k] = str(v)
    elif isinstance(page_images, Sequence) and not isinstance(page_images, (str, bytes)):
        for idx, p in enumerate(page_images, start=1):
            image_lookup[idx] = str(p)
    elif doc_stem:
        pages_dir = Path("output") / doc_stem / "pages"
        if pages_dir.is_dir():
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
                for p_file in pages_dir.glob(ext):
                    m = re.search(
                        r"(?:page|slide|halaman)[_-]?(\d+)",
                        p_file.stem,
                        re.IGNORECASE,
                    )
                    if m:
                        image_lookup[int(m.group(1))] = str(p_file.resolve())

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    # Level 1: Split berdasarkan heading hierarki struktur
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    header_splits = markdown_splitter.split_text(markdown_content)

    # Level 2: Split rekursif berbasis karakter & overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    final_docs = text_splitter.split_documents(header_splits)

    items: list[ChunkItem] = []
    current_page_num: int | None = 1

    for idx, doc in enumerate(final_docs, start=1):
        content = doc.page_content.strip()
        preview = content[:120].replace("\n", " ")

        # Deteksi penanda halaman di dalam chunk
        detected_page = _find_page_number_for_text(content, [])
        if detected_page is not None:
            current_page_num = detected_page

        matched_image_path = image_lookup.get(current_page_num) if current_page_num is not None else None
        image_meta: dict[str, Any] = {}
        if matched_image_path:
            image_meta = {
                "source": "page_canvas",
                "page_number": current_page_num,
                "path": matched_image_path,
            }

        item = ChunkItem(
            chunk_id=idx,
            char_count=len(content),
            token_estimate=max(1, len(content) // 4),
            preview=preview,
            content=content,
            metadata=dict(doc.metadata),
            page_number=current_page_num,
            image_path=matched_image_path,
            image_metadata=image_meta,
        )
        items.append(item)

    total_chars = sum(c.char_count for c in items)
    avg_size = total_chars / len(items) if items else 0.0
    multimodal_count = sum(1 for c in items if c.is_multimodal)

    return ChunkingPreview(
        source_file=source_file,
        total_characters=total_chars,
        total_chunks=len(items),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        avg_chunk_size=avg_size,
        chunks=items,
        multimodal_chunks_count=multimodal_count,
    )


def chunk_multimodal_pages(
    pages: list[Any],
    *,
    doc_stem: str = "document",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> ChunkingPreview:
    """
    Potong dokumen multi-halaman langsung dari daftar DocumentPage atau dict halaman,
    mempertahankan citra halaman visual dan nomor halaman pada setiap chunk.

    Args:
        pages: Daftar objek yang memiliki atribut `page_number`, `markdown_content`,
               dan opsional `image_path` (misalnya `DocumentPage` atau dict serupa).
        doc_stem: Identifier dokumen.
        chunk_size: Target ukuran karakter per chunk.
        chunk_overlap: Overlap karakter per chunk.
    """
    if not pages:
        return ChunkingPreview(
            source_file=doc_stem,
            total_characters=0,
            total_chunks=0,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            avg_chunk_size=0.0,
            chunks=[],
            multimodal_chunks_count=0,
        )

    all_chunks: list[ChunkItem] = []
    chunk_counter = 1

    try:
        from langchain_text_splitters import (
            MarkdownHeaderTextSplitter,
            RecursiveCharacterTextSplitter,
        )
    except ImportError as exc:
        raise ImportError(
            "langchain_text_splitters diperlukan untuk menjalankan fungsi chunking. "
            "Instal dengan: uv pip install langchain-text-splitters"
        ) from exc

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    for page in pages:
        page_num: int
        md_text: str
        img_path: str | None = None

        if hasattr(page, "page_number"):
            page_num = int(page.page_number)
            md_text = str(getattr(page, "markdown_content", "") or "")
            img_path = getattr(page, "image_path", None)
        elif isinstance(page, dict):
            page_num = int(page.get("page_number", 1))
            md_text = str(page.get("markdown_content") or page.get("content") or "")
            img_path = page.get("image_path")
        else:
            continue

        if not md_text.strip():
            continue

        header_splits = markdown_splitter.split_text(md_text)
        sub_docs = text_splitter.split_documents(header_splits)

        for doc in sub_docs:
            content = doc.page_content.strip()
            preview = content[:120].replace("\n", " ")
            image_meta: dict[str, Any] = {}
            if img_path:
                image_meta = {
                    "source": "page_canvas",
                    "page_number": page_num,
                    "path": str(img_path),
                }

            meta = dict(doc.metadata)
            meta["page_number"] = page_num
            meta["doc_stem"] = doc_stem

            all_chunks.append(
                ChunkItem(
                    chunk_id=chunk_counter,
                    char_count=len(content),
                    token_estimate=max(1, len(content) // 4),
                    preview=preview,
                    content=content,
                    metadata=meta,
                    page_number=page_num,
                    image_path=str(img_path) if img_path else None,
                    image_metadata=image_meta,
                )
            )
            chunk_counter += 1

    total_chars = sum(c.char_count for c in all_chunks)
    avg_size = total_chars / len(all_chunks) if all_chunks else 0.0
    multimodal_count = sum(1 for c in all_chunks if c.is_multimodal)

    return ChunkingPreview(
        source_file=doc_stem,
        total_characters=total_chars,
        total_chunks=len(all_chunks),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        avg_chunk_size=avg_size,
        chunks=all_chunks,
        multimodal_chunks_count=multimodal_count,
    )


# ==============================================================================
# 3. Layanan Embedding Multimodal & Qwen-VL Architecture
# ==============================================================================


class BaseEmbeddingService(Embeddings, ABC):
    """Blueprint interface dasar penyedia representasi vektor teks yang kompatibel LangChain Embeddings."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Ubah teks menjadi vektor representasi berdimensi n."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Ubah sekumpulan teks menjadi kumpulan vektor secara efisien."""

    # LangChain Embeddings Protocol
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_batch(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_text(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


class BaseMultimodalEmbeddingService(BaseEmbeddingService):
    """
    Blueprint interface penyedia representasi vektor multimodal (Teks + Citra Visual).
    Mendukung model Vision-Language seperti Qwen-VL-Chat / Qwen2-VL.
    """

    @abstractmethod
    def embed_multimodal(
        self, text: str, image_path: str | Path | None = None
    ) -> list[float]:
        """Ubah pasangan teks dan citra visual halaman menjadi vektor embedding bersama."""

    def embed_multimodal_batch(
        self, items: list[ChunkItem] | list[tuple[str, str | Path | None]]
    ) -> list[list[float]]:
        """Ubah sekumpulan pasangan teks & citra visual menjadi list vektor embedding."""
        vectors: list[list[float]] = []
        for item in items:
            if isinstance(item, ChunkItem):
                vectors.append(self.embed_multimodal(item.content, item.image_path))
            else:
                txt, img = item
                vectors.append(self.embed_multimodal(txt, img))
        return vectors


class DeterministicMultimodalMockEmbeddings(
    BaseMultimodalEmbeddingService
):
    """
    Offline embedding deterministik berbasis representasi hash token dan fitur visual citra.
    Menjamin 100% reliabilitas pengujian unit dan kekebalan terhadap koneksi jaringan offline.
    """

    def __init__(self, dimensions: int = 1536) -> None:
        self.dimensions = dimensions

    def _embed_components(
        self, text: str, image_path: str | Path | None = None
    ) -> list[float]:
        if (not text or not text.strip()) and not image_path:
            return [0.0] * self.dimensions

        vec = [0.0] * self.dimensions

        # Hash komponen teks
        tokens = text.lower().split()
        for idx, token in enumerate(tokens):
            h = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
            pos = h % self.dimensions
            val = ((h >> 8) % 1000) / 1000.0 - 0.5
            vec[pos] += val
            pos2 = (h >> 16) % self.dimensions
            vec[pos2] += 1.0 / (idx + 1)

        # Hash komponen visual citra (jika ada)
        if image_path:
            img_str = str(image_path)
            h_img = int(hashlib.sha256(img_str.encode("utf-8")).hexdigest(), 16)
            pos_img = (h_img >> 4) % self.dimensions
            val_img = ((h_img >> 12) % 1000) / 1000.0
            vec[pos_img] += val_img + 0.5

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            return [v / norm for v in vec]
        return [1.0 / math.sqrt(self.dimensions)] * self.dimensions

    def embed_text(self, text: str) -> list[float]:
        return self._embed_components(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_components(t) for t in texts]

    def embed_multimodal(
        self, text: str, image_path: str | Path | None = None
    ) -> list[float]:
        return self._embed_components(text, image_path)


class QwenVLEmbeddingService(BaseMultimodalEmbeddingService):
    """
    Layanan embedding multimodal Qwen-VL (Qwen-VL-Chat / Qwen2-VL).
    Dirancang untuk menghasilkan representasi vektor gabungan antara teks halaman
    dan citra visual halaman.

    Bila endpoint API belum aktif atau koneksi gagal, otomatis beralih ke
    DeterministicMultimodalMockEmbeddings secara transparan.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen2-VL-7B-Instruct",
        base_url: str | None = None,
        api_key: str | None = None,
        dimensions: int = 1536,
        fallback_embeddings: BaseMultimodalEmbeddingService | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.dimensions = dimensions
        self.fallback = (
            fallback_embeddings
            or DeterministicMultimodalMockEmbeddings(dimensions=dimensions)
        )
        self._fallback_active = False

    def format_multimodal_prompt(
        self, text: str, image_path: str | Path | None = None
    ) -> dict[str, Any]:
        """
        Format payload multimodal standar untuk Qwen-VL (OpenAI / vLLM compatible format).
        Menyiapkan struktur message yang memadukan elemen citra dan teks.
        """
        content_parts: list[dict[str, Any]] = []
        if image_path:
            content_parts.append(
                {"type": "image_url", "image_url": {"url": str(image_path)}}
            )
        content_parts.append({"type": "text", "text": text or ""})
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": content_parts}],
        }

    def embed_multimodal(
        self, text: str, image_path: str | Path | None = None
    ) -> list[float]:
        """
        Kirim request embedding ke endpoint Qwen-VL; fallback deterministik jika offline.
        """
        if self._fallback_active or not self.base_url:
            return self.fallback.embed_multimodal(text, image_path)

        try:
            url = f"{self.base_url.rstrip('/')}/embeddings"
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            input_data: Any = text
            if image_path:
                input_data = [self.format_multimodal_prompt(text, image_path)]

            payload_bytes = json.dumps(
                {"model": self.model, "input": input_data}
            ).encode("utf-8")
            req = urllib.request.Request(
                url, data=payload_bytes, headers=headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                resp_json = json.loads(resp.read().decode("utf-8"))
                vec = resp_json["data"][0]["embedding"]
                return [float(x) for x in vec]
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Endpoint Qwen-VL embedding gagal (%s). Beralih ke fallback deterministik.",
                exc,
            )
            self._fallback_active = True
            return self.fallback.embed_multimodal(text, image_path)

    def embed_text(self, text: str) -> list[float]:
        return self.embed_multimodal(text, image_path=None)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]


# ==============================================================================
# 4. Blueprint Interface Vector Store & Reranker
# ==============================================================================


class BaseVectorStore(ABC):
    """Blueprint interface konektor Vector Database (Chroma, InMemory JSON, FAISS)."""

    @abstractmethod
    def add_chunks(
        self,
        chunks: list[ChunkItem],
        embeddings: list[list[float]] | None = None,
    ) -> list[str] | None:
        """Simpan chunk teks beserta vektor embedding ke koleksi vector database."""

    @abstractmethod
    def search(
        self,
        query: str | list[float],
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Cari potongan teks dengan similaritas tertinggi terhadap query teks atau vektor."""


class BaseReranker(ABC):
    """Blueprint interface model Reranking (Cross-Encoder / Qwen-VL-Reranker)."""

    @abstractmethod
    def rerank(
        self, query: str, candidates: list[RetrievalResult], top_n: int = 3
    ) -> list[RetrievalResult]:
        """Urutkan ulang kandidat dokumen berdasarkan relevansi semantik multimodal."""


class QwenVLReranker(BaseReranker):
    """
    Reranker multimodal berbasis penalaran visual Qwen-VL.
    Mengevaluasi kesesuaian antara query pengguna, teks chunk, dan citra halaman dokumen.
    """

    def __init__(self, model: str = "Qwen/Qwen2-VL-7B-Instruct") -> None:
        self.model = model

    def rerank(
        self, query: str, candidates: list[RetrievalResult], top_n: int = 3
    ) -> list[RetrievalResult]:
        """
        Urutkan ulang hasil retrieval berdasarkan skor kesesuaian multimodal.
        Pada mode awal, mempertahankan urutan relevansi dengan pembobotan visual.
        """
        if not candidates:
            return []

        # Berikan bonus skor pada kandidat yang memiliki bukti visual citra halaman
        scored_candidates = []
        for cand in candidates:
            adjusted_score = cand.score
            if cand.image_path:
                adjusted_score = min(1.0, adjusted_score * 1.05)
            scored_candidates.append(
                RetrievalResult(
                    chunk_id=cand.chunk_id,
                    content=cand.content,
                    score=adjusted_score,
                    metadata=dict(cand.metadata),
                    page_number=cand.page_number,
                    image_path=cand.image_path,
                    image_metadata=dict(cand.image_metadata),
                )
            )

        scored_candidates.sort(key=lambda x: x.score, reverse=True)
        return scored_candidates[:top_n]


# ==============================================================================
# 5. Pipeline Ingesti Dokumen & Temu-Kembali Multimodal (Multimodal RAG Pipeline)
# ==============================================================================


class MultimodalRAGPipeline:
    """
    Pipeline Ingestion, Injection, & Retrieval Multimodal.

    ARSITEKTUR & RUANG LINGKUP:
      - Pipeline ini hanya bertindak sebagai penyedia injeksi data dokumen dan retrieval semantik.
      - Hasil retrieval menyediakan teks dan referensi citra visual halaman (`image_path`).
      - Pipeline ini BUKAN chatbot; antarmuka percakapan ditangani oleh chatbot eksternal via API.
    """

    def __init__(
        self,
        embedding_service: BaseMultimodalEmbeddingService | BaseEmbeddingService | None = None,
        vector_store: BaseVectorStore | None = None,
        reranker: BaseReranker | None = None,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.reranker = reranker

    def index_markdown_document(
        self,
        markdown_text: str,
        source_file: str = "document",
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        *,
        doc_stem: str | None = None,
        page_images: (
            Mapping[int, str | Path]
            | Mapping[str, str | Path]
            | Sequence[str | Path]
            | None
        ) = None,
    ) -> dict[str, Any]:
        """
        Injeksi dan indeks dokumen Markdown hasil ekstraksi:
          1. Pecah Markdown ke chunk-chunk hierarkis multimodal via preview_markdown_chunks().
          2. Generate embedding tiap chunk via self.embedding_service (bila tersedia).
          3. Simpan ke vector database via self.vector_store atau default DocumentVectorStore.
        """
        target_stem = doc_stem or source_file
        if self.vector_store is None and self.embedding_service is None:
            from .vector_store import index_markdown_document as default_index

            return default_index(
                markdown_text=markdown_text,
                doc_stem=target_stem,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                page_images=page_images,
            )

        preview = preview_markdown_chunks(
            markdown_text,
            source_file=target_stem,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            page_images=page_images,
            doc_stem=target_stem,
        )
        embeddings = None
        if self.embedding_service is not None:
            if isinstance(self.embedding_service, BaseMultimodalEmbeddingService):
                embeddings = self.embedding_service.embed_multimodal_batch(preview.chunks)
            else:
                texts = [c.content for c in preview.chunks]
                embeddings = self.embedding_service.embed_batch(texts)

        if self.vector_store is not None:
            added_ids = self.vector_store.add_chunks(preview.chunks, embeddings=embeddings)
            return {
                "status": "indexed",
                "source_file": target_stem,
                "doc_stem": target_stem,
                "total_chunks": preview.total_chunks,
                "multimodal_chunks": preview.multimodal_chunks_count,
                "total_characters": preview.total_characters,
                "chunk_ids": added_ids or [],
            }

        from .vector_store import DocumentVectorStore

        emb_for_store = (
            self.embedding_service
            if isinstance(self.embedding_service, Embeddings)
            else None
        )
        store = DocumentVectorStore(doc_stem=target_stem, embeddings=emb_for_store)
        added_ids = store.add_chunks(preview.chunks, embeddings=embeddings)
        return {
            "status": "indexed",
            "source_file": target_stem,
            "doc_stem": target_stem,
            "total_chunks": preview.total_chunks,
            "multimodal_chunks": preview.multimodal_chunks_count,
            "total_characters": preview.total_characters,
            "chunk_ids": added_ids or [],
        }

    def index_pages(
        self,
        pages: list[Any],
        doc_stem: str = "document",
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ) -> dict[str, Any]:
        """
        Injeksi dan indeks daftar halaman dokumen (DocumentPage) dengan preservasi citra visual.
        """
        if self.vector_store is None and self.embedding_service is None:
            from .vector_store import index_document_pages as default_index_pages

            return default_index_pages(
                pages=pages,
                doc_stem=doc_stem,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

        preview = chunk_multimodal_pages(
            pages,
            doc_stem=doc_stem,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        embeddings = None
        if self.embedding_service is not None:
            if isinstance(self.embedding_service, BaseMultimodalEmbeddingService):
                embeddings = self.embedding_service.embed_multimodal_batch(preview.chunks)
            else:
                texts = [c.content for c in preview.chunks]
                embeddings = self.embedding_service.embed_batch(texts)

        if self.vector_store is not None:
            added_ids = self.vector_store.add_chunks(preview.chunks, embeddings=embeddings)
            return {
                "status": "indexed",
                "doc_stem": doc_stem,
                "total_chunks": preview.total_chunks,
                "multimodal_chunks": preview.multimodal_chunks_count,
                "total_characters": preview.total_characters,
                "chunk_ids": added_ids or [],
            }

        from .vector_store import DocumentVectorStore

        emb_for_store = (
            self.embedding_service
            if isinstance(self.embedding_service, Embeddings)
            else None
        )
        store = DocumentVectorStore(doc_stem=doc_stem, embeddings=emb_for_store)
        added_ids = store.add_chunks(preview.chunks, embeddings=embeddings)
        return {
            "status": "indexed",
            "doc_stem": doc_stem,
            "total_chunks": preview.total_chunks,
            "multimodal_chunks": preview.multimodal_chunks_count,
            "total_characters": preview.total_characters,
            "chunk_ids": added_ids or [],
        }

    def query_context(
        self,
        query: str,
        top_k: int = 5,
        use_reranker: bool = True,
        doc_stem: str = "document",
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """
        Temu kembali potongan teks dan referensi citra visual dari vector store.
        Hasilnya siap diteruskan ke chatbot eksternal atau Deep Agent.
        """
        if (
            self.vector_store is None
            and self.embedding_service is None
            and (self.reranker is None or not use_reranker)
        ):
            from .vector_store import query_document_knowledge_base

            return query_document_knowledge_base(
                query=query,
                doc_stem=doc_stem,
                top_k=top_k,
                filter_metadata=filter_metadata,
            )

        store = self.vector_store
        if store is None:
            from .vector_store import get_document_vector_store

            emb_for_store = (
                self.embedding_service
                if isinstance(self.embedding_service, Embeddings)
                else None
            )
            store = get_document_vector_store(
                doc_stem=doc_stem,
                embeddings=emb_for_store,
            )

        query_input: str | list[float] = query
        if self.embedding_service is not None:
            try:
                query_input = self.embedding_service.embed_text(query)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Gagal embed_text query (%s), menggunakan teks mentah.", exc)
                query_input = query

        results = store.search(
            query_input, top_k=top_k, filter_metadata=filter_metadata
        )
        if self.reranker is not None and use_reranker:
            results = self.reranker.rerank(query, results, top_n=top_k)
        return results


# Backward compatibility alias
RAGPipelineBlueprint = MultimodalRAGPipeline


__all__ = [
    "PAGE_DELIMITER_RE",
    "BaseEmbeddingService",
    "BaseMultimodalEmbeddingService",
    "BaseReranker",
    "BaseVectorStore",
    "ChunkItem",
    "ChunkingPreview",
    "DeterministicMultimodalMockEmbeddings",
    "MultimodalRAGPipeline",
    "QwenVLEmbeddingService",
    "QwenVLReranker",
    "RAGDocument",
    "RAGPipelineBlueprint",
    "RetrievalResult",
    "chunk_multimodal_pages",
    "preview_markdown_chunks",
]
