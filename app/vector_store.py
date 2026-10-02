"""
Modul Persistent Vector Store & Retrieval untuk Dokumen RAG.

Menyediakan:
  - DocumentVectorStore       : Penyimpan vektor per-dokumen di output/{doc_stem}/chroma/
  - DeterministicMockEmbeddings: Offline fallback embedding deterministik berbasis token hash
  - SafeFallbackEmbeddings    : Wrapper yang mencoba OpenAIEmbeddings dan otomatis fallback jika offline
  - index_markdown_document   : Fungsi integrasi pemotongan Markdown dan indexing ke vector store
  - query_document_knowledge_base: Helper query semantik ke knowledge base dokumen
  - get_document_vector_store : Factory helper untuk memuat atau membuat vector store dokumen
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore

from .config import Settings, get_settings
from .llm import build_embeddings
from .rag import (
    BaseMultimodalEmbeddingService,
    BaseVectorStore,
    ChunkItem,
    DeterministicMultimodalMockEmbeddings,
    RetrievalResult,
    chunk_multimodal_pages,
    preview_markdown_chunks,
)

logger = logging.getLogger("app.vector_store")


# ==============================================================================
# 1. Fallback & Mock Embeddings (Offline / Sandbox Resilience)
# ==============================================================================


class DeterministicMockEmbeddings(DeterministicMultimodalMockEmbeddings):
    """
    Offline embedding deterministik berbasis representasi hash token dan fitur visual.
    Menghasilkan vektor ter-normalisasi unit berdimensi n untuk pengujian dan
    situasi saat endpoint embedding lokal belum aktif.
    """

    def _embed_single(self, text: str) -> list[float]:
        return self._embed_components(text)


class SafeFallbackEmbeddings(BaseMultimodalEmbeddingService):
    """
    Embeddings wrapper yang memprioritaskan endpoint utama (OpenAIEmbeddings)
    dan otomatis beralih ke DeterministicMockEmbeddings bila koneksi gagal atau offline.
    Mendukung delegasi multimodal embedding.
    """

    def __init__(
        self,
        primary_embeddings: Embeddings,
        fallback_embeddings: Embeddings | None = None,
        *,
        warn_on_fallback: bool = True,
    ) -> None:
        self.primary = primary_embeddings
        self.fallback = fallback_embeddings or DeterministicMockEmbeddings()
        self.warn_on_fallback = warn_on_fallback
        self._fallback_active = False

    def embed_text(self, text: str) -> list[float]:
        return self.embed_query(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    def embed_multimodal(
        self, text: str, image_path: str | Path | None = None
    ) -> list[float]:
        if self._fallback_active:
            if isinstance(self.fallback, BaseMultimodalEmbeddingService):
                return self.fallback.embed_multimodal(text, image_path)
            return self.fallback.embed_query(text)
        try:
            if isinstance(self.primary, BaseMultimodalEmbeddingService):
                return self.primary.embed_multimodal(text, image_path)
            if isinstance(self.fallback, BaseMultimodalEmbeddingService):
                return self.fallback.embed_multimodal(text, image_path)
            return self.primary.embed_query(text)
        except Exception as exc:  # noqa: BLE001
            if not self._fallback_active:
                if self.warn_on_fallback:
                    logger.warning(
                        "Primary embedding endpoint tidak dapat dihubungi (%s: %s). "
                        "Beralih ke offline mock embeddings.",
                        type(exc).__name__,
                        exc,
                    )
                self._fallback_active = True
            if isinstance(self.fallback, BaseMultimodalEmbeddingService):
                return self.fallback.embed_multimodal(text, image_path)
            return self.fallback.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._fallback_active:
            return self.fallback.embed_documents(texts)
        try:
            return self.primary.embed_documents(texts)
        except Exception as exc:  # noqa: BLE001
            if not self._fallback_active:
                if self.warn_on_fallback:
                    logger.warning(
                        "Primary embedding endpoint tidak dapat dihubungi (%s: %s). "
                        "Beralih ke offline mock embeddings.",
                        type(exc).__name__,
                        exc,
                    )
                self._fallback_active = True
            return self.fallback.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        if self._fallback_active:
            return self.fallback.embed_query(text)
        try:
            return self.primary.embed_query(text)
        except Exception as exc:  # noqa: BLE001
            if not self._fallback_active:
                if self.warn_on_fallback:
                    logger.warning(
                        "Primary embedding endpoint tidak dapat dihubungi (%s: %s). "
                        "Beralih ke offline mock embeddings.",
                        type(exc).__name__,
                        exc,
                    )
                self._fallback_active = True
            return self.fallback.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._fallback_active:
            return await self.fallback.aembed_documents(texts)
        try:
            return await self.primary.aembed_documents(texts)
        except Exception as exc:  # noqa: BLE001
            if not self._fallback_active:
                if self.warn_on_fallback:
                    logger.warning(
                        "Primary embedding endpoint tidak dapat dihubungi (%s: %s). "
                        "Beralih ke offline mock embeddings.",
                        type(exc).__name__,
                        exc,
                    )
                self._fallback_active = True
            return await self.fallback.aembed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        if self._fallback_active:
            return await self.fallback.aembed_query(text)
        try:
            return await self.primary.aembed_query(text)
        except Exception as exc:  # noqa: BLE001
            if not self._fallback_active:
                if self.warn_on_fallback:
                    logger.warning(
                        "Primary embedding endpoint tidak dapat dihubungi (%s: %s). "
                        "Beralih ke offline mock embeddings.",
                        type(exc).__name__,
                        exc,
                    )
                self._fallback_active = True
            return await self.fallback.aembed_query(text)


def create_safe_embeddings(
    settings: Settings | None = None,
    explicit_embeddings: Embeddings | None = None,
) -> Embeddings:
    """Buat instance embedding aman dengan fallback otomatis ke offline mock."""
    if explicit_embeddings is not None:
        return explicit_embeddings
    primary = build_embeddings(settings)
    return SafeFallbackEmbeddings(primary)


# ==============================================================================
# 2. Konversi Model Dokumen & Chunk
# ==============================================================================


def chunk_to_document(chunk: ChunkItem, doc_stem: str = "document") -> Document:
    """Ubah ChunkItem menjadi objek Document LangChain dengan metadata lengkap multimodal."""
    meta = dict(chunk.metadata)
    meta["chunk_id"] = chunk.chunk_id
    meta["doc_stem"] = doc_stem
    meta["char_count"] = chunk.char_count
    meta["token_estimate"] = chunk.token_estimate
    meta["start_char"] = chunk.start_char
    meta["end_char"] = chunk.end_char
    if getattr(chunk, "image_path", None) is not None:
        meta["image_path"] = str(chunk.image_path)
    if getattr(chunk, "page_number", None) is not None:
        meta["page_number"] = chunk.page_number
    if getattr(chunk, "image_metadata", None):
        meta["image_metadata"] = chunk.image_metadata
        try:
            meta["image_metadata_json"] = json.dumps(
                chunk.image_metadata, ensure_ascii=False
            )
        except (TypeError, ValueError):
            meta["image_metadata_json"] = str(chunk.image_metadata)
    return Document(
        page_content=chunk.content,
        metadata=meta,
        id=f"{doc_stem}_chunk_{chunk.chunk_id}",
    )


def document_to_retrieval_result(doc: Document, score: float = 0.0) -> RetrievalResult:
    """Ubah Document LangChain menjadi RetrievalResult standar pipeline multimodal."""
    chunk_id = doc.metadata.get("chunk_id", doc.id or "0")
    img_meta = doc.metadata.get("image_metadata")
    if isinstance(img_meta, str):
        try:
            img_meta = json.loads(img_meta)
        except (ValueError, json.JSONDecodeError):
            img_meta = {"raw": img_meta}
    elif not isinstance(img_meta, dict):
        img_meta_json = doc.metadata.get("image_metadata_json")
        if isinstance(img_meta_json, str):
            try:
                img_meta = json.loads(img_meta_json)
            except (ValueError, json.JSONDecodeError):
                img_meta = {}
        else:
            img_meta = {}

    return RetrievalResult(
        chunk_id=chunk_id,
        content=doc.page_content,
        score=score,
        metadata=dict(doc.metadata),
        page_number=doc.metadata.get("page_number"),
        image_path=doc.metadata.get("image_path"),
        image_metadata=img_meta,
    )


# ==============================================================================
# 3. Persistent Document Vector Store
# ==============================================================================


class DocumentVectorStore(BaseVectorStore):
    """
    Penyimpan vektor per-dokumen terisolasi (output/{doc_stem}/chroma/).
    Mendukung ChromaDB jika tersedia, dengan fallback otomatis ke
    InMemoryVectorStore dengan persistensi file JSON lokal yang tahan restart.
    """

    def __init__(
        self,
        doc_stem: str = "document",
        persist_directory: str | Path | None = None,
        embeddings: Embeddings | None = None,
        *,
        prefer_chroma: bool = True,
        settings: Settings | None = None,
    ) -> None:
        self.doc_stem = doc_stem
        self.settings = settings or get_settings()
        self.embeddings = create_safe_embeddings(self.settings, embeddings)

        if persist_directory is not None:
            self.persist_directory = Path(persist_directory).resolve()
        else:
            self.persist_directory = (Path("output") / doc_stem / "chroma").resolve()
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.store: VectorStore
        self.backend_type: str = "in_memory_json"

        chroma_initialized = False
        if prefer_chroma:
            try:
                import importlib

                importlib.import_module("chromadb")
                chroma_mod = importlib.import_module("langchain_chroma")
                chroma_cls = chroma_mod.Chroma

                safe_collection = f"doc_{doc_stem}".replace("-", "_").replace(".", "_")[
                    :63
                ]
                self.store = chroma_cls(
                    collection_name=safe_collection,
                    embedding_function=self.embeddings,
                    persist_directory=str(self.persist_directory),
                )
                self.backend_type = "chroma"
                chroma_initialized = True
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "ChromaDB tidak aktif atau belum terpasang (%s). Menggunakan persistent JSON store.",
                    exc,
                )

        if not chroma_initialized:
            json_file = self.persist_directory / "vector_store.json"
            if json_file.is_file() and json_file.stat().st_size > 0:
                try:
                    self.store = InMemoryVectorStore.load(
                        str(json_file), embedding=self.embeddings
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Gagal memuat vector_store.json (%s). Membuat vector store baru.",
                        exc,
                    )
                    self.store = InMemoryVectorStore(embedding=self.embeddings)
            else:
                self.store = InMemoryVectorStore(embedding=self.embeddings)
            self.backend_type = "in_memory_json"

    def add_chunks(
        self,
        chunks: list[ChunkItem],
        embeddings: list[list[float]] | None = None,
    ) -> list[str]:
        """Simpan sekumpulan chunk dokumen ke vector database dan persist ke disk."""
        if not chunks:
            return []
        docs = [chunk_to_document(c, doc_stem=self.doc_stem) for c in chunks]
        ids = [doc.id for doc in docs if doc.id]
        id_list = (
            ids
            if len(ids) == len(docs)
            else [doc.id or f"{self.doc_stem}_chunk_{idx}" for idx, doc in enumerate(docs, 1)]
        )

        if embeddings is not None:
            if len(embeddings) != len(docs):
                msg = (
                    f"Jumlah vektor embeddings ({len(embeddings)}) harus sama dengan "
                    f"jumlah dokumen/chunk ({len(docs)})."
                )
                raise ValueError(msg)

            if isinstance(self.store, InMemoryVectorStore):
                for doc, vec, doc_id in zip(docs, embeddings, id_list, strict=False):
                    self.store.store[doc_id] = {
                        "id": doc_id,
                        "vector": [float(x) for x in vec],
                        "text": doc.page_content,
                        "metadata": doc.metadata,
                    }
                self.persist()
                return id_list

            try:
                added_ids = self.store.add_documents(
                    docs, ids=id_list, embeddings=embeddings
                )
                self.persist()
                return added_ids
            except TypeError:
                pass
            except Exception as exc:  # noqa: BLE001
                logger.debug("add_documents with embeddings failed: %s", exc)

        if self.backend_type == "chroma":
            # Sanitize metadata for Chroma: values must be str, int, float, bool
            sanitized_docs = []
            for d in docs:
                clean_meta = {}
                for k, v in d.metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        clean_meta[k] = v
                    elif v is not None:
                        clean_meta[k] = json.dumps(v, ensure_ascii=False)
                sanitized_docs.append(
                    Document(page_content=d.page_content, metadata=clean_meta, id=d.id)
                )
            docs = sanitized_docs

        added_ids = self.store.add_documents(
            docs, ids=id_list if len(id_list) == len(docs) else None
        )
        self.persist()
        return added_ids

    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        """Simpan dokumen LangChain ke vector store dan persist ke disk."""
        if not documents:
            return []
        added_ids = self.store.add_documents(documents, **kwargs)
        self.persist()
        return added_ids

    def persist(self) -> None:
        """Simpan state vector store ke disk."""
        if self.backend_type == "in_memory_json":
            json_file = self.persist_directory / "vector_store.json"
            if isinstance(self.store, InMemoryVectorStore):
                self.store.dump(str(json_file))
        else:
            persist_fn = getattr(self.store, "persist", None)
            if callable(persist_fn):
                persist_fn()

    def clear(self) -> None:
        """Hapus seluruh dokumen dan reset vector store."""
        if self.backend_type == "in_memory_json":
            self.store = InMemoryVectorStore(embedding=self.embeddings)
            json_file = self.persist_directory / "vector_store.json"
            if json_file.is_file():
                try:
                    json_file.unlink()
                except OSError:
                    pass
        elif hasattr(self.store, "delete_collection"):
            delete_fn = getattr(self.store, "delete_collection", None)
            if callable(delete_fn):
                try:
                    delete_fn()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Gagal delete_collection Chroma (%s).", exc)

    def search(
        self,
        query: str | list[float],
        top_k: int = 4,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Cari potongan teks dengan similaritas kosinus tertinggi terhadap query teks atau vektor."""
        if not query:
            return []
        if isinstance(query, str) and not query.strip():
            return []

        filter_kwargs: dict[str, Any] = {}
        if filter_metadata:
            if self.backend_type == "in_memory_json":
                filter_kwargs["filter"] = lambda doc: all(
                    doc.metadata.get(k) == v for k, v in filter_metadata.items()
                )
            else:
                filter_kwargs["filter"] = filter_metadata

        if isinstance(query, list):
            search_vec_score_fn = (
                getattr(self.store, "similarity_search_with_score_by_vector", None)
                or getattr(
                    self.store, "similarity_search_by_vector_with_relevance_scores", None
                )
            )
            if callable(search_vec_score_fn):
                try:
                    docs_and_scores = search_vec_score_fn(
                        query, k=top_k, **filter_kwargs
                    )
                    return [
                        document_to_retrieval_result(d, score=float(s))
                        for d, s in docs_and_scores
                    ]
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "search_vec_score_fn error: %s", exc
                    )

            search_vec_fn = getattr(self.store, "similarity_search_by_vector", None)
            if callable(search_vec_fn):
                docs = search_vec_fn(query, k=top_k, **filter_kwargs)
                return [document_to_retrieval_result(d, score=1.0) for d in docs]
            return []

        if hasattr(self.store, "similarity_search_with_relevance_scores"):
            try:
                docs_and_scores = self.store.similarity_search_with_relevance_scores(
                    query, k=top_k, **filter_kwargs
                )
                return [
                    document_to_retrieval_result(d, score=float(s))
                    for d, s in docs_and_scores
                ]
            except Exception as exc:  # noqa: BLE001
                logger.debug("similarity_search_with_relevance_scores error: %s", exc)

        if hasattr(self.store, "similarity_search_with_score"):
            try:
                docs_and_scores = self.store.similarity_search_with_score(
                    query, k=top_k, **filter_kwargs
                )
                return [
                    document_to_retrieval_result(d, score=float(s))
                    for d, s in docs_and_scores
                ]
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "similarity_search_with_score gagal (%s), fallback ke similarity_search biasa.",
                    exc,
                )

        docs = self.store.similarity_search(query, k=top_k, **filter_kwargs)
        return [document_to_retrieval_result(d, score=1.0) for d in docs]

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        """Delegasi pencarian kemiripan langsung ke VectorStore LangChain."""
        return self.store.similarity_search(query, k=k, **kwargs)

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        """Pencarian kemiripan disertai skor relevansi."""
        if hasattr(self.store, "similarity_search_with_score"):
            return self.store.similarity_search_with_score(query, k=k, **kwargs)
        docs = self.store.similarity_search(query, k=k, **kwargs)
        return [(d, 1.0) for d in docs]

    def as_retriever(self, **kwargs: Any) -> Any:
        """Kembalikan instance LangChain Retriever."""
        return self.store.as_retriever(**kwargs)

    def count(self) -> int:
        """Jumlah dokumen atau chunk yang telah diindeks."""
        if isinstance(self.store, InMemoryVectorStore):
            return len(self.store.store)
        coll = getattr(self.store, "_collection", None)
        if coll is not None and hasattr(coll, "count"):
            count_fn = getattr(coll, "count", None)
            if callable(count_fn):
                try:
                    return int(count_fn())
                except Exception as exc:  # noqa: BLE001
                    logger.debug("coll.count() error: %s", exc)
        return 0


# ==============================================================================
# 4. Fungsi Integrasi Indexing & Knowledge Retrieval
# ==============================================================================


def index_markdown_document(
    markdown_text: str,
    doc_stem: str = "document",
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    page_images: (
        Mapping[int, str | Path]
        | Mapping[str, str | Path]
        | Sequence[str | Path]
        | None
    ) = None,
    persist_directory: str | Path | None = None,
    settings: Settings | None = None,
    embeddings: Embeddings | None = None,
    prefer_chroma: bool = True,
    clear_existing: bool = True,
) -> dict[str, Any]:
    """
    Pecah dokumen Markdown menjadi chunk-chunk hierarkis dan indeks ke vector store per-dokumen.

    Returns:
        dict ringkasan proses indexing (status, total_chunks, total_characters, persist_directory, backend).
    """
    resolved_settings = settings or get_settings()
    size = chunk_size or resolved_settings.rag_default_chunk_size
    overlap = chunk_overlap or resolved_settings.rag_default_chunk_overlap

    store = DocumentVectorStore(
        doc_stem=doc_stem,
        persist_directory=persist_directory,
        embeddings=embeddings,
        prefer_chroma=prefer_chroma,
        settings=resolved_settings,
    )

    if clear_existing:
        store.clear()

    if not markdown_text or not markdown_text.strip():
        return {
            "status": "empty",
            "doc_stem": doc_stem,
            "total_chunks": 0,
            "multimodal_chunks": 0,
            "total_characters": 0,
            "avg_chunk_size": 0.0,
            "chunk_size": size,
            "chunk_overlap": overlap,
            "persist_directory": str(store.persist_directory),
            "backend": store.backend_type,
            "chunk_ids": [],
        }

    preview = preview_markdown_chunks(
        markdown_text,
        source_file=doc_stem,
        chunk_size=size,
        chunk_overlap=overlap,
        page_images=page_images,
        doc_stem=doc_stem,
    )

    added_ids = store.add_chunks(preview.chunks)

    return {
        "status": "success",
        "doc_stem": doc_stem,
        "total_chunks": preview.total_chunks,
        "multimodal_chunks": preview.multimodal_chunks_count,
        "total_characters": preview.total_characters,
        "avg_chunk_size": preview.avg_chunk_size,
        "chunk_size": preview.chunk_size,
        "chunk_overlap": preview.chunk_overlap,
        "persist_directory": str(store.persist_directory),
        "backend": store.backend_type,
        "chunk_ids": added_ids,
    }


def index_document_pages(
    pages: list[Any],
    doc_stem: str = "document",
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    persist_directory: str | Path | None = None,
    settings: Settings | None = None,
    embeddings: Embeddings | None = None,
    prefer_chroma: bool = True,
    clear_existing: bool = True,
) -> dict[str, Any]:
    """
    Injeksi dan indeks sekumpulan DocumentPage / halaman dengan referensi citra visual.
    """
    resolved_settings = settings or get_settings()
    size = chunk_size or resolved_settings.rag_default_chunk_size
    overlap = chunk_overlap or resolved_settings.rag_default_chunk_overlap

    store = DocumentVectorStore(
        doc_stem=doc_stem,
        persist_directory=persist_directory,
        embeddings=embeddings,
        prefer_chroma=prefer_chroma,
        settings=resolved_settings,
    )

    if clear_existing:
        store.clear()

    if not pages:
        return {
            "status": "empty",
            "doc_stem": doc_stem,
            "total_chunks": 0,
            "multimodal_chunks": 0,
            "total_characters": 0,
            "avg_chunk_size": 0.0,
            "chunk_size": size,
            "chunk_overlap": overlap,
            "persist_directory": str(store.persist_directory),
            "backend": store.backend_type,
            "chunk_ids": [],
        }

    preview = chunk_multimodal_pages(
        pages,
        doc_stem=doc_stem,
        chunk_size=size,
        chunk_overlap=overlap,
    )

    added_ids = store.add_chunks(preview.chunks)

    return {
        "status": "success",
        "doc_stem": doc_stem,
        "total_chunks": preview.total_chunks,
        "multimodal_chunks": preview.multimodal_chunks_count,
        "total_characters": preview.total_characters,
        "avg_chunk_size": preview.avg_chunk_size,
        "chunk_size": preview.chunk_size,
        "chunk_overlap": preview.chunk_overlap,
        "persist_directory": str(store.persist_directory),
        "backend": store.backend_type,
        "chunk_ids": added_ids,
    }


def get_document_vector_store(
    doc_stem: str = "document",
    *,
    persist_directory: str | Path | None = None,
    settings: Settings | None = None,
    embeddings: Embeddings | None = None,
    prefer_chroma: bool = True,
) -> DocumentVectorStore:
    """Helper singleton/factory untuk mendapatkan atau membuat DocumentVectorStore."""
    return DocumentVectorStore(
        doc_stem=doc_stem,
        persist_directory=persist_directory,
        embeddings=embeddings,
        prefer_chroma=prefer_chroma,
        settings=settings,
    )


def query_document_knowledge_base(
    query: str,
    doc_stem: str = "document",
    top_k: int = 4,
    *,
    persist_directory: str | Path | None = None,
    settings: Settings | None = None,
    embeddings: Embeddings | None = None,
    filter_metadata: dict[str, Any] | None = None,
) -> list[RetrievalResult]:
    """Cari potongan teks relevan dari knowledge base dokumen berdasarkan kesamaan semantik query."""
    store = get_document_vector_store(
        doc_stem=doc_stem,
        persist_directory=persist_directory,
        settings=settings,
        embeddings=embeddings,
    )
    return store.search(query=query, top_k=top_k, filter_metadata=filter_metadata)
