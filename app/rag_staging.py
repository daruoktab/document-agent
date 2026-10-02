"""
MODUL TRANSISI & RE-EKSPOR ARSITEKTUR RAG (STAGING -> PRODUCTION APP.RAG).

Modul ini dipertahankan untuk backward-compatibility bagi komponen sistem yang
sebelumnya mengimpor dari `app.rag_staging`. Seluruh implementasi aktif kini
berada di `app.rag`.
"""

from __future__ import annotations

from .rag import (
    PAGE_DELIMITER_RE,
    BaseEmbeddingService,
    BaseMultimodalEmbeddingService,
    BaseReranker,
    BaseVectorStore,
    ChunkingPreview,
    ChunkItem,
    DeterministicMultimodalMockEmbeddings,
    MultimodalRAGPipeline,
    QwenVLEmbeddingService,
    QwenVLReranker,
    RAGDocument,
    RAGPipelineBlueprint,
    RetrievalResult,
    chunk_multimodal_pages,
    preview_markdown_chunks,
)

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
