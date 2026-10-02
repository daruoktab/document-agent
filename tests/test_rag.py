"""
Unit tests untuk Sistem RAG Dokumen, Chunking Engine, Embedding, dan Persistent Vector Store.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from app.config import Settings
from app.llm import build_embeddings
from app.rag_staging import (
    BaseReranker,
    ChunkItem,
    RAGPipelineBlueprint,
    RetrievalResult,
    preview_markdown_chunks,
)
from app.vector_store import (
    DeterministicMockEmbeddings,
    DocumentVectorStore,
    SafeFallbackEmbeddings,
    chunk_to_document,
    document_to_retrieval_result,
    index_markdown_document,
    query_document_knowledge_base,
)

# ==============================================================================
# 1. Chunking Engine Tests
# ==============================================================================


def test_preview_markdown_chunks_hierarchical_headers() -> None:
    content = """# Judul Dokumen Utama
Paragraf pembuka dokumen.

## Bab 1: Ketentuan Umum
Ketentuan umum menjelaskan definisi dan ruang lingkup kebijakan.

### Pasal 1.1: Definisi
Definisi operasional mencakup sistem RAG dan VLM.

## Bab 2: Tata Cara Pelaksanaan
Langkah-langkah teknis pemrosesan dokumen perusahaan.
"""
    preview = preview_markdown_chunks(
        content,
        source_file="kebijakan_internal",
        chunk_size=300,
        chunk_overlap=50,
    )

    assert preview.source_file == "kebijakan_internal"
    assert preview.total_chunks >= 3
    assert preview.total_characters > 0
    assert preview.avg_chunk_size > 0

    first_chunk = preview.chunks[0]
    assert isinstance(first_chunk, ChunkItem)
    assert first_chunk.chunk_id == 1
    assert first_chunk.token_estimate >= 1
    assert "Header 1" in first_chunk.metadata or "Judul Dokumen" in first_chunk.content


def test_preview_markdown_chunks_empty_and_whitespace() -> None:
    empty_preview = preview_markdown_chunks("", source_file="empty_doc")
    assert empty_preview.total_chunks == 0
    assert empty_preview.total_characters == 0
    assert empty_preview.chunks == []

    ws_preview = preview_markdown_chunks("   \n\n  \t  ", source_file="ws_doc")
    assert ws_preview.total_chunks == 0
    assert ws_preview.total_characters == 0


def test_preview_markdown_chunks_plain_text_without_headers() -> None:
    plain_text = (
        "Ini adalah paragraf naratif panjang tanpa struktur heading apapun. "
        * 15
    )
    preview = preview_markdown_chunks(
        plain_text,
        source_file="plain_doc",
        chunk_size=200,
        chunk_overlap=30,
    )

    assert preview.total_chunks > 1
    for chunk in preview.chunks:
        assert chunk.char_count <= 250  # toleransi pemisahan


def test_preview_markdown_chunks_code_blocks_and_tables() -> None:
    md_content = """# Dokumentasi Teknis

Berikut adalah cuplikan kode integrasi:

```python
def extract_text(image):
    return "text"
```

| Kolom A | Kolom B |
|---|---|
| Nilai 1 | Nilai 2 |
| Nilai 3 | Nilai 4 |
"""
    preview = preview_markdown_chunks(
        md_content,
        source_file="code_and_table",
        chunk_size=500,
        chunk_overlap=50,
    )

    assert preview.total_chunks >= 1
    all_text = " ".join(c.content for c in preview.chunks)
    assert "extract_text" in all_text
    assert "Kolom A" in all_text


def test_chunk_and_document_conversion() -> None:
    chunk = ChunkItem(
        chunk_id=1,
        char_count=50,
        token_estimate=12,
        preview="Pratinjau teks...",
        content="Isi dokumen lengkap untuk pengujian konversi chunk.",
        metadata={"section": "intro"},
        start_char=0,
        end_char=50,
    )
    doc = chunk_to_document(chunk, doc_stem="sop_keuangan")
    assert doc.page_content == chunk.content
    assert doc.id == "sop_keuangan_chunk_1"
    assert doc.metadata["chunk_id"] == 1
    assert doc.metadata["doc_stem"] == "sop_keuangan"

    retrieval = document_to_retrieval_result(doc, score=0.88)
    assert retrieval.chunk_id == 1
    assert retrieval.content == doc.page_content
    assert retrieval.score == 0.88
    assert retrieval.metadata["doc_stem"] == "sop_keuangan"


# ==============================================================================
# 2. Embedding Configuration & Builders
# ==============================================================================


def test_build_embeddings_default_and_custom_settings() -> None:
    default_settings = Settings()
    emb = build_embeddings(default_settings)

    assert emb.model == "text-embedding-nomic-embed-text-v1.5"
    assert "8080" in str(emb.openai_api_base)
    assert emb.check_embedding_ctx_length is False

    custom_settings = Settings(
        embedding_model="custom-embed-model",
        embedding_base_url="http://10.0.0.1:9000/v1",
        embedding_api_key="secret-key",
        embedding_dimensions=768,
        embedding_timeout=60.0,
    )
    custom_emb = build_embeddings(custom_settings)

    assert custom_emb.model == "custom-embed-model"
    assert "9000" in str(custom_emb.openai_api_base)
    assert custom_emb.dimensions == 768


def test_deterministic_mock_embeddings_properties() -> None:
    mock_emb = DeterministicMockEmbeddings(dimensions=256)

    # Identical texts yield identical vectors
    vec1 = mock_emb.embed_query("sistem rag vision vlm")
    vec2 = mock_emb.embed_query("sistem rag vision vlm")
    assert len(vec1) == 256
    assert vec1 == vec2

    # Different texts yield different vectors
    vec3 = mock_emb.embed_query("keuangan perbankan tabulasi")
    assert vec1 != vec3

    # Empty string and whitespace handling
    empty_vec = mock_emb.embed_query("")
    assert len(empty_vec) == 256
    assert all(v == 0.0 for v in empty_vec)

    ws_vec = mock_emb.embed_query("   \n\t  ")
    assert len(ws_vec) == 256
    assert all(v == 0.0 for v in ws_vec)

    # Batch embedding
    batch = mock_emb.embed_documents(["doc one", "doc two"])
    assert len(batch) == 2
    assert len(batch[0]) == 256


def test_safe_fallback_embeddings_offline_resilience() -> None:
    failing_primary = MagicMock()
    failing_primary.embed_documents.side_effect = ConnectionError("Endpoint down")
    failing_primary.embed_query.side_effect = ConnectionError("Endpoint down")

    fallback_emb = DeterministicMockEmbeddings(dimensions=128)
    safe = SafeFallbackEmbeddings(failing_primary, fallback_emb, warn_on_fallback=False)

    # Should not raise exception, falls back to deterministic mock
    res_batch = safe.embed_documents(["uji offline"])
    assert len(res_batch) == 1
    assert len(res_batch[0]) == 128
    assert safe._fallback_active is True
    assert failing_primary.embed_documents.call_count == 1

    # Second call should bypass primary completely because fallback is active
    res_single = safe.embed_query("uji single")
    assert len(res_single) == 128
    assert failing_primary.embed_query.call_count == 0


# ==============================================================================
# 3. Document Vector Store Persistence & Search
# ==============================================================================


def test_document_vector_store_persistence_and_reload(tmp_path: Path) -> None:
    persist_dir = tmp_path / "custom_doc" / "chroma"
    mock_emb = DeterministicMockEmbeddings(dimensions=128)

    store = DocumentVectorStore(
        doc_stem="custom_doc",
        persist_directory=persist_dir,
        embeddings=mock_emb,
        prefer_chroma=False,  # Uji backend JSON persistensi
    )

    chunks = [
        ChunkItem(
            chunk_id=1,
            char_count=45,
            token_estimate=11,
            preview="Prosedur peminjaman buku perpustakaan daerah.",
            content="Prosedur peminjaman buku perpustakaan daerah harus membawa KTP.",
            metadata={"category": "library"},
        ),
        ChunkItem(
            chunk_id=2,
            char_count=50,
            token_estimate=12,
            preview="Laporan keuangan neraca saldo kuartal ketiga.",
            content="Laporan keuangan neraca saldo kuartal ketiga menunjukkan laba bersih.",
            metadata={"category": "finance"},
        ),
    ]

    added = store.add_chunks(chunks)
    assert len(added) == 2
    assert store.count() == 2

    # Verifikasi file persistensi dibuat di disk
    json_file = persist_dir / "vector_store.json"
    assert json_file.is_file()
    assert json_file.stat().st_size > 0

    # Reload store baru dari path yang sama
    reloaded_store = DocumentVectorStore(
        doc_stem="custom_doc",
        persist_directory=persist_dir,
        embeddings=mock_emb,
        prefer_chroma=False,
    )

    assert reloaded_store.count() == 2

    # Lakukan pencarian semantik
    results = reloaded_store.search("buku perpustakaan pinjam", top_k=2)
    assert len(results) >= 1
    assert isinstance(results[0], RetrievalResult)
    assert "perpustakaan" in results[0].content


def test_document_vector_store_metadata_filter(tmp_path: Path) -> None:
    persist_dir = tmp_path / "filter_test" / "chroma"
    mock_emb = DeterministicMockEmbeddings(dimensions=64)

    store = DocumentVectorStore(
        doc_stem="filter_test",
        persist_directory=persist_dir,
        embeddings=mock_emb,
        prefer_chroma=False,
    )

    docs = [
        Document(
            page_content="Data keuangan divisi operasional",
            metadata={"department": "finance", "year": 2026},
        ),
        Document(
            page_content="Data rekrutmen karyawan divisi operasional",
            metadata={"department": "hr", "year": 2026},
        ),
    ]
    store.add_documents(docs)

    results = store.search(
        "operasional", top_k=2, filter_metadata={"department": "hr"}
    )
    assert len(results) == 1
    assert "rekrutmen" in results[0].content
    assert results[0].metadata.get("department") == "hr"


# ==============================================================================
# 4. Indexing Integration & Helper Functions
# ==============================================================================


def test_index_markdown_document_and_query_helper(tmp_path: Path) -> None:
    persist_dir = tmp_path / "notulen_rapat" / "chroma"
    mock_emb = DeterministicMockEmbeddings(dimensions=128)

    md_text = """# Notulen Rapat Koordinasi

## Agenda 1: Implementasi Model VLM
Disepakati penggunaan Gemma 4 12B untuk visi dan Qwen untuk agent reasoning.

## Agenda 2: Arsitektur RAG & Vector Store
Vector database disimpan terisolasi per-dokumen pada direktori output/{doc}/chroma.
"""

    index_result = index_markdown_document(
        markdown_text=md_text,
        doc_stem="notulen_rapat",
        persist_directory=persist_dir,
        embeddings=mock_emb,
        prefer_chroma=False,
    )

    assert index_result["status"] == "success"
    assert index_result["total_chunks"] >= 2
    assert index_result["doc_stem"] == "notulen_rapat"
    assert (persist_dir / "vector_store.json").is_file()

    # Query helper
    query_results = query_document_knowledge_base(
        query="bagaimana arsitektur vector store disimpan",
        doc_stem="notulen_rapat",
        top_k=1,
        persist_directory=persist_dir,
        embeddings=mock_emb,
    )

    assert len(query_results) == 1
    assert "chroma" in query_results[0].content.lower()


def test_index_markdown_document_empty(tmp_path: Path) -> None:
    persist_dir = tmp_path / "empty_doc" / "chroma"
    res = index_markdown_document(
        markdown_text="",
        doc_stem="empty_doc",
        persist_directory=persist_dir,
        prefer_chroma=False,
    )
    assert res["status"] == "empty"
    assert res["total_chunks"] == 0
    assert res["chunk_ids"] == []


def test_rag_pipeline_blueprint_integration(tmp_path: Path) -> None:
    persist_dir = tmp_path / "blueprint_doc" / "chroma"
    mock_emb = DeterministicMockEmbeddings(dimensions=64)

    vector_store = DocumentVectorStore(
        doc_stem="blueprint_doc",
        persist_directory=persist_dir,
        embeddings=mock_emb,
        prefer_chroma=False,
    )

    class DummyReranker(BaseReranker):
        def rerank(
            self, query: str, candidates: list[RetrievalResult], top_n: int = 3
        ) -> list[RetrievalResult]:
            # Reverse order to verify reranker invocation
            return list(reversed(candidates))[:top_n]

    pipeline = RAGPipelineBlueprint(
        vector_store=vector_store,
        reranker=DummyReranker(),
    )

    doc_text = """# Judul
## Bagian A
Konteks bagian A tentang pengolahan berkas.

## Bagian B
Konteks bagian B tentang penyimpanan arsip.
"""
    index_res = pipeline.index_markdown_document(
        markdown_text=doc_text, source_file="blueprint_doc"
    )
    assert index_res["status"] == "indexed"
    assert index_res["total_chunks"] >= 2

    # Query context with reranker
    results = pipeline.query_context(
        "berkas", top_k=2, use_reranker=True
    )
    assert len(results) >= 1


# ==============================================================================
# 5. DeepAgent Tool Registrations
# ==============================================================================


def test_deep_agent_tools_registration(tmp_path: Path) -> None:
    from app.deep_agent import build_deep_agent

    visual = MagicMock()
    language = MagicMock()
    pipeline = MagicMock()
    pipeline.ocr_extractor = None

    with (
        patch("app.deep_agent.build_vlm", return_value=visual),
        patch("app.deep_agent.build_language_vlm", return_value=language),
        patch("app.deep_agent.DocumentExtractionPipeline", return_value=pipeline),
        patch("app.deep_agent.VisionExtractor"),
        patch("app.deep_agent.create_deep_agent") as create,
    ):
        build_deep_agent(
            Settings(language_vlm_model="agent-model"),
            output_markdown_path=tmp_path / "output_test" / "output_test.md",
        )

        assert create.called
        tools_by_name = {
            item.name: item for item in create.call_args.kwargs["tools"]
        }

        # preview_chunks, query_document_knowledge_base, and index_document_to_knowledge_base must exist
        assert "preview_chunks" in tools_by_name
        assert "query_document_knowledge_base" in tools_by_name
        assert "index_document_to_knowledge_base" in tools_by_name

        # Test preview_chunks invocation
        sample_md = "# Header 1\nParagraf konten teks untuk simulasi chunk."
        preview_json = tools_by_name["preview_chunks"].invoke({
            "markdown_text": sample_md,
            "chunk_size": 200,
            "chunk_overlap": 50,
        })
        parsed_preview = json.loads(preview_json)
        assert parsed_preview["total_chunks"] >= 1
        assert parsed_preview["chunk_size"] == 200

        # Test query_document_knowledge_base invocation with empty/new store
        with patch("app.deep_agent.query_doc_kb", return_value=[
            RetrievalResult(chunk_id=1, content="Konten RAG hasil temu kembali", score=0.95)
        ]):
            kb_json = tools_by_name["query_document_knowledge_base"].invoke({
                "query": "uji cari konteks"
            })
            parsed_kb = json.loads(kb_json)
            assert len(parsed_kb) == 1
            assert parsed_kb[0]["content"] == "Konten RAG hasil temu kembali"
            assert parsed_kb[0]["score"] == 0.95

        # Test index_document_to_knowledge_base invocation
        with patch("app.deep_agent.index_markdown_doc", return_value={
            "status": "success", "doc_stem": "output_test", "total_chunks": 1
        }):
            idx_json = tools_by_name["index_document_to_knowledge_base"].invoke({
                "markdown_text": "# Judul\nKonten"
            })
            parsed_idx = json.loads(idx_json)
            assert parsed_idx["status"] == "success"
            assert parsed_idx["doc_stem"] == "output_test"


def test_document_vector_store_whitespace_search_and_clear(tmp_path: Path) -> None:
    persist_dir = tmp_path / "ws_test" / "chroma"
    mock_emb = DeterministicMockEmbeddings(dimensions=64)
    store = DocumentVectorStore(
        doc_stem="ws_test",
        persist_directory=persist_dir,
        embeddings=mock_emb,
        prefer_chroma=False,
    )
    store.add_documents([Document(page_content="Data operasional", metadata={"k": "v"})])
    assert store.count() == 1

    # Whitespace queries return empty results
    assert store.search("   \n  ") == []
    assert store.search("") == []

    # Clear resets store and deletes persistent file
    store.clear()
    assert store.count() == 0
    assert not (persist_dir / "vector_store.json").is_file()


def test_document_vector_store_vector_search_with_filter(tmp_path: Path) -> None:
    persist_dir = tmp_path / "vec_filter" / "chroma"
    mock_emb = DeterministicMockEmbeddings(dimensions=64)
    store = DocumentVectorStore(
        doc_stem="vec_filter",
        persist_directory=persist_dir,
        embeddings=mock_emb,
        prefer_chroma=False,
    )
    store.add_documents([
        Document(page_content="Dokumen Keuangan", metadata={"dept": "keuangan"}),
        Document(page_content="Dokumen SDM", metadata={"dept": "sdm"}),
    ])
    query_vec = mock_emb.embed_query("keuangan")
    results = store.search(query_vec, top_k=2, filter_metadata={"dept": "keuangan"})
    assert len(results) == 1
    assert "Keuangan" in results[0].content


def test_index_markdown_document_reindex_clears_stale_chunks(tmp_path: Path) -> None:
    persist_dir = tmp_path / "reindex_test" / "chroma"
    mock_emb = DeterministicMockEmbeddings(dimensions=64)

    # First index with 2 sections
    md1 = "# Bagian 1\nKonten pertama umum.\n## Bagian 2\nKonten rahasia kadaluarsa."
    res1 = index_markdown_document(
        md1,
        doc_stem="reindex_test",
        persist_directory=persist_dir,
        embeddings=mock_emb,
        prefer_chroma=False,
    )
    assert res1["total_chunks"] >= 2

    # Second index with 1 section only (reindexing clears old stale chunks)
    md2 = "# Bagian 1\nKonten baru saja."
    res2 = index_markdown_document(
        md2,
        doc_stem="reindex_test",
        persist_directory=persist_dir,
        embeddings=mock_emb,
        prefer_chroma=False,
        clear_existing=True,
    )
    assert res2["total_chunks"] == 1

    store = DocumentVectorStore(
        doc_stem="reindex_test",
        persist_directory=persist_dir,
        embeddings=mock_emb,
        prefer_chroma=False,
    )
    assert store.count() == 1
    # Verify stale content is completely gone
    search_res = store.search("kadaluarsa", top_k=5)
    assert not any("kadaluarsa" in r.content for r in search_res)


def test_rag_pipeline_blueprint_doc_stem_custom() -> None:
    pipeline = RAGPipelineBlueprint()
    with patch("app.vector_store.index_markdown_document", return_value={"status": "indexed", "doc_stem": "my_doc"}) as mock_idx:
        res = pipeline.index_markdown_document("# Teks", doc_stem="my_doc")
        assert res["doc_stem"] == "my_doc"
        assert mock_idx.called

    with patch("app.vector_store.query_document_knowledge_base", return_value=[
        RetrievalResult(chunk_id=1, content="Konteks ditemukan", score=1.0)
    ]) as mock_query:
        query_res = pipeline.query_context("query test", doc_stem="my_doc")
        assert len(query_res) == 1
        assert mock_query.call_args.kwargs.get("doc_stem") == "my_doc"


# ==============================================================================
# 6. Multimodal Chunking & Qwen-VL Architecture Tests
# ==============================================================================


def test_multimodal_chunk_item_properties() -> None:
    from app.rag import ChunkItem

    # Text-only chunk
    text_chunk = ChunkItem(
        chunk_id=1,
        char_count=20,
        token_estimate=5,
        preview="Teks pratinjau...",
        content="Isi dokumen teks biasa.",
    )
    assert text_chunk.is_multimodal is False
    assert text_chunk.image_path is None
    assert text_chunk.page_number is None

    # Multimodal chunk
    mm_chunk = ChunkItem(
        chunk_id=2,
        char_count=35,
        token_estimate=8,
        preview="Teks slide visual...",
        content="Slide 2: Diagram arsitektur VLM.",
        page_number=2,
        image_path="/data/output/doc/pages/page_002.png",
        image_metadata={"format": "png", "width": 1920, "height": 1080},
    )
    assert mm_chunk.is_multimodal is True
    assert mm_chunk.page_number == 2
    assert "page_002.png" in str(mm_chunk.image_path)
    d = mm_chunk.to_dict()
    assert d["chunk_id"] == 2
    assert d["image_metadata"]["width"] == 1920


def test_preview_markdown_chunks_with_page_images_and_delimiters(tmp_path: Path) -> None:
    from app.rag import preview_markdown_chunks

    md_with_pages = """<!-- PAGE: 1 -->
# Laporan Tahunan
Kinerja operasional tahun berjalan menunjukkan peningkatan efisiensi yang signifikan.

<!-- PAGE: 2 -->
## Neraca Finansial
Total aset lancar dan liabilitas tertera pada tabel berikut.
"""
    img_map = {
        1: str(tmp_path / "page_001.png"),
        2: str(tmp_path / "page_002.png"),
    }

    preview = preview_markdown_chunks(
        md_with_pages,
        source_file="laporan_tahunan",
        chunk_size=300,
        chunk_overlap=30,
        page_images=img_map,
    )

    assert preview.total_chunks >= 2
    assert preview.multimodal_chunks_count >= 2

    # Check page 1 chunk
    chunk1 = preview.chunks[0]
    assert chunk1.page_number == 1
    assert chunk1.image_path == img_map[1]
    assert chunk1.is_multimodal is True

    # Check page 2 chunk
    chunk2 = preview.chunks[1]
    assert chunk2.page_number == 2
    assert chunk2.image_path == img_map[2]
    assert chunk2.is_multimodal is True


def test_chunk_multimodal_pages_with_document_page_objects() -> None:
    from app.rag import chunk_multimodal_pages
    from app.schemas import DocumentPage

    pages = [
        DocumentPage(
            page_number=1,
            markdown_content="# Halaman 1: Pengantar\nKebijakan pemrosesan dokumen perusahaan.",
            image_path="output/doc/pages/page_001.png",
        ),
        DocumentPage(
            page_number=2,
            markdown_content="## Halaman 2: Tata Kelola\nProsedur eskalasi dan verifikasi dokumen.",
            image_path="output/doc/pages/page_002.png",
        ),
    ]

    preview = chunk_multimodal_pages(pages, doc_stem="sop_dokumen", chunk_size=200, chunk_overlap=20)
    assert preview.total_chunks == 2
    assert preview.multimodal_chunks_count == 2
    assert preview.chunks[0].page_number == 1
    assert preview.chunks[0].image_path == "output/doc/pages/page_001.png"
    assert preview.chunks[1].page_number == 2
    assert preview.chunks[1].image_path == "output/doc/pages/page_002.png"


def test_deterministic_multimodal_mock_embeddings() -> None:
    from app.rag import DeterministicMultimodalMockEmbeddings

    embedder = DeterministicMultimodalMockEmbeddings(dimensions=128)

    # Identical text + image yields identical vector
    vec1 = embedder.embed_multimodal("analisis keuangan", image_path="/path/img1.png")
    vec2 = embedder.embed_multimodal("analisis keuangan", image_path="/path/img1.png")
    assert len(vec1) == 128
    assert vec1 == vec2

    # Same text, different image yields different vector
    vec3 = embedder.embed_multimodal("analisis keuangan", image_path="/path/different_img.png")
    assert vec1 != vec3

    # Text-only embedding
    vec_text = embedder.embed_text("analisis keuangan")
    assert len(vec_text) == 128
    assert vec_text != vec1

    # Batch embedding
    batch = embedder.embed_batch(["teks satu", "teks dua"])
    assert len(batch) == 2
    assert len(batch[0]) == 128


def test_qwen_vl_embedding_service_formatting_and_fallback() -> None:
    from app.rag import DeterministicMultimodalMockEmbeddings, QwenVLEmbeddingService

    fallback = DeterministicMultimodalMockEmbeddings(dimensions=64)
    service = QwenVLEmbeddingService(
        model="Qwen/Qwen2-VL-7B-Instruct",
        base_url=None,  # Offline mode -> automatic fallback
        dimensions=64,
        fallback_embeddings=fallback,
    )

    # Format multimodal prompt verification
    payload = service.format_multimodal_prompt(
        text="Jelaskan isi diagram berikut",
        image_path="/images/diagram_flow.png",
    )
    assert payload["model"] == "Qwen/Qwen2-VL-7B-Instruct"
    parts = payload["messages"][0]["content"]
    assert len(parts) == 2
    assert parts[0]["type"] == "image_url"
    assert parts[0]["image_url"]["url"] == "/images/diagram_flow.png"
    assert parts[1]["type"] == "text"
    assert parts[1]["text"] == "Jelaskan isi diagram berikut"

    # Embedding with offline fallback
    vec = service.embed_multimodal("Teks uji", image_path="/images/diagram_flow.png")
    assert len(vec) == 64

    # Batch multimodal embedding
    batch_vecs = service.embed_multimodal_batch([
        ("Teks 1", "/images/img1.png"),
        ("Teks 2", None),
    ])
    assert len(batch_vecs) == 2
    assert len(batch_vecs[0]) == 64


def test_qwen_vl_reranker_multimodal_scoring() -> None:
    from app.rag import QwenVLReranker, RetrievalResult

    reranker = QwenVLReranker(model="Qwen/Qwen2-VL-7B-Instruct")

    candidates = [
        RetrievalResult(chunk_id=1, content="Teks relevan tanpa gambar", score=0.80),
        RetrievalResult(
            chunk_id=2,
            content="Teks relevan dengan bukti gambar halaman",
            score=0.79,
            image_path="/data/page_002.png",
            page_number=2,
        ),
    ]

    reranked = reranker.rerank("cari bukti gambar", candidates, top_n=2)
    assert len(reranked) == 2
    # Candidate with image gets visual bonus and moves to front
    assert reranked[0].chunk_id == 2
    assert reranked[0].image_path == "/data/page_002.png"
    assert reranked[0].score > 0.79


def test_multimodal_rag_pipeline_with_pages_and_retrieval(tmp_path: Path) -> None:
    from app.rag import MultimodalRAGPipeline
    from app.schemas import DocumentPage
    from app.vector_store import DeterministicMockEmbeddings, DocumentVectorStore

    persist_dir = tmp_path / "mm_pipeline_doc" / "chroma"
    mock_emb = DeterministicMockEmbeddings(dimensions=64)
    vstore = DocumentVectorStore(
        doc_stem="mm_pipeline_doc",
        persist_directory=persist_dir,
        embeddings=mock_emb,
        prefer_chroma=False,
    )
    pipeline = MultimodalRAGPipeline(vector_store=vstore)

    pages = [
        DocumentPage(
            page_number=1,
            markdown_content="# Ringkasan Eksekutif\nProyeksi pertumbuhan kuartal pertama.",
            image_path=str(tmp_path / "page_001.png"),
        ),
        DocumentPage(
            page_number=2,
            markdown_content="## Data Penjualan Regional\nPenjualan region barat mencapai target.",
            image_path=str(tmp_path / "page_002.png"),
        ),
    ]

    index_res = pipeline.index_pages(pages, doc_stem="mm_pipeline_doc")
    assert index_res["status"] == "indexed"
    assert index_res["total_chunks"] == 2
    assert index_res["multimodal_chunks"] == 2

    # Query context: must return image_path and page_number
    results = pipeline.query_context("penjualan region barat", top_k=1, doc_stem="mm_pipeline_doc")
    assert len(results) == 1
    top_res = results[0]
    assert "penjualan" in top_res.content.lower()
    assert top_res.page_number == 2
    assert top_res.image_path == str(tmp_path / "page_002.png")


def test_index_document_pages_vector_store_persistence(tmp_path: Path) -> None:
    from app.schemas import DocumentPage
    from app.vector_store import (
        DeterministicMockEmbeddings,
        DocumentVectorStore,
        index_document_pages,
    )

    persist_dir = tmp_path / "idx_pages_test" / "chroma"
    mock_emb = DeterministicMockEmbeddings(dimensions=64)

    pages = [
        DocumentPage(
            page_number=1,
            markdown_content="# Prosedur Standar Operasional\nDokumen resmi organisasi.",
            image_path="/data/doc/page_001.png",
        ),
    ]

    res = index_document_pages(
        pages=pages,
        doc_stem="idx_pages_test",
        persist_directory=persist_dir,
        embeddings=mock_emb,
        prefer_chroma=False,
    )
    assert res["status"] == "success"
    assert res["total_chunks"] == 1
    assert res["multimodal_chunks"] == 1

    # Reload store and query
    reloaded = DocumentVectorStore(
        doc_stem="idx_pages_test",
        persist_directory=persist_dir,
        embeddings=mock_emb,
        prefer_chroma=False,
    )
    hits = reloaded.search("prosedur standar", top_k=1)
    assert len(hits) == 1
    assert hits[0].page_number == 1
    assert hits[0].image_path == "/data/doc/page_001.png"


def test_deep_agent_retrieve_document_context_tool(tmp_path: Path) -> None:
    from app.deep_agent import build_deep_agent

    visual = MagicMock()
    language = MagicMock()
    pipeline = MagicMock()
    pipeline.ocr_extractor = None

    with (
        patch("app.deep_agent.build_vlm", return_value=visual),
        patch("app.deep_agent.build_language_vlm", return_value=language),
        patch("app.deep_agent.DocumentExtractionPipeline", return_value=pipeline),
        patch("app.deep_agent.VisionExtractor"),
        patch("app.deep_agent.create_deep_agent") as create,
    ):
        build_deep_agent(
            Settings(language_vlm_model="agent-model"),
            output_markdown_path=tmp_path / "mm_agent" / "mm_agent.md",
        )

        assert create.called
        tools_by_name = {item.name: item for item in create.call_args.kwargs["tools"]}

        assert "retrieve_document_context" in tools_by_name
        assert "query_document_knowledge_base" in tools_by_name

        with patch("app.deep_agent.query_doc_kb", return_value=[
            RetrievalResult(
                chunk_id=1,
                content="Bukti konteks teks dokumen",
                score=0.92,
                page_number=3,
                image_path="/data/page_003.png",
            )
        ]):
            res_json = tools_by_name["retrieve_document_context"].invoke({
                "query": "cari bukti konteks",
            })
            parsed = json.loads(res_json)
            assert len(parsed) == 1
            assert parsed[0]["content"] == "Bukti konteks teks dokumen"
            assert parsed[0]["page_number"] == 3
            assert parsed[0]["image_path"] == "/data/page_003.png"


def test_api_rag_index_and_retrieve_endpoints(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from app.api import app

    client = TestClient(app)

    with patch("app.vector_store.index_markdown_document", return_value={
        "status": "success",
        "doc_stem": "api_test_doc",
        "total_chunks": 3,
        "multimodal_chunks": 2,
        "total_characters": 500,
        "persist_directory": str(tmp_path),
        "backend": "in_memory_json",
        "chunk_ids": ["c1", "c2", "c3"],
    }):
        resp_idx = client.post(
            "/rag/index",
            json={
                "markdown_text": "# Judul Dokumen\nKonten paragraf.",
                "doc_stem": "api_test_doc",
            },
        )
        assert resp_idx.status_code == 200
        idx_data = resp_idx.json()
        assert idx_data["status"] == "success"
        assert idx_data["total_chunks"] == 3
        assert idx_data["multimodal_chunks"] == 2

    with patch("app.vector_store.query_document_knowledge_base", return_value=[
        RetrievalResult(
            chunk_id=1,
            content="Konteks teks hasil retrieval",
            score=0.88,
            page_number=1,
            image_path="/output/doc/pages/page_001.png",
        )
    ]):
        resp_ret = client.post(
            "/rag/retrieve",
            json={
                "query": "cari informasi dokumen",
                "doc_stem": "api_test_doc",
                "top_k": 2,
            },
        )
        assert resp_ret.status_code == 200
        ret_data = resp_ret.json()
        assert ret_data["total_results"] == 1
        item = ret_data["results"][0]
        assert item["content"] == "Konteks teks hasil retrieval"
        assert item["page_number"] == 1
        assert item["image_path"] == "/output/doc/pages/page_001.png"


def test_document_vector_store_add_chunks_with_precomputed_embeddings(tmp_path: Path) -> None:
    import pytest

    from app.rag import ChunkItem
    from app.vector_store import DocumentVectorStore

    store = DocumentVectorStore(
        doc_stem="precomputed_doc",
        persist_directory=tmp_path / "precomputed" / "chroma",
        prefer_chroma=False,
    )

    chunk = ChunkItem(
        chunk_id=1,
        char_count=20,
        token_estimate=5,
        preview="Uji precomputed",
        content="Konten precomputed embedding.",
        page_number=1,
        image_path="/data/page_1.png",
    )

    custom_vec = [0.42] * 1536
    added_ids = store.add_chunks([chunk], embeddings=[custom_vec])
    from langchain_core.vectorstores import InMemoryVectorStore

    assert isinstance(store.store, InMemoryVectorStore)
    stored_item = store.store.store[added_ids[0]]
    assert stored_item["vector"] == custom_vec
    assert stored_item["metadata"]["image_path"] == "/data/page_1.png"

    # Mismatched length raises ValueError
    with pytest.raises(ValueError, match="harus sama dengan"):
        store.add_chunks([chunk], embeddings=[custom_vec, custom_vec])


def test_document_vector_store_vector_search_returns_actual_score(tmp_path: Path) -> None:
    from app.vector_store import (
        DeterministicMockEmbeddings,
        DocumentVectorStore,
        index_markdown_document,
    )

    persist_dir = tmp_path / "vec_score_doc" / "chroma"
    mock_emb = DeterministicMockEmbeddings(dimensions=64)

    index_markdown_document(
        "# Judul\nDokumen finansial dan perbankan digital.",
        doc_stem="vec_score_doc",
        persist_directory=persist_dir,
        embeddings=mock_emb,
        prefer_chroma=False,
    )

    store = DocumentVectorStore(
        doc_stem="vec_score_doc",
        persist_directory=persist_dir,
        embeddings=mock_emb,
        prefer_chroma=False,
    )
    query_vec = mock_emb.embed_query("finansial perbankan")
    results = store.search(query_vec, top_k=1)
    assert len(results) == 1
    # Must return calculated similarity score, not dummy 1.0
    assert isinstance(results[0].score, float)
    assert results[0].score > 0.0


def test_multimodal_rag_pipeline_default_store_with_embedding_and_reranker(tmp_path: Path) -> None:
    from app.rag import (
        DeterministicMultimodalMockEmbeddings,
        MultimodalRAGPipeline,
        QwenVLReranker,
    )

    mock_emb = DeterministicMultimodalMockEmbeddings(dimensions=64)
    reranker = QwenVLReranker()
    pipeline = MultimodalRAGPipeline(
        embedding_service=mock_emb,
        vector_store=None,  # Tests default store integration
        reranker=reranker,
    )

    md = """<!-- PAGE: 1 -->
# Laporan Analisis
Kinerja Q1 sangat memuaskan.
<!-- PAGE: 2 -->
## Grafik Pertumbuhan
Proyeksi pendapatan melonjak tajam.
"""
    res = pipeline.index_markdown_document(md, doc_stem="pipeline_def_doc")
    assert res["status"] == "indexed"
    assert res["total_chunks"] >= 2

    # Query context should execute embedding service and reranker
    hits = pipeline.query_context("pendapatan melonjak", top_k=2, doc_stem="pipeline_def_doc")
    assert len(hits) >= 1
    assert any("pertumbuhan" in h.content.lower() or "pendapatan" in h.content.lower() for h in hits)


def test_qwen_vl_embedding_service_langchain_embeddings_protocol() -> None:
    from langchain_core.embeddings import Embeddings

    from app.rag import QwenVLEmbeddingService

    service = QwenVLEmbeddingService(dimensions=32)
    assert isinstance(service, Embeddings)

    doc_vecs = service.embed_documents(["teks satu", "teks dua"])
    assert len(doc_vecs) == 2
    assert len(doc_vecs[0]) == 32

    q_vec = service.embed_query("query pencarian")
    assert len(q_vec) == 32


def test_preview_markdown_chunks_with_tuple_and_string_page_keys() -> None:
    from app.rag import preview_markdown_chunks

    md = """<!-- PAGE: 1 -->
# Halaman Pertama
Teks halaman 1.
<!-- PAGE: 2 -->
## Halaman Kedua
Teks halaman 2.
"""
    # Tuple of strings (Sequence)
    preview_tuple = preview_markdown_chunks(
        md,
        page_images=("/data/page_1.png", "/data/page_2.png"),
    )
    assert preview_tuple.total_chunks >= 2
    assert preview_tuple.chunks[0].image_path == "/data/page_1.png"
    assert preview_tuple.chunks[-1].image_path == "/data/page_2.png"
    assert preview_tuple.chunks[-1].page_number == 2

    # Dict with string keys
    preview_dict = preview_markdown_chunks(
        md,
        page_images={"page_1": "/data/p1.png", "page_2": "/data/p2.png"},
    )
    assert preview_dict.total_chunks >= 2
    assert preview_dict.chunks[0].image_path == "/data/p1.png"
    assert preview_dict.chunks[-1].image_path == "/data/p2.png"
    assert preview_dict.chunks[-1].page_number == 2


def test_preview_markdown_chunks_default_page_number_one() -> None:
    from app.rag import preview_markdown_chunks

    md_plain = "# Bab 1\nPengantar tanpa delimiter halaman eksplisit."
    preview = preview_markdown_chunks(md_plain)
    assert preview.total_chunks >= 1
    assert preview.chunks[0].page_number == 1


def test_deep_agent_search_document_chunks_and_stem_sanitization(tmp_path: Path) -> None:
    from unittest.mock import MagicMock, patch

    from app.config import Settings
    from app.deep_agent import build_deep_agent
    from app.rag import RetrievalResult

    visual = MagicMock()
    language = MagicMock()
    pipeline = MagicMock()

    with (
        patch("app.deep_agent.build_vlm", return_value=visual),
        patch("app.deep_agent.build_language_vlm", return_value=language),
        patch("app.deep_agent.DocumentExtractionPipeline", return_value=pipeline),
        patch("app.deep_agent.VisionExtractor"),
        patch("app.deep_agent.create_deep_agent") as create,
    ):
        build_deep_agent(
            Settings(language_vlm_model="agent-model"),
            output_markdown_path=tmp_path / "deep_test" / "doc.md",
        )

        assert create.called
        tools_by_name = {item.name: item for item in create.call_args.kwargs["tools"]}

        assert "search_document_chunks" in tools_by_name

        with patch("app.deep_agent.query_doc_kb", return_value=[
            RetrievalResult(chunk_id=1, content="Hasil search", score=0.95, page_number=1)
        ]) as mock_kb:
            # Pass stem with .md extension
            res_str = tools_by_name["search_document_chunks"].invoke({
                "query": "uji search",
                "doc_stem": "laporan_keuangan.md",
            })
            assert "Hasil search" in res_str
            # Should have stripped .md extension
            assert mock_kb.call_args.kwargs["doc_stem"] == "laporan_keuangan"


def test_api_rag_index_with_list_page_images_and_stem_cleanup(tmp_path: Path) -> None:
    from unittest.mock import patch

    from fastapi.testclient import TestClient

    from app.api import app

    client = TestClient(app)

    with patch("app.vector_store.index_markdown_document", return_value={
        "status": "success",
        "doc_stem": "laporan",
        "total_chunks": 2,
        "multimodal_chunks": 2,
        "total_characters": 100,
        "persist_directory": str(tmp_path),
        "backend": "in_memory_json",
        "chunk_ids": ["c1", "c2"],
    }) as mock_idx:
        resp = client.post(
            "/rag/index",
            json={
                "markdown_text": "# Laporan",
                "doc_stem": "laporan.md",
                "page_images": ["/pages/p1.png", "/pages/p2.png"],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert mock_idx.call_args.kwargs["doc_stem"] == "laporan"
        assert mock_idx.call_args.kwargs["page_images"] == ["/pages/p1.png", "/pages/p2.png"]


