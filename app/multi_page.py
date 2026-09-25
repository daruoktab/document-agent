"""
Modul penyambung halaman dokumen multi-halaman (Multi-Page Document Stitcher).

Menyediakan logika cerdas untuk:
  - Menyambungkan Markdown dari banyak halaman PDF / Scan / PPT menjadi satu dokumen utuh.
  - Menjaga kontinuitas hierarki heading Markdown (#, ##, ###) antar halaman.
  - Membersihkan header & footer berulang (mis. "Halaman 1 dari 10", running headers).
  - Menyambung paragraf yang terpotong di akhir halaman.
  - Modul RAG & Chunking telah dialihkan ke staging: app.rag_staging (blueprint masa depan).
"""

from __future__ import annotations

import re
from typing import Any


def strip_embedded_image_markup(markdown: str) -> str:
    """Remove embedded image references; visual descriptions remain as text."""
    if not markdown:
        return markdown
    image_only_div = r"<div\b[^>]*>\s*<img\b[^>]*>\s*</div\s*>"
    cleaned = re.sub(image_only_div, "", markdown, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<img\b[^>]*>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"!\[[^\]\n]*\]\([^\n)]*\)", "", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _clean_page_artifacts(markdown: str) -> str:
    """Bersihkan artefak header/footer halaman umum."""
    lines = markdown.splitlines()
    cleaned_lines = []

    # Pola running page number: "Page 1", "Halaman 2 dari 10", "- 3 -", "1 / 15"
    page_num_pat = re.compile(
        r"^(halaman|page|\-)?\s*\d+\s*(dari|of|\/)?\s*\d*\s*(\-)?$", re.IGNORECASE
    )

    for line in lines:
        stripped = line.strip()
        if page_num_pat.match(stripped):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


PAGE_DELIMITER_RE = re.compile(
    r"<!--\s*(PAGE|SLIDE)\s*:\s*(\d+)\s*-->|<!--\s*(Page|Slide)\s*(\d+)\s*-->",
    re.IGNORECASE,
)


def format_page_delimiter(page_number: int, is_slide: bool = False) -> str:
    """Format penanda batas halaman/slide standar sistem."""
    tag = "SLIDE" if is_slide else "PAGE"
    return f"<!-- {tag}: {page_number} -->"


def strip_thinking_process(markdown: str) -> str:
    """
    Hapus blok penalaran model / reasoning token (<think>...</think>)
    yang dihasilkan oleh model reasoning (seperti DeepSeek-R1, Qwen-2.5-Coder-Reasoning, QwQ)
    agar tidak bocor ke output dokumen Markdown.

    Menangani kasus:
    1. Blok <think>...</think> lengkap.
    2. Tag </think> dangling tanpa pembuka.
    3. Blok <think> unclosed di awal (terpotong oleh token limit).
    4. Sisa-sisa preamble reasoning seperti 'Wait, I need to check...' sebelum konten dokumen.
    """
    if not markdown:
        return ""

    text = markdown

    # 1. Hapus blok lengkap <think>...</think>
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)

    # 2. Jika ada </think> dangling di awal
    if "</think>" in text.lower():
        parts = re.split(r"</think>", text, flags=re.IGNORECASE, maxsplit=1)
        if len(parts) > 1:
            text = parts[1]

    # 3. Jika ada <think> unclosed di awal sebelum konten
    if "<think>" in text.lower():
        parts = re.split(r"<think>", text, flags=re.IGNORECASE, maxsplit=1)
        text = parts[0]

    # 4. Bersihkan sisa rambling reasoning di awal jika ada
    lines = text.splitlines()
    start_idx = 0
    reasoning_prefixes = (
        "wait, i need to check",
        "let me check",
        "let me re-read",
        "let me verify",
        "draft table",
        "thinking process:",
        "i will output",
        "mari kita",
        "mari saya",
        "ini melanggar aturan",
        "aturan ",
        "perbaikan teks",
        "koreksi draft",
        "draft markdown",
        "langkah demi langkah",
        "evaluasi draf",
        "catatan koreksi",
    )
    found_real_content = False
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        lower_line = stripped.lower()
        if any(lower_line.startswith(p) for p in reasoning_prefixes):
            continue
        if (
            stripped.startswith(("#", "|", ">", "---", "```", "- ", "* ", "1. ", "<!--"))
            or (stripped.startswith("**") and stripped.endswith("**"))
        ):
            start_idx = idx
            found_real_content = True
            break

    if found_real_content and start_idx > 0:
        preceding_text = "\n".join(lines[:start_idx]).lower()
        if any(p in preceding_text for p in reasoning_prefixes):
            text = "\n".join(lines[start_idx:])

    return text.strip()


def strip_page_markers(markdown: str) -> str:
    """
    Buang seluruh penanda `<!-- PAGE: N -->` / `<!-- SLIDE: N -->` yang mungkin
    ikut ditulis oleh VLM di dalam konten halaman, agar penomoran hanya berasal
    dari stitcher (sumber kebenaran tunggal).
    """
    cleaned = strip_thinking_process(markdown)
    cleaned = PAGE_DELIMITER_RE.sub("", cleaned)
    return "\n".join(cleaned.splitlines()).strip()


def collapse_consecutive_duplicate_blocks(markdown: str) -> str:
    """
    Runtuhkan pengulangan blok identik yang berurutan (gejala degenerasi/loop
    keluaran VLM). Blok dipisahkan oleh baris kosong; hanya blok yang sama persis
    dan berurutan yang disatukan menjadi satu kemunculan.
    """
    text = markdown.strip()
    if not text:
        return markdown
    blocks = re.split(r"\n{2,}", text)
    if len(blocks) <= 1:
        return markdown
    out: list[str] = [blocks[0]]
    for b in blocks[1:]:
        if b.strip() != out[-1].strip():
            out.append(b)
    return "\n\n".join(out)


def split_markdown_by_pages(
    markdown: str,
    default_page_number: int = 1,
    default_type: str = "page",
) -> list[dict[str, Any]]:
    """
    Pisahkan teks Markdown berdasarkan penanda halaman/slide standar sistem
    (mis. `<!-- PAGE: 1 -->` atau `<!-- SLIDE: 1 -->`).

    Returns:
        List dictionary berisi page_number, type ('page' atau 'slide'), dan konten teksnya.
    """
    matches = list(PAGE_DELIMITER_RE.finditer(markdown))
    if not matches:
        return [
            {
                "page_number": default_page_number,
                "type": default_type,
                "content": markdown.strip(),
            }
        ]

    pages: list[dict[str, Any]] = []
    for i, m in enumerate(matches):
        tag_type = (m.group(1) or m.group(3) or "page").lower()
        page_num_str = m.group(2) or m.group(4) or str(i + 1)
        page_num = int(page_num_str)

        start_pos = m.end()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        content = markdown[start_pos:end_pos].strip()

        pages.append(
            {
                "page_number": page_num,
                "type": tag_type,
                "content": content,
            }
        )
    return pages


def extract_preamble(markdown: str) -> str:
    """Ekstrak teks/judul sebelum penanda halaman pertama jika ada."""
    m = PAGE_DELIMITER_RE.search(markdown)
    if m:
        return markdown[: m.start()].strip()
    return ""


def merge_and_stitch_markdown_pages(
    existing_markdown: str | None,
    incoming_markdown: str,
    *,
    default_page_number: int = 1,
    is_slide: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Gabungkan potongan Markdown halaman/slide yang baru diproses dengan Markdown
    yang sudah ada sebelumnya (inkremental per batch 10 halaman).

    Args:
        existing_markdown: Konten Markdown yang sudah ada di disk (atau None/kosong jika baru).
        incoming_markdown: Konten batch Markdown baru dari agent.
        default_page_number: Nomor halaman/slide default jika incoming_markdown tidak memiliki penanda delimiter.
        is_slide: True jika dokumen adalah presentasi (menggunakan tag SLIDE).

    Returns:
        tuple berisi (merged_full_markdown, sorted_pages_metadata_list).
    """
    tag_default = "slide" if is_slide else "page"
    incoming_pages = split_markdown_by_pages(
        incoming_markdown,
        default_page_number=default_page_number,
        default_type=tag_default,
    )
    incoming_preamble = extract_preamble(incoming_markdown)

    pages_dict: dict[int, dict[str, Any]] = {}
    preamble = incoming_preamble

    if existing_markdown and existing_markdown.strip():
        existing_pages = split_markdown_by_pages(
            existing_markdown, default_type=tag_default
        )
        existing_preamble = extract_preamble(existing_markdown)
        if not preamble and existing_preamble:
            preamble = existing_preamble
        for p in existing_pages:
            pages_dict[p["page_number"]] = p

    # Timpa atau sisipkan halaman dari batch incoming
    for p in incoming_pages:
        pages_dict[p["page_number"]] = p

    sorted_page_numbers = sorted(pages_dict.keys())
    sorted_pages = [pages_dict[num] for num in sorted_page_numbers]

    blocks: list[str] = []
    if preamble:
        blocks.append(preamble)

    for p in sorted_pages:
        delimiter = format_page_delimiter(p["page_number"], is_slide=is_slide)
        raw_content = p["content"].strip()
        cleaned_content = re.sub(r"^(\s*---\s*\n)+", "", raw_content)
        cleaned_content = re.sub(r"(\n\s*---\s*)+$", "", cleaned_content).strip()
        blocks.append(f"{delimiter}\n{cleaned_content}")

    merged_full_markdown = "\n\n---\n\n".join(blocks).strip() + "\n"
    return merged_full_markdown, sorted_pages


def extract_document_title(
    markdown_content: str,
    fallback_title: str | None = None,
) -> str | None:
    """
    Ekstrak judul dokumen dari teks Markdown (biasanya halaman pertama) secara cerdas.

    Strategi pencarian:
    1. Mencari baris heading level 1 (# Judul) yang bukan label halaman/dokumen generic.
    2. Mencari baris heading level 2 (## Judul) jika tidak ada #.
    3. Mencari teks tebal (**Judul**) yang berdiri sendiri pada baris awal.
    4. Fallback ke parameter `fallback_title` jika tidak ditemukan judul yang meyakinkan.
    """
    if not markdown_content or not markdown_content.strip():
        return fallback_title

    cleaned = strip_page_markers(strip_thinking_process(markdown_content)).strip()
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]

    generic_titles = {
        "document",
        "dokumen",
        "page 1",
        "halaman 1",
        "slide 1",
        "overview",
        "ringkasan",
        "pendahuluan",
        "untitled",
        "table of contents",
        "daftar isi",
    }

    # 1. Cari baris heading # (H1)
    for line in lines:
        if line.startswith("# ") and not line.startswith("##"):
            title = line[2:].strip()
            if title and title.lower() not in generic_titles:
                return title

    # 2. Cari baris heading ## (H2) jika H1 tidak ada
    for line in lines:
        if line.startswith("## ") and not line.startswith("###"):
            title = line[3:].strip()
            if title and title.lower() not in generic_titles:
                return title

    # 3. Cari baris tebal **Judul** pada 5 baris pertama
    for line in lines[:5]:
        m = re.match(r"^\*\*(.+?)\*\*$", line)
        if m:
            title = m.group(1).strip()
            if len(title) > 3 and title.lower() not in generic_titles:
                return title

    return fallback_title


def stitch_pages_to_markdown(
    pages_markdown: list[str] | list[Any],
    *,
    document_title: str | None = None,
    include_page_markers: bool = True,
    is_slide: bool = False,
    **kwargs: Any,
) -> str:
    """
    Gabungkan daftar Markdown per-halaman menjadi satu dokumen utuh yang konsisten.

    Args:
        pages_markdown: List string markdown (atau objek DocumentPage) dari setiap halaman (berurutan).
        document_title: Judul dokumen (opsional, akan diselaraskan sebagai heading # utama di halaman 1).
        include_page_markers: Jika True, sisipkan komentar `<!-- PAGE: N -->` / `<!-- SLIDE: N -->`.
        is_slide: Jika True, gunakan penanda `<!-- SLIDE: N -->`.

    Returns:
        String Markdown utuh siap dichunking.
    """
    if not pages_markdown:
        return ""

    raw_pages: list[str] = []
    for item in pages_markdown:
        if isinstance(item, str):
            raw_pages.append(item)
        elif hasattr(item, "markdown_content"):
            raw_pages.append(item.markdown_content)
        elif isinstance(item, dict) and "content" in item:
            raw_pages.append(item["content"])
        else:
            raw_pages.append(str(item))

    clean_title: str | None = None
    if document_title and document_title.strip():
        clean_title = document_title.strip()
        if clean_title.startswith("#"):
            clean_title = clean_title.lstrip("#").strip()

    stitched_blocks: list[str] = []

    # Jika tanpa page markers dan ada document_title, letakkan di paling awal
    if clean_title and not include_page_markers:
        stitched_blocks.append(f"# {clean_title}\n")

    for idx, page_md in enumerate(raw_pages, start=1):
        cleaned_md = _clean_page_artifacts(page_md).strip()
        if not cleaned_md:
            continue

        # Penanganan khusus judul dokumen:
        if clean_title:
            lines = cleaned_md.splitlines()
            if idx == 1:
                # Pada Halaman 1: pastikan diawali dengan '# <clean_title>'
                first_line = lines[0].strip() if lines else ""
                if first_line.startswith("# ") and not first_line.startswith("##"):
                    pass
                else:
                    cleaned_md = f"# {clean_title}\n\n{cleaned_md}"
            else:
                # Pada Halaman 2+: HAPUS baris '# <clean_title>' jika model mengulanginya
                if lines and lines[0].strip().lower() == f"# {clean_title.lower()}":
                    cleaned_md = "\n".join(lines[1:]).strip()

        if include_page_markers:
            stitched_blocks.append(
                f"\n{format_page_delimiter(idx, is_slide=is_slide)}\n"
            )

        # Cek kontinuitas paragraf: jika blok sebelumnya tidak diakhiri titik/header
        # dan halaman baru dimulai dengan huruf kecil, sambungkan secara mulus jika tanpa marker.
        if stitched_blocks and not include_page_markers:
            last_block = stitched_blocks[-1].rstrip()
            if last_block and not last_block.endswith(
                (".", ":", "!", "?", "#", ">", "|", "```")
            ):
                first_line = (
                    cleaned_md.splitlines()[0].strip()
                    if cleaned_md.splitlines()
                    else ""
                )
                if first_line and not first_line.startswith(
                    ("#", "-", "*", ">", "|", "1.", "2.")
                ):
                    stitched_blocks[-1] = last_block + " " + cleaned_md
                    continue

        stitched_blocks.append(cleaned_md)

    # Gabungkan dengan pemisah standar
    if include_page_markers:
        return "\n".join(stitched_blocks).strip() + "\n"
    return "\n\n".join(stitched_blocks).strip() + "\n"


def preview_markdown_chunks(
    markdown_content: str,
    *,
    source_file: str = "document",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> Any:
    """
    [STAGING / BLUEPRINT] Delegasi ke modul staging app.rag_staging.
    Simulasikan pemecahan dokumen Markdown menjadi chunk-chunk hierarkis.
    """
    from app.rag_staging import preview_markdown_chunks as _preview

    return _preview(
        markdown_content,
        source_file=source_file,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


__all__ = [
    "PAGE_DELIMITER_RE",
    "collapse_consecutive_duplicate_blocks",
    "extract_document_title",
    "extract_preamble",
    "format_page_delimiter",
    "merge_and_stitch_markdown_pages",
    "preview_markdown_chunks",
    "split_markdown_by_pages",
    "stitch_pages_to_markdown",
    "strip_page_markers",
    "strip_thinking_process",
]
