"""
Modul Spesialis Diagram & Visual Artifacts untuk Ekstraksi Vision VLM -> Mermaid.js.

Fitur:
  1. Klasifikasi kelayakan konversi diagram ke sintaks Mermaid.js.
  2. Ekstraksi visual diagram ke kode Mermaid (flowchart, sequence, ERD, class, state, mindmap, block architecture, dsb).
  3. Sanitasi dan validasi sintaks Mermaid.js.
  4. Fallback ke representasi deskriptif terstruktur jika diagram tidak cocok untuk Mermaid (grafik statistik kontinu, peta spasial, skematik sirkuit mikro, foto).
"""

from __future__ import annotations

import base64
import json
import logging
import platform
import re
import shutil
from pathlib import Path
from typing import cast

from langchain_core.language_models.chat_models import BaseChatModel

from .llm import image_data_uri
from .multi_page import strip_thinking_process
from .schemas import (
    DiagramConvertibilityResult,
    DiagramExtractionResult,
    DiagramFormatRecommendation,
    DiagramTypeLiteral,
)

logger = logging.getLogger(__name__)

MERMAID_KEYWORDS: tuple[str, ...] = (
    "flowchart",
    "graph",
    "sequencediagram",
    "classdiagram",
    "statediagram",
    "erdiagram",
    "mindmap",
    "gantt",
    "pie",
    "gitgraph",
    "c4context",
    "block-beta",
)


def _sanitize_mermaid_line(line: str) -> str:
    """Sanitasi baris kode Mermaid: auto-quote label berisiko dan bersihkan classDef ilegal."""
    stripped = line.strip()
    if not stripped or stripped.startswith("%%"):
        return line

    # 0. Normalisasi keyword legacy 'graph' menjadi 'flowchart'
    if re.match(r"^graph\s+(TD|TB|LR|RL|BT)\b", stripped, re.IGNORECASE):
        line = re.sub(r"^(\s*)graph\b", r"\1flowchart", line, flags=re.IGNORECASE)
        stripped = line.strip()

    # 1. Bersihkan classDef dari atribut non-CSS (rx, ry, dsb yang sering merusak parser Mermaid)
    if stripped.startswith("classDef"):
        cleaned_class = re.sub(r",\s*r[xy]:\s*[^,;]+", "", line)
        cleaned_class = re.sub(r"\br[xy]:\s*[^,;]+[;,]?", "", cleaned_class)
        cleaned_class = re.sub(r",\s*;", ";", cleaned_class)
        cleaned_class = re.sub(r",\s*,", ",", cleaned_class)
        return cleaned_class

    # 2. Lewati direktif styling atau struktur kontrol
    if any(
        stripped.startswith(prefix)
        for prefix in ("style ", "class ", "subgraph", "end", "linkStyle", "click")
    ):
        return line

    # 3. Perbaiki penutup kurung ganda umum: `"]]` -> `"]`, `"]}}` -> `"]}`
    line = re.sub(r'(?<=\w)\["([^"\n]*)"\]\]', r'["\1"]', line)
    line = re.sub(r'(?<=\w)\[([^\]\n]*)\]\](?!\s*\])', r'[\1]', line)
    line = re.sub(r'(?<=\w)\{"([^"\n]*)"\}\}', r'{"\1"}', line)

    # Bersihkan pembungkus (' ... ') atau (" ... ") di dalam label simpul bertanda kutip
    line = re.sub(r'\["\(\'([^\'\n]+)\'\)"\]', r'["\1"]', line)
    line = re.sub(r'\["\(\"([^\"\n]+)\"\)"\]', r'["\1"]', line)
    line = re.sub(r'\[\(\'([^\'\n]+)\'\)\]', r'["\1"]', line)

    def _quote_content(content: str) -> str:
        c = content.strip()
        c = re.sub(r"<br\s*/?>", "<br/>", c, flags=re.IGNORECASE)
        # Ganti escaped quote yang membingungkan tokenizer
        c = c.replace('\\"', "'")
        # Jika ada tanda kurung siku ganda di akhir
        if c.endswith("]"):
            c = c[:-1].strip()
        # Bersihkan pembungkus (' ... ') atau (" ... ")
        if (c.startswith("('") and c.endswith("')")) or (c.startswith('("') and c.endswith('")')):
            c = c[2:-2].strip()
        # Jika sudah dibungkus tanda kutip ganda atau tunggal
        if c.startswith('"') and c.endswith('"'):
            inner = c[1:-1].replace('"', "'")
            return f'"{inner}"'
        # Cek apakah mengandung karakter berisiko parse error: (), <>, :, /, ;, &, #, atau spasi
        risky_chars = set("()<>:/;&#,")
        if any(ch in c for ch in risky_chars) or " " in c:
            c = c.replace('"', "'")
            return f'"{c}"'
        return c

    # Ganti node bentuk circle (( ... ))
    line = re.sub(
        r'(?<=\w)\(\((?!\")([^\)\n]+)\)\)',
        lambda m: f"(({_quote_content(m.group(1))}))",
        line,
    )
    # Ganti node bentuk stadium ([ ... ])
    line = re.sub(
        r'(?<=\w)\(\[(?!\")([^\]\n]+)\]\)',
        lambda m: f"([{_quote_content(m.group(1))}])",
        line,
    )
    # Ganti node bentuk cylinder [( ... )]
    line = re.sub(
        r'(?<=\w)\[\((?!\")([^\)\n]+)\)\]',
        lambda m: f"[({_quote_content(m.group(1))})]",
        line,
    )
    # Ganti node bentuk square [ ... ]
    line = re.sub(
        r'(?<=\w)\[(?!\")([^\]\n]+)\]',
        lambda m: f"[{_quote_content(m.group(1))}]",
        line,
    )
    # Ganti node bentuk rhombus { ... }
    line = re.sub(
        r'(?<=\w)\{(?!\")([^\}\n]+)\}',
        lambda m: f"{{{_quote_content(m.group(1))}}}",
        line,
    )

    # 4. Bersihkan dangling <br/> dan teks di luar simpul sebelum panah
    # Contoh: RP1_RP0["RP1<br/>RP0"]<br/>(2)<br/>["Bank Select"] --> D_Mem
    line = re.sub(
        r'(\b\w+\["[^"\n]+"\])\s*<br\s*/?>\s*(?:<br\s*/?>|\([^\)\n]*\)|\"[^\"]*\"|\[[^\]\n]*\]|[^\s\-]+)*\s*(-->|---|-\.->|==>)',
        r"\1 \2",
        line,
        flags=re.IGNORECASE,
    )

    return line


def sanitize_mermaid_code(raw_text: str) -> str | None:
    """
    Ekstrak dan bersihkan blok kode Mermaid dari output model AI.
    Menghilangkan wrapper Markdown (```mermaid ... ```) dan memperbaiki anomali formatting umum.
    Termasuk auto-quoting label simpul dan pembersihan atribut ilegal classDef.
    """
    if not raw_text or not raw_text.strip():
        return None

    cleaned = raw_text.strip()

    # Ekstrak dari blok markdown ```mermaid ... ```
    mermaid_match = re.search(r"```(?:mermaid)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if mermaid_match:
        cleaned = mermaid_match.group(1).strip()

    # Periksa apakah baris pertama diawali kata kunci Mermaid yang valid
    lines = [line for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return None

    first_line = lines[0].strip().lower().replace(" ", "").replace("-", "")
    is_valid_header = any(first_line.startswith(kw.replace("-", "")) for kw in MERMAID_KEYWORDS)

    if not is_valid_header:
        # Coba perbaiki jika ada preamble teks sebelum diagram
        for idx, line in enumerate(lines):
            line_clean = line.strip().lower().replace(" ", "").replace("-", "")
            if any(line_clean.startswith(kw.replace("-", "")) for kw in MERMAID_KEYWORDS):
                cleaned = "\n".join(lines[idx:]).strip()
                break
        else:
            logger.warning("[Diagram:Sanitize] Tidak ditemukan header Mermaid valid.")
            return None

    # Sanitasi karakter yang sering merusak parsing Mermaid di web UI
    cleaned = cleaned.replace("\r\n", "\n")

    # 1. Sanitasi per baris (auto-quoting node labels & cleaning illegal classDef properties)
    sanitized_lines = [_sanitize_mermaid_line(l) for l in cleaned.splitlines()]

    # 2. Pemrosesan struktural multi-baris: Subgraph Cycles, Node Conflicts, Redundant Declarations, Duplicate Edges
    subgraph_stack: list[str] = []
    declared_nodes: dict[str, str] = {}  # node_id -> label
    seen_edges: set[str] = set()
    final_lines: list[str] = []

    arrow_regex = re.compile(r"(-->|---|-.->|==>|<--|<---|<-.-|<==)")

    for line in sanitized_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("%%"):
            final_lines.append(line)
            continue

        # Deteksi awal subgraph: subgraph ID [Title] atau subgraph ID ["Title"]
        m_sub = re.match(r"^subgraph\s+([A-Za-z0-9_]+)\b", stripped, re.IGNORECASE)
        if m_sub:
            sub_id = m_sub.group(1)
            subgraph_stack.append(sub_id)
            final_lines.append(line)
            continue

        if stripped.lower() == "end":
            if subgraph_stack:
                subgraph_stack.pop()
            final_lines.append(line)
            continue

        # A. Resolusi Subgraph Self-Cycle (mis. UserMem -.-> LowerMem di dalam subgraph UserMem)
        arrow_tokens = r"(?:-->|---|-.->|==>|<--|<---|<-.-|<==)"
        if subgraph_stack and arrow_regex.search(line):
            curr_sub = subgraph_stack[-1]
            esc_sub = re.escape(curr_sub)
            pattern_src = r"\b" + esc_sub + r"\b(?=\s*(?:\[|\(|\{|\s*" + arrow_tokens + r"))"
            line = re.sub(pattern_src, f"{curr_sub}_node", line)

            pattern_dst = r"(" + arrow_tokens + r"\s*(?:\|[^|\n]*\|\s*)?)\b" + esc_sub + r"\b"
            line = re.sub(pattern_dst, r"\g<1>" + f"{curr_sub}_node", line)

        # B. Resolusi Konflik Deklarasi Node ID yang sama dengan label berbeda antar-subgraph
        # Contoh: Block_Low["4Fh - 7Fh"] di Bank 0 vs Block_Low["4Fh"] di Bank 1
        node_matches = list(re.finditer(r'\b([A-Za-z0-9_]+)\["([^"\n]+)"\]', line))
        for nm in node_matches:
            nid = nm.group(1)
            nlabel = nm.group(2).strip()
            if nid in declared_nodes:
                prev_label = declared_nodes[nid]
                if prev_label != nlabel and subgraph_stack:
                    curr_sub = subgraph_stack[-1]
                    unique_nid = f"{curr_sub}_{nid}"
                    line = re.sub(r"\b" + re.escape(nid) + r"\b", unique_nid, line)
                    declared_nodes[unique_nid] = nlabel
                elif prev_label == nlabel and arrow_regex.search(line):
                    # C. Hapus deklarasi kurung label yang berulang pada relasi panah
                    redecl_pattern = r"(" + arrow_tokens + r"\s*(?:\|[^|\n]*\|\s*)?)\b" + re.escape(nid) + r'\["[^"\n]+"\]'
                    line = re.sub(redecl_pattern, r"\g<1>" + nid, line)
            else:
                declared_nodes[nid] = nlabel

        # D. Pencegahan Edge Duplikat (Duplicate Edges)
        if arrow_regex.search(line):
            norm_edge = re.sub(r"\s+", " ", line.strip())
            if norm_edge in seen_edges:
                continue
            seen_edges.add(norm_edge)

        final_lines.append(line)

    cleaned = "\n".join(final_lines)
    return cleaned


def validate_mermaid_syntax(mermaid_code: str) -> tuple[bool, str | None]:
    """
    Validasi sintaks Mermaid komprehensif (Linter Compiler).
    Mengembalikan (is_valid, error_message).
    """
    if not mermaid_code or not mermaid_code.strip():
        return False, "Kode Mermaid kosong"

    lines = [l.strip() for l in mermaid_code.splitlines() if l.strip()]
    if not lines:
        return False, "Kode Mermaid tidak memiliki baris instruksi"

    # Periksa header
    first_line = lines[0].lower().replace(" ", "").replace("-", "")
    if first_line.startswith("graph"):
        return False, "Keyword 'graph' tidak dianjurkan; gunakan 'flowchart' (contoh: flowchart TD)."
    if not any(first_line.startswith(kw.replace("-", "")) for kw in MERMAID_KEYWORDS):
        return False, f"Header Mermaid tidak dikenali: '{lines[0]}'"

    # Hapus string literal dalam tanda kutip ganda untuk menghindari false positive pada teks label
    code_without_strings = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', '""', mermaid_code)

    # Deteksi kurung siku ganda pada simpul tunggal (contoh: Node["Teks"]] )
    if re.search(r'(?<=\w)\[[^\]\n]*\]\]', code_without_strings):
        return False, "Terdapat penutup kurung siku ganda ']]' pada simpul berawalan tunggal '['."

    # Deteksi dangling <br/> di luar node pada baris relasi
    if re.search(r'(?:\]|\)|\})\s*<br\s*/?>', code_without_strings, re.IGNORECASE):
        return False, "Tag <br/> ditemukan di luar tanda kurung simpul. Semua pemisah baris harus berada di dalam label simpul bertanda kutip."

    # Deteksi unquoted label yang mengandung tanda kurung di dalam simpul kotak (penyebab utama error 'got PS')
    unquoted_paren_match = re.search(r'\b\w+\[[^"\]\n]*[\(\)][^"\]\n]*\]', code_without_strings)
    if unquoted_paren_match:
        return (
            False,
            (
                f"Label simpul mengandung tanda kurung tanpa tanda kutip dua: '{unquoted_paren_match.group(0)}'. "
                "Gunakan tanda kutip dua, contoh: A[\"Teks (detail)\"]."
            ),
        )

    # Deteksi Subgraph Self-Cycle (ID Subgraph digunakan sebagai simpul relasi di dalam dirinya sendiri)
    subgraph_stack_val: list[str] = []
    arrow_val_pat = re.compile(r"(-->|---|-.->|==>|<--|<---|<-.-|<==)")
    for line_idx, line_str in enumerate(lines):
        st = line_str.strip()
        m_s = re.match(r"^subgraph\s+([A-Za-z0-9_]+)\b", st, re.IGNORECASE)
        if m_s:
            subgraph_stack_val.append(m_s.group(1))
            continue
        if st.lower() == "end" and subgraph_stack_val:
            subgraph_stack_val.pop()
            continue

        if subgraph_stack_val and arrow_val_pat.search(st):
            curr_s = subgraph_stack_val[-1]
            if re.search(rf"\b{curr_s}\b", st):
                return (
                    False,
                    (
                        f"Subgraph '{curr_s}' digunakan sebagai simpul relasi di dalam dirinya sendiri pada baris {line_idx + 1} "
                        f"('{st}'). Ini memicu error fatal: 'Setting {curr_s} as parent of {curr_s} would create a cycle'. "
                        f"Gunakan simpul terpisah di dalam subgraph (contoh: {curr_s}_node) atau hubungkan antarsimpul secara langsung."
                    ),
                )

    # Deteksi Node ID yang dideklarasikan ulang dengan label berbeda (Conflicting Redeclaration)
    declared_nodes_val: dict[str, str] = {}
    for m in re.finditer(r'\b([A-Za-z0-9_]+)\["([^"\n]+)"\]', mermaid_code):
        nid = m.group(1)
        nlabel = m.group(2).strip()
        if nid in declared_nodes_val and declared_nodes_val[nid] != nlabel:
            return (
                False,
                (
                    f"Simpul '{nid}' dideklarasikan dengan dua label berbeda ('{declared_nodes_val[nid]}' vs '{nlabel}'). "
                    f"ID simpul Mermaid bersifat global di seluruh diagram. Gunakan ID unik untuk setiap simpul "
                    f"(contoh: {nid}_1 dan {nid}_2)."
                ),
            )
        declared_nodes_val[nid] = nlabel

    # Cek keseimbangan kurung pada kode di luar string literal
    open_brackets = code_without_strings.count("[")
    close_brackets = code_without_strings.count("]")
    open_braces = code_without_strings.count("{")
    close_braces = code_without_strings.count("}")
    open_parens = code_without_strings.count("(")
    close_parens = code_without_strings.count(")")

    # Toleransi kecil jika ada label khusus, namun beri warning jika selisih banyak
    if abs(open_brackets - close_brackets) > 1:
        return False, f"Ketidakseimbangan tanda kurung siku: [{open_brackets} vs ]{close_brackets}"
    if abs(open_braces - close_braces) > 1:
        return False, f"Ketidakseimbangan kurung kurawal: {{{open_braces}}} vs {{{close_braces}}}"
    if abs(open_parens - close_parens) > 1:
        return False, f"Ketidakseimbangan tanda kurung biasa: ({open_parens} vs ){close_parens}"

    return True, None


def render_mermaid_to_png(
    mermaid_code: str,
    output_path: str | Path | None = None,
    width: int = 1200,
    height: int = 800,
    background_color: str = "transparent",
    theme: str = "default",
    timeout: int = 60,
) -> tuple[bool, bytes | None, str | None]:
    """
    Render kode diagram Mermaid ke format gambar PNG menggunakan engine Mermaid CLI (pymmdc).
    Mengembalikan tuple (is_success, png_bytes, error_message).
    Jika rendering gagal, error_message berisi pesan error presisi dari compiler Mermaid CLI.
    """
    if not mermaid_code or not mermaid_code.strip():
        return False, None, "Kode Mermaid kosong"

    try:
        from mmdc import LocalMermaidConverter  # ty: ignore[unresolved-import]
    except ImportError:
        logger.warning("[Diagram:Render] Modul 'mmdc' (pymmdc) tidak ditemukan. Render visual dilewati.")
        return False, None, "Pustaka 'pymmdc' tidak terinstal"

    is_windows = platform.system() == "Windows"
    mmdc_cmd = "mmdc.cmd" if is_windows and shutil.which("mmdc.cmd") else "mmdc"

    try:
        conv = LocalMermaidConverter(
            mmdc_path=mmdc_cmd,
            timeout=timeout,
            validate_system=False,  # Hindari hardcoded check yang mencari executable mmdc tanpa ekstensi di Windows
        )
        conv.set_config(
            width=width,
            height=height,
            backgroundColor=background_color,
            theme=theme,
        )

        if output_path:
            out_file = Path(output_path)
            res = conv.convert_and_save(mermaid_code, str(out_file))
            if res.status.value == "success" and out_file.exists():
                png_bytes = out_file.read_bytes()
                return True, png_bytes, None
            clean_err = res.error_message or "Konversi Mermaid ke PNG gagal"
            return False, None, clean_err
        else:
            png_bytes = conv.convert_to_png(mermaid_code)
            if png_bytes:
                return True, png_bytes, None
            return False, None, "Output PNG kosong dari compiler Mermaid"

    except Exception as exc:  # noqa: BLE001
        err_msg = str(exc)
        logger.warning("[Diagram:Render] Gagal merender diagram Mermaid: %s", err_msg)
        return False, None, err_msg


def get_diagram_recommendation(diagram_type: DiagramTypeLiteral | str) -> DiagramFormatRecommendation:
    """Berikan rekomendasi format ekstraksi berdasarkan kategori diagram."""
    mermaid_compatible = {
        "flowchart": ("flowchart TD", "mermaid_code", "Cocok untuk alur kerja terstruktur"),
        "sequence_diagram": ("sequenceDiagram", "mermaid_code", "Cocok untuk urutan pesan/proses"),
        "class_diagram": ("classDiagram", "mermaid_code", "Cocok untuk hierarki kelas OOP"),
        "state_diagram": ("stateDiagram-v2", "mermaid_code", "Cocok untuk transisi state finite"),
        "er_diagram": ("erDiagram", "mermaid_code", "Cocok untuk skema relasi entitas/database"),
        "mindmap": ("mindmap", "mermaid_code", "Cocok untuk hierarki konsep & taksonomi"),
        "gantt_chart": ("gantt", "mermaid_code", "Cocok untuk jadwal & lini masa proyek"),
        "block_architecture": ("block-beta", "mermaid_code", "Cocok untuk diagram blok arsitektur"),
        "pin_diagram": ("flowchart LR", "mermaid_code", "Diagram pinout/koneksi kaki IC"),
        "memory_map": ("flowchart TD", "mermaid_code", "Peta alokasi memori atau register map"),
        "circuit_diagram": ("flowchart LR", "mermaid_code", "Diagram sirkuit logika atau interkoneksi"),
        "timing_diagram": ("sequenceDiagram", "mermaid_code", "Diagram waktu sinyal/timing diagram"),
        "git_graph": ("gitGraph", "mermaid_code", "Cocok untuk visualisasi alur branching git"),
        "generic_diagram": ("flowchart LR", "mermaid_code", "Diagram umum, gunakan representasi flowchart"),
    }

    if diagram_type in mermaid_compatible:
        syntax, fmt, rationale = mermaid_compatible[diagram_type]
        return DiagramFormatRecommendation(
            diagram_type=diagram_type,
            recommended_format=fmt,
            is_mermaid_compatible=True,
            suggested_syntax=syntax,
            rationale=rationale,
        )

    unsuitable_reasons = {
        "unsuitable_statistical_chart": "Grafik statistik kontinu (bar, line, scatter) lebih baik diekstrak ke tabel data Markdown dan deskripsi tren.",
        "unsuitable_map_or_spatial": "Peta geografis atau spasial tidak dapat diwakili oleh simpul graf Mermaid.",
        "unsuitable_photo_or_illustration": "Foto atau ilustrasi artistik memerlukan deskripsi visual naratif.",
        "unsuitable_complex_schematic": "Skematik sirkuit elektrik mikro atau CAD terlalu rumit untuk abstraksi Mermaid.",
        "non_diagram": "Konten berupa teks standar atau halaman polos.",
    }

    rationale = unsuitable_reasons.get(
        diagram_type,
        "Format visual tidak kompatibel dengan generator kode diagram relasional.",
    )
    return DiagramFormatRecommendation(
        diagram_type=diagram_type,
        recommended_format="text_description",
        is_mermaid_compatible=False,
        suggested_syntax=None,
        rationale=rationale,
    )


def classify_diagram_convertibility(
    image_path: str | Path,
    llm: BaseChatModel,
) -> DiagramConvertibilityResult:
    """
    Evaluasi visual: Apakah gambar mengandung diagram yang cocok dikonversi ke kode Mermaid?
    Mengembalikan `DiagramConvertibilityResult` berisi keputusan konvertibilitas dan format yang dianjurkan.
    """
    path_obj = Path(image_path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"File citra diagram tidak ditemukan: {path_obj}")

    image_uri = image_data_uri(path_obj)

    prompt = (
        "Analisis gambar ini dengan teliti untuk mengevaluasi apakah gambar ini berisi DIAGRAM yang cocok "
        "dikonversi ke sintaks kode diagram relasional (Mermaid.js).\n\n"
        "Kategori diagram yang SANGAT COCOK untuk Mermaid:\n"
        "  - Flowchart / Alur Proses (flowchart TD/LR)\n"
        "  - Sequence Diagram / Interaksi Pesan (sequenceDiagram)\n"
        "  - Entity Relationship Diagram / ERD (erDiagram)\n"
        "  - Class Diagram / OOP Hierarchy (classDiagram)\n"
        "  - State Machine / State Diagram (stateDiagram-v2)\n"
        "  - Mindmap / Pohon Konsep (mindmap)\n"
        "  - Block Architecture / Blok Sistem Komponen (block-beta / flowchart)\n"
        "  - Gantt Chart / Timeline (gantt)\n\n"
        "Kategori yang TIDAK COCOK untuk Mermaid:\n"
        "  - Grafik statistik data kuantitatif / kurva / bar chart (unsuitable_statistical_chart)\n"
        "  - Peta spasial / denah / geografis (unsuitable_map_or_spatial)\n"
        "  - Foto / Gambar ilustrasi bebas (unsuitable_photo_or_illustration)\n"
        "  - Skematik sirkuit mikro / blueprint CAD (unsuitable_complex_schematic)\n"
        "  - Dokumen teks biasa tanpa diagram (non_diagram)\n\n"
        "Berikan output HANYA dalam format JSON valid berikut tanpa penjelasan tambahan:\n"
        "{\n"
        '  "is_convertible": true/false,\n'
        '  "diagram_type": "flowchart" | "sequence_diagram" | "class_diagram" | "state_diagram" | "er_diagram" | "mindmap" | "gantt_chart" | "block_architecture" | "unsuitable_statistical_chart" | "unsuitable_map_or_spatial" | "unsuitable_photo_or_illustration" | "unsuitable_complex_schematic" | "non_diagram",\n'
        '  "recommended_format": "mermaid" | "text_description" | "markdown_table",\n'
        '  "mermaid_type": "flowchart" | "sequenceDiagram" | "erDiagram" | "classDiagram" | "stateDiagram-v2" | "mindmap" | "gantt" | "block-beta" | null,\n'
        '  "confidence": 0.0 - 1.0,\n'
        '  "reasoning": "Alasan singkat dalam bahasa Indonesia",\n'
        '  "nodes_or_entities": ["Entitas1", "Entitas2", ...]\n'
        "}"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": image_uri},
                },
            ],
        }
    ]

    try:
        resp = llm.invoke(messages)
        text_resp = str(resp.content).strip()
        logger.debug("[Diagram:Classify] Respon mentah VLM: %s", text_resp)

        # Cari JSON blok
        match = re.search(r"\{.*\}", text_resp, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                dtype = data.get("diagram_type", "generic_diagram")
                rec_fmt = data.get("recommended_format", "text_description")
                is_conv = bool(data.get("is_convertible", False))
                mtype = data.get("mermaid_type")
                conf = float(data.get("confidence", 0.9))
                reason = data.get("reasoning", "")
                nodes = data.get("nodes_or_entities", [])

                return DiagramConvertibilityResult(
                    is_convertible=is_conv,
                    diagram_type=cast(DiagramTypeLiteral, dtype),
                    recommended_format=str(rec_fmt),
                    mermaid_type=mtype,
                    confidence=conf,
                    reasoning=reason,
                    nodes_or_entities=nodes if isinstance(nodes, list) else [],
                )
            except (json.JSONDecodeError, ValueError) as json_err:
                logger.warning(
                    "[Diagram:Classify] Gagal decode JSON (%s): %s", json_err, text_resp
                )

        # Fallback heuristik berbasis teks
        lower_resp = text_resp.lower()
        if (
            "flowchart" in lower_resp
            or "alur" in lower_resp
            or "workflow" in lower_resp
        ):
            return DiagramConvertibilityResult(
                is_convertible=True,
                diagram_type="flowchart",
                recommended_format="mermaid",
                mermaid_type="flowchart",
                confidence=0.8,
                reasoning="Terdeteksi pola alur proses/flowchart dari analisis teks model.",
            )
        if "sequence" in lower_resp or "urutan" in lower_resp:
            return DiagramConvertibilityResult(
                is_convertible=True,
                diagram_type="sequence_diagram",
                recommended_format="mermaid",
                mermaid_type="sequenceDiagram",
                confidence=0.8,
                reasoning="Terdeteksi urutan interaksi aktor/komponen.",
            )

        return DiagramConvertibilityResult(
            is_convertible=False,
            diagram_type="generic_diagram",
            recommended_format="text_description",
            mermaid_type=None,
            confidence=0.6,
            reasoning="Diagram tidak memiliki struktur relasional diskrit yang terbukti cocok untuk Mermaid.",
        )

    except Exception:
        logger.exception(
            "[Diagram:Classify] Gagal mengklasifikasikan gambar diagram '%s'",
            path_obj.name,
        )
        return DiagramConvertibilityResult(
            is_convertible=False,
            diagram_type="generic_diagram",
            recommended_format="text_description",
            mermaid_type=None,
            confidence=0.0,
            reasoning="Terjadi error saat memanggil Vision LLM classifier.",
        )


def extract_diagram_to_mermaid(
    image_path: str | Path,
    llm: BaseChatModel,
    *,
    forced_diagram_type: str | None = None,
) -> DiagramExtractionResult:
    """
    Ekstrak citra diagram menjadi kode Mermaid.js terstruktur yang siap dirender,
    atau deskripsi teks komprehensif jika diagram tidak cocok untuk Mermaid.
    """
    path_obj = Path(image_path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"File citra tidak ditemukan: {path_obj}")

    image_uri = image_data_uri(path_obj)

    # 1. Evaluasi kelayakan jika tidak dipaksa
    convertibility: DiagramConvertibilityResult | None = None
    if not forced_diagram_type:
        convertibility = classify_diagram_convertibility(path_obj, llm)
        if not convertibility.is_convertible:
            desc_prompt = (
                "Gambar ini berisi elemen visual/grafik/diagram yang tidak cocok dijadikan diagram kode Mermaid. "
                "Berikan deskripsi terstruktur mengenai gambar ini mencakup:\n"
                "1. Judul / Tema Visual\n"
                "2. Komponen / Data Utama yang digambarkan\n"
                "3. Tren / Hubungan / Kesimpulan utama dari gambar\n"
                "Sajikan dalam format Markdown bersih."
            )
            desc_resp = llm.invoke(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": desc_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_uri
                                },
                            },
                        ],
                    }
                ]
            )
            return DiagramExtractionResult(
                status="unsuitable",
                is_mermaid=False,
                diagram_type=convertibility.diagram_type,
                mermaid_code=None,
                text_description=str(desc_resp.content).strip(),
                text_summary=str(desc_resp.content).strip(),
                reasoning=convertibility.reasoning,
                convertibility=convertibility,
            )

    # 2. Prompt Ekstraksi Mermaid.js
    syntax_hint = ""
    if forced_diagram_type:
        syntax_hint = f"Paksa gunakan tipe diagram: {forced_diagram_type}."
    elif convertibility and convertibility.mermaid_type:
        syntax_hint = f"Gunakan tipe diagram Mermaid: {convertibility.mermaid_type}."

    extract_prompt = (
        "Tugas Anda adalah mengekstrak seluruh elemen diagram, simpul (nodes), label teks, dan relasi berarah "
        "dari gambar ini menjadi KODE MERMAID.JS YANG VALID dan LENGKAP.\n\n"
        f"{syntax_hint}\n\n"
        "PANDUAN KETAT SINTAKS MERMAID (WAJIB DIIKUTI AGAR TIDAK PARSE ERROR):\n"
        "1. Baris pertama WAJIB deklarasi tipe diagram (gunakan 'flowchart TD' atau 'flowchart LR'; DILARANG menggunakan keyword 'graph').\n"
        "2. SEMUA LABEL SIMPUL WAJIB DIAPIT TANDA KUTIP DUA (\") DI DALAM BENTUK SIMPUL:\n"
        "   - CONTOH BENAR: A[\"<b>Hidup Saleh</b><br/>(Tit 1:8)\"] atau B((\"Lingkaran\")) atau Center{\"4 Syarat\"}\n"
        "   - CONTOH SALAH: A[<b>Hidup Saleh</b><br>(Tit 1:8)]  <-- FATAL ERROR! Karakter '(' tanpa tanda kutip memicu crash 'got PS'!\n"
        "3. DILARANG menggunakan tanda kurung siku ganda ']]' jika pembukanya hanya '[' (CONTOH SALAH: Node[\"Teks\"]] ).\n"
        "4. DILARANG meletakkan tag <br/> atau teks di luar tanda kurung simpul pada baris relasi (CONTOH SALAH: Node[\"A\"]<br/>(2) --> B).\n"
        "5. ID Simpul harus sederhana dan alfanumerik pendek (misal: A, B, node1, Step1).\n"
        "6. Untuk baris baru (line break) di dalam label teks, gunakan tag <br/> di dalam tanda kutip dua.\n"
        "7. PADA 'classDef', HANYA GUNAKAN PROPERTI CSS STANDAR:\n"
        "   - Properti yang diizinkan: fill, stroke, stroke-width, color, stroke-dasharray.\n"
        "   - DILARANG KERAS menggunakan atribut SVG seperti 'rx' atau 'ry' (contoh SALAH: rx:15, ry:15).\n"
        "8. Pertahankan semua label relasi/panah secara akurat (contoh: A -->|Label| B).\n"
        "9. DILARANG KERAS menggunakan ID subgraph sebagai simpul di dalam dirinya sendiri:\n"
        "   - CONTOH SALAH (di dalam 'subgraph UserMem'): UserMem -.-> LowerMem\n"
        "     Hal ini menyebabkan crash fatal: 'Setting UserMem as parent of UserMem would create a cycle'.\n"
        "     Hubungkan antarsimpul langsung (contoh: UpperMem -.-> LowerMem) atau buat simpul baru di dalamnya.\n"
        "10. ID SIMPUL BERSIFAT GLOBAL (DILARANG MENGGUNAKAN ID YANG SAMA UNTUK DUA LABEL BERBEDA):\n"
        "    - Jika ada dua kotak dengan teks berbeda di subgraph berbeda, WAJIB beri ID unik (contoh: B0_Block_Low dan B1_Block_Low, JANGAN keduanya dinamai Block_Low).\n"
        "11. JANGAN MENGULANG DEFINISI KURUNG LABEL PADA RELASI BERIKUTNYA:\n"
        "    - Jika Row8_Reg[\"EEDATA\"] sudah didefinisikan sebelumnya, relasi panah berikutnya cukup tulis: note1 -.-> Row8_Reg.\n"
        "12. DILARANG MEMBUNGKUS TEKS LABEL DENGAN KURUNG-KUTIP GANDA SEPERTI [\"('...')\"]:\n"
        "    - Gunakan format bersih: Node[\"Teks\"] (bukan Node[\"('Teks')\"]).\n"
        "13. Berikan output di dalam blok markdown ```mermaid\\n...\\n```.\n"
        "14. Tambahkan ringkasan 1-2 kalimat di bawah blok kode yang menjelaskan arti diagram tersebut."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": extract_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": image_uri},
                },
            ],
        }
    ]

    try:
        resp = llm.invoke(messages)
        text_resp = strip_thinking_process(str(resp.content).strip())

        # Ekstrak blok mermaid
        mermaid_match = re.search(
            r"```(?:mermaid)?\s*([\s\S]*?)\s*```", text_resp, re.IGNORECASE
        )
        mermaid_block: str | None = None
        summary_text: str | None = None

        if mermaid_match:
            mermaid_block = sanitize_mermaid_code(mermaid_match.group(1))
            # Ambil sisa teks sebagai summary
            summary_text = text_resp.replace(mermaid_match.group(0), "").strip()
        else:
            # Periksa apakah seluruh respon atau bagian respon dimulai dengan kata kunci Mermaid
            stripped_resp = text_resp.strip()
            first_line = (
                stripped_resp.splitlines()[0].strip().lower() if stripped_resp else ""
            )
            if any(first_line.startswith(kw) for kw in MERMAID_KEYWORDS):
                mermaid_block = sanitize_mermaid_code(stripped_resp)
                summary_text = "Diagram Mermaid berhasil diekstrak."
            else:
                summary_text = text_resp

        diag_type = cast(
            DiagramTypeLiteral,
            forced_diagram_type
            or (convertibility.diagram_type if convertibility else "flowchart"),
        )

        # Linter Compiler, CLI Rendering, & Multimodal Visual Self-Correction Loop
        MAX_MERMAID_RETRIES = 2
        rendered_png_bytes: bytes | None = None

        for attempt in range(MAX_MERMAID_RETRIES + 1):
            if not mermaid_block:
                break

            # 1. Jalankan auto-sanitizer
            mermaid_block = sanitize_mermaid_code(mermaid_block)
            if not mermaid_block:
                break

            # 2. Validasi dengan compiler linter statis
            is_valid, syntax_err = validate_mermaid_syntax(mermaid_block)
            if not is_valid:
                logger.warning(
                    "[Diagram:Extract] Sintaks Mermaid bermasalah pada validasi linter (percobaan %d): %s",
                    attempt + 1,
                    syntax_err,
                )
                if attempt < MAX_MERMAID_RETRIES:
                    fix_prompt = (
                        f"Kode Mermaid yang Anda hasilkan memiliki KESALAHAN SINTAKS:\n"
                        f"- Error Compiler/Linter: {syntax_err}\n\n"
                        f"Kode yang bermasalah:\n```mermaid\n{mermaid_block}\n```\n\n"
                        "PERBAIKI KODE DI ATAS DENGAN ATURAN KETAT BERIKUT:\n"
                        "1. Gunakan 'flowchart TD' atau 'flowchart LR' (DILARANG menggunakan 'graph').\n"
                        "2. Selalu gunakan tanda kutip ganda pada label simpul: Node[\"Teks Label\"].\n"
                        "3. DILARANG menggunakan penutup kurung ganda ']]' jika pembukanya hanya satu '['.\n"
                        "4. DILARANG meletakkan <br/> atau teks di luar tanda kurung simpul pada baris relasi.\n"
                        "5. DILARANG menggunakan ID subgraph sebagai simpul relasi panah di dalam subgraph itu sendiri.\n"
                        "6. DILARANG menggunakan ID simpul yang sama untuk dua label teks berbeda (gunakan ID unik per simpul).\n"
                        "7. JANGAN mengulang penulisan kurung label pada simpul yang sudah pernah dideklarasikan sebelumnya.\n"
                        "8. Outputkan HANYA blok kode ```mermaid\\n...\\n``` yang sudah diperbaiki tanpa teks tambahan."
                    )
                    try:
                        fix_resp = llm.invoke(
                            [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": fix_prompt},
                                        {"type": "image_url", "image_url": {"url": image_uri}},
                                    ],
                                }
                            ]
                        )
                        fix_text = strip_thinking_process(str(fix_resp.content).strip())
                        m_fix = re.search(r"```(?:mermaid)?\s*([\s\S]*?)\s*```", fix_text, re.IGNORECASE)
                        mermaid_block = m_fix.group(1).strip() if m_fix else fix_text.strip()
                        continue
                    except Exception as e_fix:  # noqa: BLE001
                        logger.warning("[Diagram:Extract] Gagal melakukan retry self-correction Mermaid: %s", e_fix)
                        break
                else:
                    mermaid_block = None
                    break

            # 3. Uji kompilasi & rendering via pymmdc / Mermaid CLI
            render_ok, png_data, render_err = render_mermaid_to_png(mermaid_block)
            is_env_error = any(
                marker in (render_err or "").lower()
                for marker in ("tidak terinstal", "chrome-headless-shell", "could not find chrome", "mmdc executable not found")
            )
            if not render_ok and render_err and not is_env_error:
                logger.warning(
                    "[Diagram:Extract] Compiler Mermaid CLI gagal pada percobaan ke-%d: %s",
                    attempt + 1,
                    render_err,
                )
                if attempt < MAX_MERMAID_RETRIES:
                    fix_cli_prompt = (
                        f"Kode Mermaid Anda menyebabkan ERROR saat dikompilasi oleh engine Mermaid CLI:\n"
                        f"- Pesan Error Compiler: {render_err}\n\n"
                        f"Kode yang bermasalah:\n```mermaid\n{mermaid_block}\n```\n\n"
                        "PERBAIKI KODE DI ATAS AGAR DAPAT DIKOMPILASI DENGAN SUKSES:\n"
                        "1. Perbaiki relasi atau struktur yang memicu error di atas (misal jika cycle error, pastikan ID subgraph tidak menjadi simpul di dalam dirinya sendiri).\n"
                        "2. Pastikan semua label simpul tetap menggunakan tanda kutip ganda yang valid.\n"
                        "3. Berikan HANYA blok kode ```mermaid\\n...\\n``` yang sudah diperbaiki."
                    )
                    try:
                        fix_resp = llm.invoke(
                            [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": fix_cli_prompt},
                                        {"type": "image_url", "image_url": {"url": image_uri}},
                                    ],
                                }
                            ]
                        )
                        fix_text = strip_thinking_process(str(fix_resp.content).strip())
                        m_fix = re.search(r"```(?:mermaid)?\s*([\s\S]*?)\s*```", fix_text, re.IGNORECASE)
                        mermaid_block = m_fix.group(1).strip() if m_fix else fix_text.strip()
                        continue
                    except Exception as e_cli_fix:  # noqa: BLE001
                        logger.warning("[Diagram:Extract] Gagal melakukan retry compiler CLI: %s", e_cli_fix)
                        break
                else:
                    logger.error(
                        "[Diagram:Extract] Kode Mermaid tetap gagal kompilasi setelah %d percobaan (%s).",
                        MAX_MERMAID_RETRIES + 1,
                        render_err,
                    )
                    mermaid_block = None
                    if not summary_text or "Mermaid berhasil" in summary_text:
                        summary_text = (
                            "Diagram visual terdeteksi pada dokumen, namun engine Mermaid CLI gagal "
                            f"mengompilasi struktur relasi kode diagram: {render_err}"
                        )
                    break

            # 4. Rendering Berhasil! Jalankan Multimodal Visual Verification Loop
            if render_ok and png_data:
                rendered_png_bytes = png_data
                logger.info("[Diagram:Extract] Render Mermaid CLI sukses. Menjalankan verifikasi visual multimodal.")
                try:
                    rendered_b64 = base64.b64encode(png_data).decode("utf-8")
                    rendered_uri = f"data:image/png;base64,{rendered_b64}"

                    verify_prompt = (
                        "Berikut adalah verifikasi visual diagram:\n"
                        "- Gambar 1: Potongan diagram asli dari dokumen sumber.\n"
                        "- Gambar 2: Gambar hasil rendering dari kode Mermaid yang baru Anda buat.\n\n"
                        f"Kode Mermaid saat ini:\n```mermaid\n{mermaid_block}\n```\n\n"
                        "TUGAS EVALUASI VISUAL:\n"
                        "Bandingkan Gambar 2 (hasil render) dengan Gambar 1 (asli):\n"
                        "1. Apakah semua simpul, teks, label, dan angka sudah lengkap?\n"
                        "2. Apakah arah panah atau alur relasi sudah benar dan tidak ada yang terlewat?\n"
                        "3. Apakah ada teks yang terpotong?\n\n"
                        "Jika diagram hasil render SUDAH AKURAT dan LENGKAP mencakup semua elemen dari gambar asli, jawab HANYA:\n"
                        "[CONFIRMED]\n\n"
                        "Jika ada ketidaksesuaian atau bagian yang kurang, berikan KODE MERMAID LENGKAP YANG SUDAH DIREVISI "
                        "di dalam blok ```mermaid ... ```."
                    )

                    verify_resp = llm.invoke(
                        [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": verify_prompt},
                                    {"type": "image_url", "image_url": {"url": image_uri}},
                                    {"type": "image_url", "image_url": {"url": rendered_uri}},
                                ],
                            }
                        ]
                    )
                    v_text = strip_thinking_process(str(verify_resp.content).strip())
                    if "[CONFIRMED]" not in v_text.upper():
                        m_revised = re.search(r"```(?:mermaid)?\s*([\s\S]*?)\s*```", v_text, re.IGNORECASE)
                        if m_revised:
                            revised_code = sanitize_mermaid_code(m_revised.group(1))
                            if revised_code:
                                rev_ok, rev_png, _ = render_mermaid_to_png(revised_code)
                                if rev_ok and rev_png:
                                    mermaid_block = revised_code
                                    rendered_png_bytes = rev_png
                                    logger.info("[Diagram:Extract] Kode Mermaid berhasil diselaraskan berdasarkan evaluasi visual multimodal.")
                except Exception as e_vis:  # noqa: BLE001
                    logger.warning("[Diagram:Extract] Verifikasi visual dilewati karena kendala teknis: %s", e_vis)

            # Selesai dengan sukses
            break

        return DiagramExtractionResult(
            status="success" if mermaid_block else "unsuitable",
            is_mermaid=bool(mermaid_block),
            diagram_type=diag_type,
            mermaid_code=mermaid_block,
            rendered_image_bytes=rendered_png_bytes,
            text_summary=summary_text
            or f"Diagram tipe {diag_type} berhasil diekstrak.",
            reasoning=convertibility.reasoning
            if convertibility
            else "Ekstraksi visual berhasil dijalankan.",
            convertibility=convertibility,
        )

    except Exception:
        logger.exception(
            "[Diagram:Extract] Gagal mengekstrak Mermaid dari '%s'",
            path_obj.name,
        )
        return DiagramExtractionResult(
            status="error",
            is_mermaid=False,
            diagram_type=cast(DiagramTypeLiteral, forced_diagram_type or "generic_diagram"),
            mermaid_code=None,
            text_summary="Gagal melakukan ekstraksi diagram visual.",
            reasoning="Terjadi error saat ekstraksi visual diagram.",
            convertibility=convertibility,
        )


__all__ = [
    "MERMAID_KEYWORDS",
    "classify_diagram_convertibility",
    "extract_diagram_to_mermaid",
    "get_diagram_recommendation",
    "render_mermaid_to_png",
    "sanitize_mermaid_code",
    "validate_mermaid_syntax",
]
