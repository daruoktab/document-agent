"""Multimodal DSPy extraction, reference metrics and offline GEPA optimization."""

from __future__ import annotations

import base64
import json
import re
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
from difflib import SequenceMatcher
from io import BytesIO
from typing import Any

import dspy
from dspy.teleprompt.gepa.instruction_proposal import MultiModalInstructionProposer
from PIL import Image, ImageDraw

from .config import Settings, get_settings
from .learning_store import LearningStore, digest
from .multi_page import (
    collapse_consecutive_duplicate_blocks,
    strip_page_markers,
    strip_thinking_process,
)
from .prompts import SPEC_METADATA, SYSTEM_DOCUMENT_EXTRACTOR, build_extraction_prompt


def model_fingerprint(settings: Settings | None = None) -> str:
    s = settings or get_settings()
    return digest(
        json.dumps(
            [
                s.vlm_model,
                s.vlm_base_url,
                s.vlm_temperature,
                s.vlm_max_tokens,
                s.vlm_enable_thinking,
                SYSTEM_DOCUMENT_EXTRACTOR,
                [build_extraction_prompt(specs=name) for name in sorted(SPEC_METADATA)],
            ],
            ensure_ascii=False,
        ).encode()
    )


def build_learning_lm(settings: Settings | None = None) -> dspy.LM:
    s = settings or get_settings()
    return dspy.LM(
        model=f"openai/{s.vlm_model}",
        api_base=s.vlm_base_url,
        api_key=s.vlm_api_key,
        temperature=s.vlm_temperature,
        max_tokens=s.vlm_max_tokens,
        timeout=s.vlm_timeout,
        cache=False,
        num_retries=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": s.vlm_enable_thinking}},
    )


def probe_model() -> str:
    """One small synthetic vision request; never submits user documents."""
    image = Image.new("RGB", (360, 120), "white")
    ImageDraw.Draw(image).text((20, 35), "BUKU 100", fill="black", font_size=32)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    settings = replace(get_settings(), vlm_timeout=15, vlm_max_tokens=512)
    lm = build_learning_lm(settings)
    output = predict_page(
        PageProgram("Salin teks dan angka persis dari gambar."),
        lm,
        dspy.Image(
            url="data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        ),
        "Salin tulisan pada gambar tanpa penjelasan tambahan.",
    )
    if "100" not in output or "buku" not in output.casefold():
        raise ValueError(
            "Model merespons, tetapi belum berhasil membaca gambar uji BUKU 100."
        )
    return "Koneksi, masukan gambar, dan keluaran DSPy berhasil diuji."


class PageSignature(dspy.Signature):
    """Baca gambar dokumen dan keluarkan teks serta tabel dalam Markdown."""

    image: dspy.Image = dspy.InputField(desc="Gambar halaman asli")
    context: str = dspy.InputField(
        desc="Aturan ekstraksi, jenis dokumen, dan konteks halaman sebelumnya"
    )
    markdown: str = dspy.OutputField(
        desc="Hanya hasil Markdown, pertahankan teks, angka, dan susunan tabel"
    )


def clean_markdown(raw: str) -> str:
    text = strip_thinking_process(raw.strip())
    if text.startswith("```markdown") and text.endswith("```"):
        text = text[len("```markdown") : -3].strip()
    elif text.startswith("```md") and text.endswith("```"):
        text = text[len("```md") : -3].strip()
    elif text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    text = collapse_consecutive_duplicate_blocks(strip_page_markers(text)).strip()
    if not text:
        raise ValueError("Model tidak menghasilkan teks ekstraksi.")
    return text


class PageProgram(dspy.Module):
    def __init__(self, instructions: str = SYSTEM_DOCUMENT_EXTRACTOR) -> None:
        super().__init__()
        self.extract = dspy.Predict(PageSignature.with_instructions(instructions))

    def forward(self, image: dspy.Image, context: str) -> dspy.Prediction:
        prediction = self.extract(image=image, context=context)
        return dspy.Prediction(markdown=clean_markdown(prediction.markdown))


def predict_page(program: PageProgram, lm: Any, image: dspy.Image, context: str) -> str:
    # Context-local configuration also works in Streamlit worker threads.
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter(use_json_adapter_fallback=False)):
        return program(image=image, context=context).markdown


def _parts(markdown: str) -> tuple[str, list[tuple[str, ...]]]:
    prose: list[str] = []
    rows: list[tuple[str, ...]] = []
    for line in markdown.splitlines():
        line = line.strip()
        if line.startswith("|") and line.endswith("|"):
            cells = tuple(
                re.sub(r"\s+", " ", cell.strip())
                for cell in re.split(r"(?<!\\)\|", line[1:-1])
            )
            if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
                rows.append(cells)
        elif line:
            prose.append(re.sub(r"\s+", " ", line))
    return "\n".join(prose), rows


def score_markdown(reference: str, prediction: str) -> dict[str, Any]:
    gold_text, gold_rows = _parts(reference)
    pred_text, pred_rows = _parts(prediction)
    text_score = SequenceMatcher(None, gold_text, pred_text, autojunk=False).ratio()
    table_score = SequenceMatcher(None, gold_rows, pred_rows, autojunk=False).ratio()
    gold_numbers = Counter(re.findall(r"\d+(?:[.,]\d+)*", reference))
    pred_numbers = Counter(re.findall(r"\d+(?:[.,]\d+)*", prediction))
    feedback = []
    if text_score < 1:
        feedback.append(
            "Teks berubah atau hilang; cocokkan ejaan dan urutan dengan koreksi."
        )
    if gold_rows != pred_rows:
        feedback.append(
            f"Isi/urutan/kolom tabel berbeda: acuan {len(gold_rows)} baris, hasil {len(pred_rows)} baris."
        )
    if gold_numbers != pred_numbers:
        feedback.append(
            f"Angka hilang/berubah: {dict(gold_numbers - pred_numbers)}; angka tambahan: {dict(pred_numbers - gold_numbers)}."
        )
    if not prediction.strip():
        text_score = table_score = 0.0
    return {
        "text": text_score,
        "table": table_score,
        "score": (text_score + table_score) / 2,
        "feedback": " ".join(feedback) or "Teks dan tabel sesuai koreksi.",
    }


def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    result = score_markdown(gold.corrected, pred.markdown)
    return dspy.Prediction(
        score=result["score"],
        feedback=result["feedback"] + "\nKoreksi acuan:\n" + gold.corrected,
    )


def image_for_example(example: dict[str, Any]) -> dspy.Image:
    suffix = example["suffix"].lower().lstrip(".")
    mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
    uri = f"data:image/{mime};base64," + base64.b64encode(example["image"]).decode()
    return dspy.Image(url=uri)


def evaluate_examples(examples: list[dict[str, Any]], predict) -> dict[str, Any]:
    results = []
    for example in examples:
        prediction = predict(example)
        result = score_markdown(example["corrected"], prediction)
        results.append(
            {"example_id": example["id"], "prediction": prediction, **result}
        )
    return {
        **{
            key: sum(row[key] for row in results) / len(results)
            for key in ("text", "table", "score")
        },
        "pages": results,
    }


def is_improvement(baseline: dict[str, Any], candidate: dict[str, Any]) -> bool:
    return (
        candidate["score"] > baseline["score"] + 1e-9
        and candidate["text"] >= baseline["text"]
        and candidate["table"] >= baseline["table"]
    )


def optimize(store: LearningStore, max_metric_calls: int = 50) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage

    from .llm import build_vlm

    if max_metric_calls < 1:
        raise ValueError("Batas pemanggilan metrik harus positif.")
    examples = store.examples()
    splits = store.dataset_split(examples)
    if not any(_parts(e["corrected"])[1] for e in splits["test"]):
        raise ValueError(
            "Kelompok pengujian perlu koreksi yang memuat tabel untuk menilai kualitas tabel."
        )
    baseline_run = store.active_run()
    fingerprint = model_fingerprint()
    if baseline_run and baseline_run["model_fingerprint"] != fingerprint:
        raise ValueError(
            "Konfigurasi model berbeda dari versi aktif. Pulihkan baseline terlebih dahulu."
        )
    run: dict[str, Any] = {
        "status": "running",
        "eligible": False,
        "created_at": datetime.now(UTC).isoformat(),
        "baseline_version": baseline_run["id"] if baseline_run else "baseline",
        "model_fingerprint": fingerprint,
        "example_ids": [e["id"] for e in examples],
        "splits": {name: [e["id"] for e in subset] for name, subset in splits.items()},
        "max_metric_calls": max_metric_calls,
    }
    store.save_run(run)
    try:
        lm = build_learning_lm()
        instructions = (
            baseline_run["instructions"] if baseline_run else SYSTEM_DOCUMENT_EXTRACTOR
        )
        student = PageProgram(instructions)

        def as_dspy(example):
            return dspy.Example(
                image=image_for_example(example),
                context=example["context"],
                corrected=example["corrected"],
            ).with_inputs("image", "context")

        with dspy.context(
            lm=lm, adapter=dspy.ChatAdapter(use_json_adapter_fallback=False)
        ):
            optimizer = dspy.GEPA(
                metric=gepa_metric,
                reflection_lm=lm,
                instruction_proposer=MultiModalInstructionProposer(),
                max_metric_calls=max_metric_calls,
                num_threads=1,
                track_stats=True,
                seed=0,
            )
            candidate = optimizer.compile(
                student,
                trainset=[as_dspy(e) for e in splits["train"]],
                valset=[as_dspy(e) for e in splits["validation"]],
            )
        run["instructions"] = candidate.extract.signature.instructions
        # Test exactly the instruction-only artifact that production will load.
        deployed_candidate = PageProgram(run["instructions"])
        legacy_lm = build_vlm() if baseline_run is None else None

        def baseline_predict(example):
            image = image_for_example(example)
            if legacy_lm is not None:
                response = legacy_lm.invoke(
                    [
                        SystemMessage(content=SYSTEM_DOCUMENT_EXTRACTOR),
                        HumanMessage(
                            content=[
                                {"type": "text", "text": example["context"]},
                                {"type": "image_url", "image_url": {"url": image.url}},
                            ]
                        ),
                    ]
                )
                return clean_markdown(str(response.content or ""))
            return predict_page(student, lm, image, example["context"])

        run["baseline_metrics"] = evaluate_examples(splits["test"], baseline_predict)
        run["candidate_metrics"] = evaluate_examples(
            splits["test"],
            lambda e: predict_page(
                deployed_candidate, lm, image_for_example(e), e["context"]
            ),
        )
        run["eligible"] = is_improvement(
            run["baseline_metrics"], run["candidate_metrics"]
        )
        run["status"] = "completed"
        store.save_run(run)
        return run
    except BaseException as exc:
        run["status"] = "failed"
        # Do not persist raw provider exceptions (may contain URLs, keys or payloads).
        run["error"] = type(exc).__name__
        store.save_run(run)
        raise
