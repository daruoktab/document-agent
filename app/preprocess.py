"""
Image preprocessing utilities untuk OCR & VLM.

Menyediakan fungsi untuk:
  - Memperbaiki rotasi gambar berdasarkan EXIF orientation metadata.
  - Meningkatkan kontras dokumen (AutoContrast) untuk teks pudar / scan gelap.
  - Memastikan gambar dalam format RGB standar tanpa merusak resolusi teks.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import NamedTuple

from PIL import Image, ImageEnhance, ImageOps

logger = logging.getLogger(__name__)


class PreprocessedImage(NamedTuple):
    processed_path: str
    original_path: str
    is_modified: bool
    dimensions: tuple[int, int]


def preprocess_image(
    image_path: str | Path,
    *,
    auto_orient: bool = True,
    enhance_contrast: bool = True,
    contrast_factor: float = 1.2,
    output_dir: str | Path | None = None,
) -> PreprocessedImage:
    """
    Pra-pemrosesan gambar dokumen sebelum dikirim ke OCR atau VLM.

    Args:
        image_path: Path file gambar input.
        auto_orient: Jika True, sesuaikan orientasi berdasarkan metadata EXIF.
        enhance_contrast: Jika True, terapkan penyesuaian kontras untuk dokumen teks.
        contrast_factor: Faktor pengali kontras (1.0 = tidak berubah, 1.2-1.5 cocok untuk scan pudar).
        output_dir: Direktori untuk menyimpan gambar hasil proses (default: tempdir).

    Returns:
        PreprocessedImage(processed_path, original_path, is_modified, dimensions)
    """
    orig_path = Path(image_path)
    if not orig_path.exists():
        raise FileNotFoundError(f"File gambar tidak ditemukan: {orig_path}")

    img = Image.open(orig_path)
    modified = False

    # 1. Auto-orientasi EXIF
    if auto_orient:
        try:
            transposed = ImageOps.exif_transpose(img)
            if transposed is not None:
                img = transposed
                modified = True
        except (OSError, ValueError, TypeError) as exc:
            logger.debug("Lewati penyesuaian orientasi EXIF: %s", exc)

    # 2. Pastikan RGB (bukan RGBA / Grayscale / P)
    if img.mode != "RGB":
        img = img.convert("RGB")
        modified = True

    # 3. Kontras dokumen
    if enhance_contrast:
        try:
            # Auto-contrast untuk meratakan rentang histogram
            img = ImageOps.autocontrast(img, cutoff=0.5)
            if contrast_factor != 1.0:
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(contrast_factor)
            modified = True
        except (OSError, ValueError, TypeError) as exc:
            logger.debug("Lewati penyesuaian kontras dokumen: %s", exc)

    dimensions = (img.width, img.height)

    # Simpan hasil jika ada perubahan atau format perlu diseragamkan
    if modified:
        if output_dir:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"proc_{orig_path.stem}.jpg"
        else:
            with tempfile.NamedTemporaryFile(
                prefix=f"proc_{orig_path.stem}_", suffix=".jpg", delete=False
            ) as temp_file:
                out_path = Path(temp_file.name)

        img.save(str(out_path), format="JPEG", quality=95)
        return PreprocessedImage(
            processed_path=str(out_path),
            original_path=str(orig_path),
            is_modified=True,
            dimensions=dimensions,
        )

    return PreprocessedImage(
        processed_path=str(orig_path),
        original_path=str(orig_path),
        is_modified=False,
        dimensions=dimensions,
    )


def rotate_image_right_angle(
    image_path: str | Path,
    degrees_clockwise: int,
    *,
    output_dir: str | Path | None = None,
) -> str:
    """Rotasi citra 0/90/180/270 derajat searah jarum jam dan simpan sebagai PNG."""
    if degrees_clockwise not in {0, 90, 180, 270}:
        raise ValueError("degrees_clockwise harus salah satu dari 0, 90, 180, 270")
    source_path = Path(image_path)
    if degrees_clockwise == 0:
        return str(source_path)

    if output_dir:
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / "oriented_page.png"
    else:
        with tempfile.NamedTemporaryFile(
            prefix=f"rot{degrees_clockwise}_{source_path.stem}_",
            suffix=".png",
            delete=False,
        ) as temp_file:
            target_path = Path(temp_file.name)

    with Image.open(source_path) as image:
        image.convert("RGB").rotate(-degrees_clockwise, expand=True).save(
            target_path, format="PNG"
        )
    return str(target_path.resolve())
