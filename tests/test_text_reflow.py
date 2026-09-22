import unittest
from typing import Any, cast

from app.schemas import OCRRegion
from app.text_reflow import reflow_regions


def region(
    index: int,
    kind: str,
    text: str,
    bbox: tuple[int, int, int, int],
) -> OCRRegion:
    return OCRRegion(
        index=index,
        label=kind,
        kind=cast(Any, kind),
        text=text,
        lines=text.splitlines(),
        reading_order=index,
        bbox_model=tuple(float(value) for value in bbox),
        bbox_pixels=bbox,
    )


class TestTextReflowIntegration(unittest.TestCase):
    def test_reorders_columns_and_rebuilds_paragraph(self) -> None:
        regions = [
            region(1, "title", "Dokumen", (0, 0, 1000, 80)),
            region(2, "text", "strated dengan baik.", (520, 100, 950, 250)),
            region(3, "text", "Model telah demon-", (40, 100, 480, 250)),
        ]

        markdown, reason = reflow_regions(
            regions,
            page_width=1000,
            specs=["bilingual_journal"],
            enabled=True,
        )

        self.assertEqual(reason, "applied")
        self.assertEqual(markdown, "# Dokumen\n\nModel telah demonstrated dengan baik.")

    def test_table_page_uses_original_paddle_markdown(self) -> None:
        regions = [
            region(1, "text", "Ringkasan.", (20, 20, 400, 80)),
            region(2, "table", "| A | B |", (20, 100, 900, 400)),
        ]

        markdown, reason = reflow_regions(
            regions,
            page_width=1000,
            specs=["plain"],
            enabled=True,
        )

        self.assertIsNone(markdown)
        self.assertEqual(reason, "non_text_region")

    def test_form_spec_is_not_reflowed(self) -> None:
        markdown, reason = reflow_regions(
            [region(1, "text", "Nama: Budi", (20, 20, 400, 80))],
            page_width=1000,
            specs=["signature_form"],
            enabled=True,
        )

        self.assertIsNone(markdown)
        self.assertEqual(reason, "ineligible_spec")


if __name__ == "__main__":
    unittest.main()
