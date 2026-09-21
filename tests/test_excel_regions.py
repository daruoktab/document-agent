import gc
import sqlite3
import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.styles import Alignment, Font, PatternFill

from app.config import Settings
from app.excel import (
    _convert_excel_regions_to_pdf,
    _convert_workbook_to_xlsx_copy,
    persist_excel_native_data,
    survey_excel_workbook,
)
from app.pdf import pdf_page_count
from app.ppt import _find_libreoffice_binary


class TestExcelRegionSurvey(unittest.TestCase):
    def _build_side_by_side_workbook(self, path: Path) -> None:
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Billing"

        for start_column, month, values in (
            (2, "FEB 2026", (3287, 1_036_330_654, 113_993_234, 1_150_323_888)),
            (8, "MAR 2026", (3035, 951_306_917, 104_641_630, 1_055_948_547)),
        ):
            sheet.merge_cells(
                start_row=2,
                start_column=start_column,
                end_row=2,
                end_column=start_column + 4,
            )
            title = sheet.cell(2, start_column, f"INVOICE 1 {month}")
            title.font = Font(bold=True, color="FFFFFF", size=7)
            title.fill = PatternFill("solid", fgColor="000000")
            headers = ["Category", "Subs", "Current Charges", "Tax", "Charges + Tax"]
            for offset, header in enumerate(headers):
                cell = sheet.cell(3, start_column + offset, header)
                cell.font = Font(bold=True, color="FFFFFF", size=7)
                cell.fill = PatternFill("solid", fgColor="000000")
            sheet.cell(4, start_column, "New Sub MANUAL")
            for offset, value in enumerate(values, start=1):
                sheet.cell(4, start_column + offset, value)
            sheet.cell(5, start_column, "TOTAL")
            for offset in range(1, 5):
                source = sheet.cell(4, start_column + offset).coordinate
                sheet.cell(5, start_column + offset, f"={source}")

        sheet["N2"] = "Rotated note"
        sheet["N2"].alignment = Alignment(text_rotation=45)
        chart = BarChart()
        chart.add_data(Reference(sheet, min_col=3, min_row=3, max_row=5), titles_from_data=True)
        chart.anchor = "N8"
        sheet.add_chart(chart)
        workbook.save(path)

    def test_side_by_side_tables_become_separate_regions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "billing.xlsx"
            self._build_side_by_side_workbook(path)

            survey = survey_excel_workbook(
                path,
                settings=Settings(
                    excel_small_font_points=8,
                    excel_base_dpi=300,
                    excel_max_dpi=450,
                ),
            )

            regions = survey.sheets[0].regions
            invoice_regions = [region for region in regions if region.kind == "table"]
            self.assertEqual(len(invoice_regions), 2)
            self.assertEqual(
                {region.period for region in invoice_regions},
                {"FEB 2026", "MAR 2026"},
            )
            self.assertTrue(all(region.render_dpi >= 400 for region in invoice_regions))
            self.assertTrue(
                all(region.requires_vlm_reading for region in invoice_regions)
            )
            self.assertNotIn("N2", {cell.coordinate for region in invoice_regions for cell in region.cells})
            chart_regions = [region for region in regions if region.title == "Chart 1"]
            self.assertEqual(len(chart_regions), 1)
            self.assertTrue(chart_regions[0].requires_vlm_reading)

    def test_native_tables_group_matching_monthly_regions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workbook_path = root / "billing.xlsx"
            database_path = root / "billing.sqlite"
            self._build_side_by_side_workbook(workbook_path)
            survey = survey_excel_workbook(workbook_path)

            created = persist_excel_native_data(survey, database_path)

            self.assertEqual(len(created), 1)
            connection = sqlite3.connect(database_path)
            try:
                periods = {
                    row[0]
                    for row in connection.execute(
                        f'SELECT DISTINCT period FROM "{created[0]}"'
                    ).fetchall()
                }
                source_rows = connection.execute(
                    f'SELECT COUNT(*) FROM "{created[0]}"'
                ).fetchone()[0]
                evidence_count = connection.execute(
                    "SELECT COUNT(*) FROM excel_cells"
                ).fetchone()[0]
            finally:
                connection.close()
            gc.collect()

            self.assertEqual(periods, {"FEB 2026", "MAR 2026"})
            self.assertEqual(source_rows, 4)
            self.assertGreater(evidence_count, 20)

    @unittest.skipUnless(_find_libreoffice_binary(), "LibreOffice tidak tersedia")
    def test_region_renderer_produces_one_page_per_region(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workbook_path = root / "billing.xlsx"
            pdf_path = root / "billing.pdf"
            self._build_side_by_side_workbook(workbook_path)
            survey = survey_excel_workbook(workbook_path)
            invoice_regions = [
                region
                for region in survey.sheets[0].regions
                if region.kind == "table"
            ]

            rendered = _convert_excel_regions_to_pdf(
                workbook_path,
                pdf_path,
                root / "lo_profile",
                invoice_regions,
            )

            self.assertTrue(rendered)
            self.assertEqual(pdf_page_count(pdf_path), 2)

    @unittest.skipUnless(_find_libreoffice_binary(), "LibreOffice tidak tersedia")
    def test_survey_copy_recalculates_formula_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workbook_path = root / "billing.xlsx"
            self._build_side_by_side_workbook(workbook_path)

            recalculated = _convert_workbook_to_xlsx_copy(
                workbook_path, root / "recalculated"
            )
            survey = survey_excel_workbook(recalculated)
            formula_cells = [
                cell
                for region in survey.sheets[0].regions
                for cell in region.cells
                if cell.formula
            ]

            self.assertTrue(formula_cells)
            self.assertTrue(all(cell.value is not None for cell in formula_cells))


if __name__ == "__main__":
    unittest.main()
