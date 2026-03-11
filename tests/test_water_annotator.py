"""Tests for the water_annotator module."""

from __future__ import annotations

from pathlib import Path

import pytest

from water_annotator.base import WaterCategory, classify_water
from water_annotator.watermap import WaterMapAnnotator

TEST_DATA = Path(__file__).parent / "test_data"
DRD5 = TEST_DATA / "drd5_example"
CSV_PATH = DRD5 / "8irv_wm_data.csv"
INPUT_PDB = DRD5 / "8irv_wm_waters.pdb"
EXPECTED_PDB = DRD5 / "prepared_8irv_wm_waters_fixed.pdb"


# ------------------------------------------------------------------
# classify_water
# ------------------------------------------------------------------


class TestClassifyWater:
    def test_very_unhappy(self):
        assert classify_water(6.68) == WaterCategory.VERY_UNHAPPY

    def test_unhappy(self):
        assert classify_water(2.76) == WaterCategory.UNHAPPY

    def test_bulk_like(self):
        assert classify_water(1.54) == WaterCategory.BULK_LIKE
        assert classify_water(0.0) == WaterCategory.BULK_LIKE

    def test_happy(self):
        assert classify_water(-1.82) == WaterCategory.HAPPY

    def test_boundaries(self):
        # Exactly on the threshold values
        assert classify_water(3.5) == WaterCategory.UNHAPPY   # not > 3.5
        assert classify_water(2.0) == WaterCategory.UNHAPPY    # >= 2.0
        # -1.0 is not > -1.0, so HAPPY
        assert classify_water(-1.0) == WaterCategory.HAPPY


# ------------------------------------------------------------------
# WaterMapAnnotator – using real WaterMap output files
# ------------------------------------------------------------------


class TestWaterMapAnnotator:
    """Test with real WaterMap input (CSV + raw PDB) against expected output."""

    def test_parse_sites_count(self):
        annotator = WaterMapAnnotator(CSV_PATH, INPUT_PDB)
        sites = annotator._parse_water_sites()
        assert len(sites) == 82

    def test_dg_values_match_csv(self):
        annotator = WaterMapAnnotator(CSV_PATH, INPUT_PDB)
        sites = annotator._parse_water_sites()
        assert sites[0].dG == pytest.approx(6.68)
        assert sites[3].dG == pytest.approx(-1.82)
        assert sites[-1].dG == pytest.approx(0.48)

    def test_coordinates_from_pdb(self):
        annotator = WaterMapAnnotator(CSV_PATH, INPUT_PDB)
        sites = annotator._parse_water_sites()
        assert sites[0].x == pytest.approx(-14.358)
        assert sites[0].y == pytest.approx(-12.808)
        assert sites[0].z == pytest.approx(40.384)

    def test_annotate_writes_pdb(self, tmp_path: Path):
        annotator = WaterMapAnnotator(CSV_PATH, INPUT_PDB)
        out = annotator.annotate(
            tmp_path / "waters.pdb", title="8irv_wm_waters",
        )
        assert out.exists()
        content = out.read_text()
        assert "TITLE" in content
        assert "HETATM" in content
        assert "END" in content

    def test_annotate_correct_categories(self, tmp_path: Path):
        annotator = WaterMapAnnotator(CSV_PATH, INPUT_PDB)
        out = annotator.annotate(tmp_path / "waters.pdb")
        content = out.read_text()
        lines = [
            line for line in content.splitlines()
            if line.startswith("HETATM")
        ]
        assert len(lines) == 82
        assert "WVU" in lines[0]   # site 1, dG=6.68
        assert "WHP" in lines[3]   # site 4, dG=-1.82
        assert "WBL" in lines[5]   # site 6, dG=1.54
        assert "WUN" in lines[2]   # site 3, dG=2.76

    def test_annotate_bfactor_is_dg(self, tmp_path: Path):
        annotator = WaterMapAnnotator(CSV_PATH, INPUT_PDB)
        out = annotator.annotate(tmp_path / "waters.pdb")
        content = out.read_text()
        lines = [
            line for line in content.splitlines()
            if line.startswith("HETATM")
        ]
        # B-factor field: columns 60-66
        bfactor_1 = float(lines[0][60:66])
        assert bfactor_1 == pytest.approx(6.68)

    def test_output_matches_expected(self, tmp_path: Path):
        """Compare HETATM lines against the reference output PDB."""
        annotator = WaterMapAnnotator(CSV_PATH, INPUT_PDB)
        out = annotator.annotate(
            tmp_path / "waters.pdb", title="8irv_wm_waters",
        )
        out_lines = [
            line for line in out.read_text().splitlines()
            if line.startswith("HETATM")
        ]
        expected_lines = [
            line for line in EXPECTED_PDB.read_text().splitlines()
            if line.startswith("HETATM")
        ]
        assert len(out_lines) == len(expected_lines)
        for i, (got, expected) in enumerate(
            zip(out_lines, expected_lines),
        ):
            # Compare resname (category), coords, occupancy, bfactor
            assert got[17:20] == expected[17:20], (
                f"Site {i+1}: category mismatch"
            )
            assert float(got[30:38]) == pytest.approx(
                float(expected[30:38]),
            )
            assert float(got[60:66]) == pytest.approx(
                float(expected[60:66]),
            )

    def test_sites_property(self, tmp_path: Path):
        annotator = WaterMapAnnotator(CSV_PATH, INPUT_PDB)
        annotator.annotate(tmp_path / "waters.pdb")
        assert len(annotator) == 82
        assert len(annotator.sites) == 82

    def test_missing_csv_raises(self):
        with pytest.raises(FileNotFoundError):
            WaterMapAnnotator("nonexistent.csv", INPUT_PDB)

    def test_missing_pdb_raises(self):
        with pytest.raises(FileNotFoundError):
            WaterMapAnnotator(CSV_PATH, "nonexistent.pdb")
