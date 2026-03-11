"""Tests for the WaterFLAP water annotator."""

from __future__ import annotations

from pathlib import Path

import pytest

from water_annotator.base import WaterCategory
from water_annotator.waterflap import WaterFLAPAnnotator

TEST_DATA = Path(__file__).parent / "test_data"
INPUT_PDB = TEST_DATA / "drd5_example" / "WFapo_8IRV.pdb"
EXPECTED_PDB = TEST_DATA / "drd5_example" / "WFapo_8IRV_fixed.pdb"


# ------------------------------------------------------------------
# WaterFLAPAnnotator – parsing
# ------------------------------------------------------------------


class TestWaterFLAPAnnotatorParsing:
    def test_parse_sites_count(self):
        annotator = WaterFLAPAnnotator(INPUT_PDB)
        sites = annotator._parse_water_sites()
        assert len(sites) == 66

    def test_dg_values(self):
        annotator = WaterFLAPAnnotator(INPUT_PDB)
        sites = annotator._parse_water_sites()
        # First site: dG = -2.7
        assert sites[0].dG == pytest.approx(-2.7)
        # Last site: dG = -0.0 (atom 92)
        assert sites[-1].dG == pytest.approx(0.0)

    def test_coordinates(self):
        annotator = WaterFLAPAnnotator(INPUT_PDB)
        sites = annotator._parse_water_sites()
        # First site (atom 1)
        assert sites[0].x == pytest.approx(-7.480)
        assert sites[0].y == pytest.approx(-10.590)
        assert sites[0].z == pytest.approx(60.000)

    def test_site_ids_preserved(self):
        annotator = WaterFLAPAnnotator(INPUT_PDB)
        sites = annotator._parse_water_sites()
        # IDs have gaps (no atom 5, 15, etc.)
        assert sites[0].site_id == 1
        assert sites[3].site_id == 4
        assert sites[4].site_id == 6  # gap: no 5


# ------------------------------------------------------------------
# WaterFLAPAnnotator – classification
# ------------------------------------------------------------------


class TestWaterFLAPClassification:
    """Test WaterFLAP-specific boundary conventions."""

    def test_very_unhappy(self):
        annotator = WaterFLAPAnnotator(INPUT_PDB)
        # dG = 5.0 (atom 13)
        assert annotator._classify(5.0) == WaterCategory.VERY_UNHAPPY

    def test_unhappy_at_boundary(self):
        """WaterFLAP uses >= 2.0 as the UNHAPPY boundary (inclusive)."""
        annotator = WaterFLAPAnnotator(INPUT_PDB)
        assert annotator._classify(2.0) == WaterCategory.UNHAPPY

    def test_unhappy(self):
        annotator = WaterFLAPAnnotator(INPUT_PDB)
        assert annotator._classify(2.1) == WaterCategory.UNHAPPY

    def test_bulk_like(self):
        annotator = WaterFLAPAnnotator(INPUT_PDB)
        assert annotator._classify(1.5) == WaterCategory.BULK_LIKE
        assert annotator._classify(0.0) == WaterCategory.BULK_LIKE

    def test_happy(self):
        annotator = WaterFLAPAnnotator(INPUT_PDB)
        assert annotator._classify(-2.7) == WaterCategory.HAPPY

    def test_happy_at_boundary(self):
        annotator = WaterFLAPAnnotator(INPUT_PDB)
        assert annotator._classify(-1.0) == WaterCategory.HAPPY


# ------------------------------------------------------------------
# WaterFLAPAnnotator – annotate output
# ------------------------------------------------------------------


class TestWaterFLAPAnnotatorOutput:
    def test_annotate_writes_pdb(self, tmp_path: Path):
        annotator = WaterFLAPAnnotator(INPUT_PDB)
        out = annotator.annotate(tmp_path / "waters.pdb", title="WFapo_8IRV")
        assert out.exists()
        content = out.read_text()
        assert "TITLE" in content
        assert "HETATM" in content
        assert "END" in content

    def test_annotate_site_count(self, tmp_path: Path):
        annotator = WaterFLAPAnnotator(INPUT_PDB)
        out = annotator.annotate(tmp_path / "waters.pdb")
        lines = [l for l in out.read_text().splitlines() if l.startswith("HETATM")]
        assert len(lines) == 66

    def test_annotate_correct_categories(self, tmp_path: Path):
        annotator = WaterFLAPAnnotator(INPUT_PDB)
        out = annotator.annotate(tmp_path / "waters.pdb")
        lines = [l for l in out.read_text().splitlines() if l.startswith("HETATM")]
        # site 1, dG=-2.7 → HAPPY
        assert lines[0][17:20] == "WHP"
        # site 13 (index 11), dG=5.0 → VERY UNHAPPY
        assert lines[11][17:20] == "WVU"
        # site 18 (index 15), dG=2.0 → UNHAPPY (boundary case)
        assert lines[15][17:20] == "WUN"
        # site 11 (index 9), dG=0.0 → BULK-LIKE
        assert lines[9][17:20] == "WBL"

    def test_annotate_bfactor_is_dg(self, tmp_path: Path):
        annotator = WaterFLAPAnnotator(INPUT_PDB)
        out = annotator.annotate(tmp_path / "waters.pdb")
        lines = [l for l in out.read_text().splitlines() if l.startswith("HETATM")]
        bfactor_1 = float(lines[0][60:66])
        assert bfactor_1 == pytest.approx(-2.7)

    def test_output_categories_match_expected(self, tmp_path: Path):
        """Compare category assignments against the reference PDB."""
        annotator = WaterFLAPAnnotator(INPUT_PDB)
        out = annotator.annotate(tmp_path / "waters.pdb")
        out_lines = [
            l for l in out.read_text().splitlines() if l.startswith("HETATM")
        ]
        expected_lines = [
            l for l in EXPECTED_PDB.read_text().splitlines() if l.startswith("HETATM")
        ]
        assert len(out_lines) == len(expected_lines)
        for i, (got, expected) in enumerate(zip(out_lines, expected_lines)):
            # Compare category (resname)
            assert got[17:20] == expected[17:20], (
                f"Site {i + 1}: category mismatch: "
                f"got {got[17:20]!r}, expected {expected[17:20]!r}"
            )
            # Compare coordinates
            assert float(got[30:38]) == pytest.approx(float(expected[30:38]))
            assert float(got[38:46]) == pytest.approx(float(expected[38:46]))
            assert float(got[46:54]) == pytest.approx(float(expected[46:54]))
            # Compare dG in bfactor
            assert float(got[60:66]) == pytest.approx(float(expected[60:66]))

    def test_sites_property(self, tmp_path: Path):
        annotator = WaterFLAPAnnotator(INPUT_PDB)
        annotator.annotate(tmp_path / "waters.pdb")
        assert len(annotator) == 66
        assert len(annotator.sites) == 66

    def test_missing_pdb_raises(self):
        with pytest.raises(FileNotFoundError):
            WaterFLAPAnnotator("nonexistent.pdb")
