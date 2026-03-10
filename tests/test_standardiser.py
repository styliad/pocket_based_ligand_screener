"""Tests for the pose standardiser module."""

from __future__ import annotations

from pathlib import Path

import pytest
from rdkit import Chem

from pocket_ligand_screener.standardiser.base import STANDARD_PROPS, StandardisedPoseRecord
from pocket_ligand_screener.standardiser.glide import GlideStandardiser


# ------------------------------------------------------------------
# GlideStandardiser – synthetic fixture
# ------------------------------------------------------------------


class TestGlideStandardiserSynthetic:
    """Tests using the synthetic multi-pose fixture."""

    def test_standardise_writes_and_records(self, tmp_sdf: Path, tmp_path: Path) -> None:
        gs = GlideStandardiser(tmp_sdf)
        out = gs.standardise(tmp_path / "std.sdf")
        assert out.exists()
        assert len(gs) == 6

    def test_ligand_idx_assignment(self, tmp_sdf: Path, tmp_path: Path) -> None:
        gs = GlideStandardiser(tmp_sdf)
        gs.standardise(tmp_path / "std.sdf")
        records = gs.records
        aspirin_idxs = {r.ligand_idx for r in records if r.molecule_name == "aspirin"}
        ibuprofen_idxs = {r.ligand_idx for r in records if r.molecule_name == "ibuprofen"}
        assert len(aspirin_idxs) == 1
        assert len(ibuprofen_idxs) == 1
        assert aspirin_idxs != ibuprofen_idxs

    def test_pose_idx_from_glide_posenum(self, tmp_sdf: Path, tmp_path: Path) -> None:
        gs = GlideStandardiser(tmp_sdf)
        gs.standardise(tmp_path / "std.sdf")
        aspirin_pose_idxs = sorted(
            r.pose_idx for r in gs.records if r.molecule_name == "aspirin"
        )
        assert aspirin_pose_idxs == [10, 20, 30]

    def test_docking_scores(self, tmp_sdf: Path, tmp_path: Path) -> None:
        gs = GlideStandardiser(tmp_sdf)
        gs.standardise(tmp_path / "std.sdf")
        scores = [r.docking_score for r in gs.records]
        assert all(isinstance(s, float) for s in scores)
        assert scores[0] == pytest.approx(-7.5)

    def test_docking_algorithm(self, tmp_sdf: Path, tmp_path: Path) -> None:
        gs = GlideStandardiser(tmp_sdf)
        gs.standardise(tmp_path / "std.sdf")
        assert all(r.docking_algorithm == "glide" for r in gs.records)

    def test_output_sdf_has_standard_props(self, tmp_sdf: Path, tmp_path: Path) -> None:
        gs = GlideStandardiser(tmp_sdf)
        out = gs.standardise(tmp_path / "std.sdf")

        suppl = Chem.SDMolSupplier(str(out), removeHs=False)
        mols = [m for m in suppl if m is not None]
        assert len(mols) == 6

        for mol in mols:
            for prop in STANDARD_PROPS:
                assert mol.HasProp(prop), f"Missing property: {prop}"

        first = mols[0]
        assert first.GetProp("molecule_name") == "aspirin"
        assert first.GetIntProp("ligand_idx") == 0
        assert first.GetIntProp("pose_idx") == 10
        assert first.GetProp("docking_algorithm") == "glide"

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            GlideStandardiser(tmp_path / "nonexistent.sdf")

    def test_repr_and_len(self, tmp_sdf: Path, tmp_path: Path) -> None:
        gs = GlideStandardiser(tmp_sdf)
        assert len(gs) == 0
        gs.standardise(tmp_path / "std.sdf")
        assert len(gs) == 6
        assert "GlideStandardiser" in repr(gs)
        assert "n_poses=6" in repr(gs)

    def test_records_property_returns_copy(self, tmp_sdf: Path, tmp_path: Path) -> None:
        gs = GlideStandardiser(tmp_sdf)
        gs.standardise(tmp_path / "std.sdf")
        records = gs.records
        records.clear()
        assert len(gs) == 6  # internal list unchanged

    def test_record_has_no_mol(self, tmp_sdf: Path, tmp_path: Path) -> None:
        """Verify that records are lightweight (no Mol objects in memory)."""
        gs = GlideStandardiser(tmp_sdf)
        gs.standardise(tmp_path / "std.sdf")
        for r in gs.records:
            assert isinstance(r, StandardisedPoseRecord)
            assert not hasattr(r, "mol")


# ------------------------------------------------------------------
# GlideStandardiser – real Glide SDF (skipped if not present)
# ------------------------------------------------------------------


class TestGlideStandardiserReal:
    """Integration tests using the real Glide rotigotine SDF."""

    def test_standardise_real_file(
        self, glide_real_sdf: Path | None, tmp_path: Path
    ) -> None:
        if glide_real_sdf is None:
            pytest.skip("Real Glide SDF not available")
        gs = GlideStandardiser(glide_real_sdf)
        gs.standardise(tmp_path / "rotigotine_std.sdf")
        records = gs.records
        assert len(records) == 24
        assert all(r.molecule_name == "rotigotine" for r in records)
        assert all(r.docking_algorithm == "glide" for r in records)

    def test_roundtrip_real(
        self, glide_real_sdf: Path | None, tmp_path: Path
    ) -> None:
        if glide_real_sdf is None:
            pytest.skip("Real Glide SDF not available")
        gs = GlideStandardiser(glide_real_sdf)
        out = gs.standardise(tmp_path / "rotigotine_std.sdf")

        suppl = Chem.SDMolSupplier(str(out), removeHs=False)
        mols = [m for m in suppl if m is not None]
        assert len(mols) == 24
        assert all(m.HasProp("molecule_name") for m in mols)
        assert all(m.HasProp("pose_idx") for m in mols)

        # Verify pose_idx values match original i_i_glide_posenum
        pose_idxs = [m.GetIntProp("pose_idx") for m in mols]
        assert 97 in pose_idxs  # first record from the real file
