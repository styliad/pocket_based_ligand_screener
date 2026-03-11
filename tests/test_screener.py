"""Tests for the screener module (residue contact, surface overlap, combined)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def pocket_csv(tmp_path: Path) -> Path:
    """Minimal pocket CSV with two pockets."""
    rows = [
        "gridmap,pocket,protein,residue_type,residue_number,chain,atom_name,distance_angstrom",
        "C1=,pocket_A,test.pdb,ALA,10,A,CA,2.5",
        "C1=,pocket_A,test.pdb,GLY,20,A,CA,2.8",
        "C1=,pocket_A,test.pdb,VAL,30,A,CB,2.1",
        "C1=,pocket_B,test.pdb,LEU,40,A,CA,2.3",
        "C1=,pocket_B,test.pdb,ILE,50,A,CB,2.6",
    ]
    p = tmp_path / "pocket.csv"
    p.write_text("\n".join(rows))
    return p


@pytest.fixture()
def interactions_df() -> pd.DataFrame:
    """Interaction rows for two poses contacting different residues."""
    return pd.DataFrame({
        "docked_ligand_index": [0, 0, 0, 0, 1, 1, 1],
        "interaction_type": ["Hydrophobic"] * 7,
        "ligand_atom_indices": ["(0,)"] * 7,
        "ligand_atom_types": ["('C',)"] * 7,
        "residue_name": ["ALA", "GLY", "PHE", "LEU", "ALA", "VAL", "LEU"],
        "residue_number": [10, 20, 99, 40, 10, 30, 40],
        "residue_atom_indices": ["(1,)"] * 7,
        "residue_atom_types": ["('C',)"] * 7,
        "interaction_distance": [3.5] * 7,
        "functional_groups": ["('No_fg',)"] * 7,
    })


@pytest.fixture()
def pocket_npz(tmp_path: Path) -> Path:
    """Minimal NPZ with two pockets of known vertex positions."""
    verts_a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64)
    verts_b = np.array([[10, 10, 10], [11, 10, 10]], dtype=np.float64)
    metadata = {
        "pockets": [
            {"name": "pocket_A", "num_vertices": 4, "num_triangles": 2},
            {"name": "pocket_B", "num_vertices": 2, "num_triangles": 1},
        ]
    }
    p = tmp_path / "pockets.npz"
    np.savez(
        p,
        pocket_A=verts_a,
        pocket_B=verts_b,
        metadata=json.dumps(metadata),
    )
    return p


# ===========================================================================
# ResidueContactScorer
# ===========================================================================

class TestResidueContactScorer:

    def test_score_count(self, pocket_csv, interactions_df):
        from pocket_ligand_screener.screener.residue_contact import ResidueContactScorer

        scorer = ResidueContactScorer(pocket_csv, pocket_name="pocket_A")
        assert scorer.pocket_residues == {("ALA", 10), ("GLY", 20), ("VAL", 30)}

        pose0 = interactions_df[interactions_df["docked_ligand_index"] == 0]
        # Pose 0 contacts ALA 10, GLY 20 (in pocket) + PHE 99, LEU 40 (not in pocket_A)
        assert scorer.score(pose0) == 2.0

    def test_score_coverage(self, pocket_csv, interactions_df):
        from pocket_ligand_screener.screener.residue_contact import ResidueContactScorer

        scorer = ResidueContactScorer(pocket_csv, pocket_name="pocket_A")
        pose0 = interactions_df[interactions_df["docked_ligand_index"] == 0]
        # 2 out of 3 pocket residues → 2/3
        assert scorer.score_coverage(pose0) == pytest.approx(2 / 3)

    def test_score_jaccard(self, pocket_csv, interactions_df):
        from pocket_ligand_screener.screener.residue_contact import ResidueContactScorer

        scorer = ResidueContactScorer(pocket_csv, pocket_name="pocket_A")
        pose0 = interactions_df[interactions_df["docked_ligand_index"] == 0]
        # Pose contacts: {ALA10, GLY20, PHE99, LEU40}; Pocket: {ALA10, GLY20, VAL30}
        # Intersection = 2, Union = 5
        assert scorer.score_jaccard(pose0) == pytest.approx(2 / 5)

    def test_score_tversky(self, pocket_csv, interactions_df):
        from pocket_ligand_screener.screener.residue_contact import ResidueContactScorer

        scorer = ResidueContactScorer(pocket_csv, pocket_name="pocket_A")
        pose0 = interactions_df[interactions_df["docked_ligand_index"] == 0]
        # intersection=2, pocket_only=1 (VAL30), pose_only=2 (PHE99, LEU40)
        # T = 2 / (2 + 1.0*1 + 0.3*2) = 2 / 3.6
        assert scorer.score_tversky(pose0, alpha=1.0, beta=0.3) == pytest.approx(2 / 3.6)

    def test_score_all(self, pocket_csv, interactions_df):
        from pocket_ligand_screener.screener.residue_contact import ResidueContactScorer

        scorer = ResidueContactScorer(pocket_csv, pocket_name="pocket_A")
        pose0 = interactions_df[interactions_df["docked_ligand_index"] == 0]
        result = scorer.score_all(pose0)
        assert set(result.keys()) == {"count", "coverage", "jaccard", "tversky"}
        assert result["count"] == 2.0

    def test_score_empty_pose(self, pocket_csv):
        from pocket_ligand_screener.screener.residue_contact import ResidueContactScorer

        scorer = ResidueContactScorer(pocket_csv, pocket_name="pocket_A")
        empty = pd.DataFrame(columns=["residue_name", "residue_number"])
        assert scorer.score(empty) == 0.0
        assert scorer.score_tversky(empty) == 0.0

    def test_annotate_interactions(self, pocket_csv, interactions_df):
        from pocket_ligand_screener.screener.residue_contact import ResidueContactScorer

        scorer = ResidueContactScorer(pocket_csv, pocket_name="pocket_A")
        result = scorer.annotate_interactions(interactions_df)
        assert "pocket_name" in result.columns
        # ALA 10 row should get pocket_A
        ala_row = result[(result["residue_name"] == "ALA") & (result["residue_number"] == 10)]
        assert all(ala_row["pocket_name"] == "pocket_A")
        # PHE 99 should get None
        phe_row = result[result["residue_name"] == "PHE"]
        assert all(phe_row["pocket_name"].isna())

    def test_invalid_pocket_name(self, pocket_csv):
        from pocket_ligand_screener.screener.residue_contact import ResidueContactScorer

        with pytest.raises(ValueError, match="No rows found"):
            ResidueContactScorer(pocket_csv, pocket_name="nonexistent")


class TestAnnotateAllPockets:

    def test_annotate_all_pockets(self, pocket_csv, interactions_df):
        from pocket_ligand_screener.screener.residue_contact import (
            ResidueContactScorer,
            annotate_all_pockets,
        )

        scorer_a = ResidueContactScorer(pocket_csv, pocket_name="pocket_A")
        scorer_b = ResidueContactScorer(pocket_csv, pocket_name="pocket_B")
        scorers = {"pocket_A": scorer_a, "pocket_B": scorer_b}

        result = annotate_all_pockets(interactions_df, scorers)
        assert "pocket_name" in result.columns

        # LEU 40 is in pocket_B — should appear with pocket_B
        leu_rows = result[(result["residue_name"] == "LEU") & (result["residue_number"] == 40)]
        assert "pocket_B" in leu_rows["pocket_name"].tolist()

        # PHE 99 is in no pocket — should have None
        phe_rows = result[result["residue_name"] == "PHE"]
        assert all(phe_rows["pocket_name"].isna())


# ===========================================================================
# SurfaceOverlapScorer
# ===========================================================================

class TestSurfaceOverlapScorer:

    def test_load_all_pockets(self, pocket_npz):
        from pocket_ligand_screener.screener.surface_overlap import SurfaceOverlapScorer

        scorer = SurfaceOverlapScorer(pocket_npz)
        assert set(scorer.pockets.keys()) == {"pocket_A", "pocket_B"}
        assert scorer.pockets["pocket_A"].shape == (4, 3)

    def test_load_single_pocket(self, pocket_npz):
        from pocket_ligand_screener.screener.surface_overlap import SurfaceOverlapScorer

        scorer = SurfaceOverlapScorer(pocket_npz, pocket_name="pocket_A")
        assert list(scorer.pockets.keys()) == ["pocket_A"]

    def test_invalid_pocket_name(self, pocket_npz):
        from pocket_ligand_screener.screener.surface_overlap import SurfaceOverlapScorer

        with pytest.raises(ValueError, match="not found"):
            SurfaceOverlapScorer(pocket_npz, pocket_name="nonexistent")

    def test_score_full_overlap(self, pocket_npz):
        from pocket_ligand_screener.screener.surface_overlap import SurfaceOverlapScorer

        scorer = SurfaceOverlapScorer(pocket_npz, pocket_name="pocket_A", distance_cutoff=3.0)
        # Ligand atoms sitting right on the vertices
        ligand_coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
        assert scorer.score(ligand_coords) == pytest.approx(1.0)

    def test_score_partial_overlap(self, pocket_npz):
        from pocket_ligand_screener.screener.surface_overlap import SurfaceOverlapScorer

        scorer = SurfaceOverlapScorer(pocket_npz, pocket_name="pocket_A", distance_cutoff=0.5)
        # Only near first vertex
        ligand_coords = np.array([[0.1, 0.1, 0.0]])
        score = scorer.score(ligand_coords)
        assert 0.0 < score < 1.0

    def test_score_no_overlap(self, pocket_npz):
        from pocket_ligand_screener.screener.surface_overlap import SurfaceOverlapScorer

        scorer = SurfaceOverlapScorer(pocket_npz, pocket_name="pocket_A", distance_cutoff=0.5)
        # Ligand far from pocket
        ligand_coords = np.array([[100, 100, 100]])
        assert scorer.score(ligand_coords) == pytest.approx(0.0)

    def test_score_count(self, pocket_npz):
        from pocket_ligand_screener.screener.surface_overlap import SurfaceOverlapScorer

        scorer = SurfaceOverlapScorer(pocket_npz, pocket_name="pocket_A", distance_cutoff=3.0)
        ligand_coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
        assert scorer.score_count(ligand_coords) == 4

    def test_score_all_pockets(self, pocket_npz):
        from pocket_ligand_screener.screener.surface_overlap import SurfaceOverlapScorer

        scorer = SurfaceOverlapScorer(pocket_npz, distance_cutoff=0.5)
        ligand_coords = np.array([[0, 0, 0]])  # near pocket_A only
        result = scorer.score_all_pockets(ligand_coords)
        assert result["pocket_A"] > 0
        assert result["pocket_B"] == pytest.approx(0.0)


# ===========================================================================
# Combined scoring
# ===========================================================================

class TestCombinedScoring:

    def test_score_all_poses_residue_only(self, pocket_csv, interactions_df):
        from pocket_ligand_screener.screener.residue_contact import ResidueContactScorer
        from pocket_ligand_screener.screener.combined import score_all_poses

        scorer_a = ResidueContactScorer(pocket_csv, pocket_name="pocket_A")
        scorer_b = ResidueContactScorer(pocket_csv, pocket_name="pocket_B")
        scorers = {"pocket_A": scorer_a, "pocket_B": scorer_b}

        result = score_all_poses(interactions_df, scorers)
        # 2 poses x 2 pockets = 4 rows
        assert len(result) == 4
        assert set(result.columns) >= {
            "docked_ligand_index", "pocket_name",
            "residue_count", "residue_tversky", "combined_score",
        }

    def test_select_best_pose(self, pocket_csv, interactions_df):
        from pocket_ligand_screener.screener.residue_contact import ResidueContactScorer
        from pocket_ligand_screener.screener.combined import (
            score_all_poses,
            select_best_pose,
        )

        scorer_a = ResidueContactScorer(pocket_csv, pocket_name="pocket_A")
        scorer_b = ResidueContactScorer(pocket_csv, pocket_name="pocket_B")
        scorers = {"pocket_A": scorer_a, "pocket_B": scorer_b}

        scores_df = score_all_poses(interactions_df, scorers)
        best = select_best_pose(scores_df, rank_by="residue_tversky")
        # Should return rows for the winning pose across all pockets
        assert len(best) == 2  # one row per pocket for the best pose
        assert best["docked_ligand_index"].nunique() == 1  # single winning pose
        assert "aggregated_score" in best.columns


# ===========================================================================
# Real data: DRD5 example
# ===========================================================================

DRD5_DIR = Path(__file__).resolve().parent / "test_data" / "drd5_example"

_drd5_available = (
    (DRD5_DIR / "pocket_residue_contacts_ang_3.csv").exists()
    and (DRD5_DIR / "pockets.npz").exists()
    and (DRD5_DIR / "annotated_interactions.csv").exists()
)

_glide_sdf_available = (
    Path(__file__).resolve().parent / "test_data" / "glide_real_sdf_1.sdf"
).exists()


@pytest.mark.skipif(not _drd5_available, reason="DRD5 example data not available")
class TestRealResidueContact:
    """Residue contact scoring against real DRD5 pocket data."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from pocket_ligand_screener.screener.residue_contact import ResidueContactScorer

        self.pocket_csv = DRD5_DIR / "pocket_residue_contacts_ang_3.csv"
        self.interactions_df = pd.read_csv(DRD5_DIR / "annotated_interactions.csv")
        self.pocket_names = ["C1=_pocket_1", "C1=_pocket_2", "C1=_pocket_3"]
        self.scorers = {
            name: ResidueContactScorer(self.pocket_csv, pocket_name=name)
            for name in self.pocket_names
        }

    def test_pocket_residues_loaded(self):
        for name, scorer in self.scorers.items():
            assert len(scorer.pocket_residues) > 0, f"Pocket {name} has no residues"

    def test_all_poses_scored(self):
        pose_indices = sorted(self.interactions_df["docked_ligand_index"].unique())
        assert len(pose_indices) == 24  # 24 docked poses

        for pose_idx in pose_indices:
            pose_df = self.interactions_df[
                self.interactions_df["docked_ligand_index"] == pose_idx
            ]
            for name, scorer in self.scorers.items():
                scores = scorer.score_all(pose_df)
                assert 0.0 <= scores["coverage"] <= 1.0
                assert 0.0 <= scores["jaccard"] <= 1.0
                assert 0.0 <= scores["tversky"] <= 1.0
                assert scores["count"] >= 0.0

    def test_best_pose_contacts_pocket_1(self):
        """The best pose for the main binding pocket should have non-trivial overlap."""
        scorer = self.scorers["C1=_pocket_1"]
        best_score = 0.0
        for pose_idx, pose_df in self.interactions_df.groupby("docked_ligand_index"):
            s = scorer.score_coverage(pose_df)
            best_score = max(best_score, s)
        # The co-crystallised ligand pose should cover a meaningful fraction
        assert best_score > 0.3, f"Best coverage for pocket_1 is only {best_score:.3f}"

    def test_annotate_all_pockets(self):
        from pocket_ligand_screener.screener.residue_contact import annotate_all_pockets

        result = annotate_all_pockets(self.interactions_df, self.scorers)
        assert "pocket_name" in result.columns
        # At least some rows should be annotated with a pocket
        assert result["pocket_name"].notna().sum() > 0
        # Some rows should have no pocket (contacts outside all pockets)
        assert result["pocket_name"].isna().sum() > 0


@pytest.mark.skipif(not _drd5_available, reason="DRD5 example data not available")
class TestRealSurfaceOverlap:
    """Surface overlap scoring against real DRD5 pocket vertices."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from pocket_ligand_screener.screener.surface_overlap import SurfaceOverlapScorer

        self.npz_path = DRD5_DIR / "pockets.npz"
        self.scorer = SurfaceOverlapScorer(self.npz_path, distance_cutoff=2.5)

    def test_all_pockets_loaded(self):
        assert set(self.scorer.pockets.keys()) == {
            "C1=_pocket_1", "C1=_pocket_2", "C1=_pocket_3",
        }
        assert self.scorer.pockets["C1=_pocket_1"].shape == (188, 3)
        assert self.scorer.pockets["C1=_pocket_2"].shape == (152, 3)
        assert self.scorer.pockets["C1=_pocket_3"].shape == (54, 3)

    @pytest.mark.skipif(not _glide_sdf_available, reason="glide_real_sdf_1.sdf not available")
    def test_score_real_poses(self):
        from rdkit import Chem
        from pocket_ligand_screener.screener.surface_overlap import coords_from_mol

        sdf_path = Path(__file__).resolve().parent / "test_data" / "glide_real_sdf_1.sdf"
        supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)

        for i, mol in enumerate(supplier):
            if mol is None:
                continue
            coords = coords_from_mol(mol)
            scores = self.scorer.score_all_pockets(coords)
            for pocket_name, cov in scores.items():
                assert 0.0 <= cov <= 1.0, (
                    f"Pose {i}, pocket {pocket_name}: coverage {cov} out of range"
                )

    @pytest.mark.skipif(not _glide_sdf_available, reason="glide_real_sdf_1.sdf not available")
    def test_best_pose_covers_pocket_1(self):
        """At least one pose should meaningfully overlap with the main pocket."""
        from rdkit import Chem
        from pocket_ligand_screener.screener.surface_overlap import coords_from_mol

        sdf_path = Path(__file__).resolve().parent / "test_data" / "glide_real_sdf_1.sdf"
        supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)

        best_cov = 0.0
        for mol in supplier:
            if mol is None:
                continue
            coords = coords_from_mol(mol)
            cov = self.scorer.score(coords, pocket_name="C1=_pocket_1")
            best_cov = max(best_cov, cov)

        assert best_cov > 0.1, f"Best surface coverage for pocket_1 is only {best_cov:.3f}"


@pytest.mark.skipif(
    not (_drd5_available and _glide_sdf_available),
    reason="DRD5 example data or glide SDF not available",
)
class TestRealCombined:
    """Combined scoring end-to-end with real DRD5 data."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from pocket_ligand_screener.screener.residue_contact import ResidueContactScorer
        from pocket_ligand_screener.screener.surface_overlap import SurfaceOverlapScorer

        self.pocket_csv = DRD5_DIR / "pocket_residue_contacts_ang_3.csv"
        self.interactions_df = pd.read_csv(DRD5_DIR / "annotated_interactions.csv")
        self.pocket_names = ["C1=_pocket_1", "C1=_pocket_2", "C1=_pocket_3"]
        self.residue_scorers = {
            name: ResidueContactScorer(self.pocket_csv, pocket_name=name)
            for name in self.pocket_names
        }
        self.surface_scorer = SurfaceOverlapScorer(
            DRD5_DIR / "pockets.npz", distance_cutoff=2.5,
        )
        self.sdf_path = Path(__file__).resolve().parent / "test_data" / "glide_real_sdf_1.sdf"

    def test_score_all_poses_combined(self):
        from rdkit import Chem
        from pocket_ligand_screener.screener.combined import score_all_poses

        supplier = Chem.SDMolSupplier(str(self.sdf_path), removeHs=False)
        scores_df = score_all_poses(
            self.interactions_df,
            self.residue_scorers,
            surface_scorer=self.surface_scorer,
            sdf_supplier=supplier,
        )
        # 24 poses × 3 pockets = 72 rows
        assert len(scores_df) == 24 * 3
        assert all(scores_df["combined_score"] >= 0.0)
        assert all(scores_df["combined_score"] <= 1.0)

    def test_select_best_pose(self):
        from rdkit import Chem
        from pocket_ligand_screener.screener.combined import (
            score_all_poses,
            select_best_pose,
        )

        supplier = Chem.SDMolSupplier(str(self.sdf_path), removeHs=False)
        scores_df = score_all_poses(
            self.interactions_df,
            self.residue_scorers,
            surface_scorer=self.surface_scorer,
            sdf_supplier=supplier,
        )
        best = select_best_pose(scores_df)
        # Single winning pose, one row per pocket
        assert best["docked_ligand_index"].nunique() == 1
        assert len(best) == 3
        assert "aggregated_score" in best.columns
        assert best["aggregated_score"].iloc[0] > 0.0
