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

    def test_select_best_pose_per_pocket(self, pocket_csv, interactions_df):
        from pocket_ligand_screener.screener.residue_contact import ResidueContactScorer
        from pocket_ligand_screener.screener.combined import (
            score_all_poses,
            select_best_pose_per_pocket,
        )

        scorer_a = ResidueContactScorer(pocket_csv, pocket_name="pocket_A")
        scorers = {"pocket_A": scorer_a}

        scores_df = score_all_poses(interactions_df, scorers)
        best = select_best_pose_per_pocket(scores_df, rank_by="residue_tversky")
        assert len(best) == 1  # one pocket → one best pose
