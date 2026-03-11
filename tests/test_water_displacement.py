"""Tests for the WaterDisplacementScorer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from water_annotator.base import WaterCategory, WaterSite
from pocket_ligand_screener.screener.water_displacement import WaterDisplacementScorer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mixed_water_sites() -> list[WaterSite]:
    """Six water sites spanning all four categories."""
    return [
        WaterSite(site_id=1, x=0.0, y=0.0, z=0.0, dG=5.0),   # VU
        WaterSite(site_id=2, x=3.0, y=0.0, z=0.0, dG=4.0),   # VU
        WaterSite(site_id=3, x=6.0, y=0.0, z=0.0, dG=2.5),   # UN
        WaterSite(site_id=4, x=9.0, y=0.0, z=0.0, dG=2.0),   # UN (boundary, inclusive)
        WaterSite(site_id=5, x=12.0, y=0.0, z=0.0, dG=0.5),  # BL
        WaterSite(site_id=6, x=15.0, y=0.0, z=0.0, dG=-2.0), # HP
    ]


@pytest.fixture()
def unhappy_only_sites() -> list[WaterSite]:
    """Three sites all in target categories."""
    return [
        WaterSite(site_id=1, x=0.0, y=0.0, z=0.0, dG=5.0),
        WaterSite(site_id=2, x=3.0, y=0.0, z=0.0, dG=3.0),
        WaterSite(site_id=3, x=6.0, y=0.0, z=0.0, dG=2.0),
    ]


# ===========================================================================
# Construction
# ===========================================================================

class TestConstruction:

    def test_target_count(self, mixed_water_sites):
        scorer = WaterDisplacementScorer(mixed_water_sites)
        # Sites 1, 2 (VU) + 3, 4 (UN) = 4 targets
        assert scorer.n_target_waters == 4

    def test_no_targets_when_all_happy(self):
        sites = [WaterSite(site_id=i, x=float(i), y=0, z=0, dG=-3.0) for i in range(5)]
        scorer = WaterDisplacementScorer(sites)
        assert scorer.n_target_waters == 0

    def test_custom_target_categories(self, mixed_water_sites):
        scorer = WaterDisplacementScorer(
            mixed_water_sites,
            target_categories=frozenset({WaterCategory.VERY_UNHAPPY}),
        )
        # Only sites 1, 2 (dG > 3.5)
        assert scorer.n_target_waters == 2

    def test_custom_thresholds(self, mixed_water_sites):
        scorer = WaterDisplacementScorer(
            mixed_water_sites,
            very_unhappy_threshold=4.5,
            unhappy_threshold=3.0,
        )
        # VU: dG > 4.5 → site 1 (5.0) only
        # UN: 3.0 <= dG <= 4.5 → site 2 (4.0)
        # So 2 targets
        assert scorer.n_target_waters == 2

    def test_target_sites_property(self, unhappy_only_sites):
        scorer = WaterDisplacementScorer(unhappy_only_sites)
        assert len(scorer.target_sites) == 3
        assert all(isinstance(s, WaterSite) for s in scorer.target_sites)

    def test_empty_input(self):
        scorer = WaterDisplacementScorer([])
        assert scorer.n_target_waters == 0


# ===========================================================================
# Scoring
# ===========================================================================

class TestScoring:

    def test_displace_all_targets(self, mixed_water_sites):
        scorer = WaterDisplacementScorer(mixed_water_sites, displacement_cutoff=1.0)
        # Ligand atoms sitting right on the 4 target waters
        ligand = np.array([[0, 0, 0], [3, 0, 0], [6, 0, 0], [9, 0, 0]], dtype=np.float64)
        assert scorer.score(ligand) == 4

    def test_displace_some_targets(self, mixed_water_sites):
        scorer = WaterDisplacementScorer(mixed_water_sites, displacement_cutoff=1.0)
        # Only near sites 1 and 3
        ligand = np.array([[0.5, 0, 0], [6.5, 0, 0]], dtype=np.float64)
        assert scorer.score(ligand) == 2

    def test_displace_none(self, mixed_water_sites):
        scorer = WaterDisplacementScorer(mixed_water_sites, displacement_cutoff=1.0)
        ligand = np.array([[100, 100, 100]], dtype=np.float64)
        assert scorer.score(ligand) == 0

    def test_only_displaces_targets_not_others(self, mixed_water_sites):
        """Ligand near a bulk-like water should not increase the score."""
        scorer = WaterDisplacementScorer(mixed_water_sites, displacement_cutoff=1.0)
        # Near site 5 (BL, x=12) and site 6 (HP, x=15) only
        ligand = np.array([[12, 0, 0], [15, 0, 0]], dtype=np.float64)
        assert scorer.score(ligand) == pytest.approx(0.0)

    def test_displaced_indices(self, mixed_water_sites):
        scorer = WaterDisplacementScorer(mixed_water_sites, displacement_cutoff=1.0)
        ligand = np.array([[0, 0, 0]], dtype=np.float64)
        idx = scorer.displaced_indices(ligand)
        assert len(idx) == 1
        # The displaced target site should be site_id=1 (first target)
        displaced_site = scorer.target_sites[list(idx)[0]]
        assert displaced_site.site_id == 1

    def test_score_with_no_targets(self):
        sites = [WaterSite(site_id=1, x=0, y=0, z=0, dG=-3.0)]
        scorer = WaterDisplacementScorer(sites)
        ligand = np.array([[0, 0, 0]], dtype=np.float64)
        assert scorer.score(ligand) == 0

    def test_empty_ligand_coords(self, mixed_water_sites):
        scorer = WaterDisplacementScorer(mixed_water_sites, displacement_cutoff=1.0)
        ligand = np.empty((0, 3), dtype=np.float64)
        assert scorer.score(ligand) == 0

    def test_displacement_cutoff_boundary(self, unhappy_only_sites):
        """Atom exactly at the cutoff distance should count as displaced."""
        scorer = WaterDisplacementScorer(unhappy_only_sites, displacement_cutoff=2.0)
        # Site 1 at (0,0,0); atom at (2,0,0) → distance = 2.0 exactly
        ligand = np.array([[2.0, 0, 0]], dtype=np.float64)
        assert scorer.score(ligand) >= 1

    def test_score_fraction_all(self, mixed_water_sites):
        scorer = WaterDisplacementScorer(mixed_water_sites, displacement_cutoff=1.0)
        # All 4 target waters displaced → 4/4 = 1.0
        ligand = np.array([[0, 0, 0], [3, 0, 0], [6, 0, 0], [9, 0, 0]], dtype=np.float64)
        assert scorer.score_fraction(ligand) == pytest.approx(1.0)

    def test_score_fraction_partial(self, mixed_water_sites):
        scorer = WaterDisplacementScorer(mixed_water_sites, displacement_cutoff=1.0)
        # 2 of 4 target waters displaced → 0.5
        ligand = np.array([[0.5, 0, 0], [6.5, 0, 0]], dtype=np.float64)
        assert scorer.score_fraction(ligand) == pytest.approx(0.5)

    def test_score_fraction_none(self, mixed_water_sites):
        scorer = WaterDisplacementScorer(mixed_water_sites, displacement_cutoff=1.0)
        ligand = np.array([[100, 100, 100]], dtype=np.float64)
        assert scorer.score_fraction(ligand) == pytest.approx(0.0)

    def test_score_fraction_no_targets(self):
        sites = [WaterSite(site_id=1, x=0, y=0, z=0, dG=-3.0)]
        scorer = WaterDisplacementScorer(sites)
        ligand = np.array([[0, 0, 0]], dtype=np.float64)
        assert scorer.score_fraction(ligand) == pytest.approx(0.0)


# ===========================================================================
# Integration with combined scoring
# ===========================================================================

@pytest.fixture()
def pocket_csv(tmp_path: Path) -> Path:
    rows = [
        "gridmap,pocket,protein,residue_type,residue_number,chain,atom_name,distance_angstrom",
        "C1=,pocket_A,test.pdb,ALA,10,A,CA,2.5",
        "C1=,pocket_A,test.pdb,GLY,20,A,CA,2.8",
    ]
    p = tmp_path / "pocket.csv"
    p.write_text("\n".join(rows))
    return p


@pytest.fixture()
def interactions_df() -> pd.DataFrame:
    return pd.DataFrame({
        "docked_ligand_index": [0, 0, 1, 1],
        "interaction_type": ["Hydrophobic"] * 4,
        "ligand_atom_indices": ["(0,)"] * 4,
        "ligand_atom_types": ["('C',)"] * 4,
        "residue_name": ["ALA", "GLY", "ALA", "GLY"],
        "residue_number": [10, 20, 10, 20],
        "residue_atom_indices": ["(1,)"] * 4,
        "residue_atom_types": ["('C',)"] * 4,
        "interaction_distance": [3.5] * 4,
        "functional_groups": ["('No_fg',)"] * 4,
    })


class TestCombinedIntegration:

    def test_water_columns_present(self, pocket_csv, interactions_df, mixed_water_sites):
        from pocket_ligand_screener.screener.residue_contact import ResidueContactScorer
        from pocket_ligand_screener.screener.combined import score_all_poses

        scorer_a = ResidueContactScorer(pocket_csv, pocket_name="pocket_A")
        water_scorer = WaterDisplacementScorer(mixed_water_sites)

        result = score_all_poses(
            interactions_df,
            {"pocket_A": scorer_a},
            water_scorer=water_scorer,
            water_weight=0.2,
        )
        assert "water_displaced_count" in result.columns
        assert "water_displaced_fraction" in result.columns

    def test_water_weight_zero_means_no_effect(self, pocket_csv, interactions_df, mixed_water_sites):
        from pocket_ligand_screener.screener.residue_contact import ResidueContactScorer
        from pocket_ligand_screener.screener.combined import score_all_poses

        scorer_a = ResidueContactScorer(pocket_csv, pocket_name="pocket_A")

        # Without water scorer (original behaviour)
        result_no_water = score_all_poses(interactions_df, {"pocket_A": scorer_a})

        # With water scorer but weight=0 (default)
        water_scorer = WaterDisplacementScorer(mixed_water_sites)
        result_with_water = score_all_poses(
            interactions_df,
            {"pocket_A": scorer_a},
            water_scorer=water_scorer,
            water_weight=0.0,
        )

        # Combined scores should match (water weight is zero)
        np.testing.assert_allclose(
            result_no_water["combined_score"].values,
            result_with_water["combined_score"].values,
        )

    def test_water_weight_affects_combined_score(self, pocket_csv, interactions_df, mixed_water_sites):
        from pocket_ligand_screener.screener.residue_contact import ResidueContactScorer
        from pocket_ligand_screener.screener.combined import score_all_poses
        from rdkit import Chem
        from rdkit.Chem import AllChem

        scorer_a = ResidueContactScorer(pocket_csv, pocket_name="pocket_A")

        # Create two fake mols: pose 0 near unhappy water site 1 (0,0,0), pose 1 far away
        mol0 = Chem.MolFromSmiles("O")
        mol0 = Chem.AddHs(mol0)
        AllChem.EmbedMolecule(mol0, randomSeed=42)
        # Move atoms near site 1 at (0,0,0)
        conf = mol0.GetConformer()
        for i in range(mol0.GetNumAtoms()):
            conf.SetAtomPosition(i, (0.5, 0.0, 0.0))

        mol1 = Chem.MolFromSmiles("O")
        mol1 = Chem.AddHs(mol1)
        AllChem.EmbedMolecule(mol1, randomSeed=42)
        conf1 = mol1.GetConformer()
        for i in range(mol1.GetNumAtoms()):
            conf1.SetAtomPosition(i, (100.0, 100.0, 100.0))

        supplier = [mol0, mol1]
        water_scorer = WaterDisplacementScorer(mixed_water_sites, displacement_cutoff=2.0)

        result = score_all_poses(
            interactions_df,
            {"pocket_A": scorer_a},
            water_scorer=water_scorer,
            sdf_supplier=supplier,
            water_weight=0.3,
        )

        # Pose 0 displaces waters, pose 1 doesn't
        row0 = result[result["docked_ligand_index"] == 0].iloc[0]
        row1 = result[result["docked_ligand_index"] == 1].iloc[0]
        assert row0["water_displaced_count"] > row1["water_displaced_count"]

    def test_backward_compatible_no_water(self, pocket_csv, interactions_df):
        """Calling without water_scorer works exactly as before."""
        from pocket_ligand_screener.screener.residue_contact import ResidueContactScorer
        from pocket_ligand_screener.screener.combined import score_all_poses

        scorer_a = ResidueContactScorer(pocket_csv, pocket_name="pocket_A")
        result = score_all_poses(interactions_df, {"pocket_A": scorer_a})
        assert all(result["water_displaced_count"] == 0)
        np.testing.assert_allclose(result["water_displaced_fraction"].values, 0.0)
