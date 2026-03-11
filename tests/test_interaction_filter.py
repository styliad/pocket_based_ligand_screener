"""Tests for InteractionFilter and RequiredInteraction."""

from __future__ import annotations

import pandas as pd
import pytest

from pocket_ligand_screener.screener.interaction_filter import (
    InteractionFilter,
    RequiredInteraction,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def interactions_df() -> pd.DataFrame:
    """Two poses with different interaction profiles."""
    return pd.DataFrame({
        "docked_ligand_index": [0, 0, 0, 1, 1],
        "interaction_type": ["HBDonor", "Hydrophobic", "VdWContact", "Hydrophobic", "VdWContact"],
        "ligand_atom_indices": ["(0,)"] * 5,
        "ligand_atom_types": ["('C',)"] * 5,
        "residue_name": ["ASP", "PHE", "ALA", "PHE", "GLY"],
        "residue_number": [110, 289, 10, 289, 20],
        "residue_atom_indices": ["(1,)"] * 5,
        "residue_atom_types": ["('C',)"] * 5,
        "interaction_distance": [3.0] * 5,
        "functional_groups": ["('No_fg',)"] * 5,
    })


# ---------------------------------------------------------------------------
# RequiredInteraction
# ---------------------------------------------------------------------------

class TestRequiredInteraction:

    def test_full_match(self, interactions_df):
        req = RequiredInteraction(interaction_type="HBDonor", residue_name="ASP", residue_number=110)
        pose0 = interactions_df[interactions_df["docked_ligand_index"] == 0]
        pose1 = interactions_df[interactions_df["docked_ligand_index"] == 1]
        assert req.is_satisfied_by(pose0) is True
        assert req.is_satisfied_by(pose1) is False

    def test_type_only_wildcard(self, interactions_df):
        req = RequiredInteraction(interaction_type="Hydrophobic")
        pose0 = interactions_df[interactions_df["docked_ligand_index"] == 0]
        pose1 = interactions_df[interactions_df["docked_ligand_index"] == 1]
        assert req.is_satisfied_by(pose0) is True
        assert req.is_satisfied_by(pose1) is True

    def test_residue_only_wildcard(self, interactions_df):
        req = RequiredInteraction(residue_name="ASP", residue_number=110)
        pose0 = interactions_df[interactions_df["docked_ligand_index"] == 0]
        pose1 = interactions_df[interactions_df["docked_ligand_index"] == 1]
        assert req.is_satisfied_by(pose0) is True
        assert req.is_satisfied_by(pose1) is False

    def test_no_match(self, interactions_df):
        req = RequiredInteraction(interaction_type="PiStacking")
        pose0 = interactions_df[interactions_df["docked_ligand_index"] == 0]
        assert req.is_satisfied_by(pose0) is False

    def test_empty_dataframe(self):
        req = RequiredInteraction(interaction_type="HBDonor")
        empty = pd.DataFrame(columns=[
            "interaction_type", "residue_name", "residue_number",
        ])
        assert req.is_satisfied_by(empty) is False


# ---------------------------------------------------------------------------
# InteractionFilter
# ---------------------------------------------------------------------------

class TestInteractionFilter:

    def test_single_constraint_filters(self, interactions_df):
        filt = InteractionFilter([
            RequiredInteraction(interaction_type="HBDonor", residue_name="ASP", residue_number=110),
        ])
        result = filt.filter(interactions_df)
        # Only pose 0 has HBDonor with ASP 110
        assert set(result["docked_ligand_index"].unique()) == {0}

    def test_multiple_constraints_and_logic(self, interactions_df):
        filt = InteractionFilter([
            RequiredInteraction(interaction_type="HBDonor"),
            RequiredInteraction(interaction_type="Hydrophobic"),
        ])
        result = filt.filter(interactions_df)
        # Only pose 0 has both HBDonor and Hydrophobic
        assert set(result["docked_ligand_index"].unique()) == {0}

    def test_all_pass(self, interactions_df):
        filt = InteractionFilter([
            RequiredInteraction(interaction_type="VdWContact"),
        ])
        result = filt.filter(interactions_df)
        # Both poses have VdWContact
        assert set(result["docked_ligand_index"].unique()) == {0, 1}

    def test_none_pass(self, interactions_df):
        filt = InteractionFilter([
            RequiredInteraction(interaction_type="PiStacking"),
        ])
        result = filt.filter(interactions_df)
        assert result.empty

    def test_empty_required_passes_all(self, interactions_df):
        filt = InteractionFilter([])
        result = filt.filter(interactions_df)
        assert len(result) == len(interactions_df)

    def test_passes_method(self, interactions_df):
        filt = InteractionFilter([
            RequiredInteraction(interaction_type="HBDonor"),
        ])
        pose0 = interactions_df[interactions_df["docked_ligand_index"] == 0]
        pose1 = interactions_df[interactions_df["docked_ligand_index"] == 1]
        assert filt.passes(pose0) is True
        assert filt.passes(pose1) is False
