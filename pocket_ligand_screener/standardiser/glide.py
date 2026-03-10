"""Standardiser for Schrödinger Glide docking output SDF files.

Glide property mapping
----------------------
molecule_name    ← ``_Name`` (SDF header line 1)
ligand_idx       ← assigned by base class (sequential per unique name)
pose_idx         ← ``i_i_glide_posenum``
docking_score    ← ``r_i_docking_score``
docking_algorithm = ``"glide"``
"""

from __future__ import annotations

from rdkit import Chem

from pocket_ligand_screener.standardiser.base import BaseStandardiser


class GlideStandardiser(BaseStandardiser):
    """Standardise Schrödinger Glide SDF output."""

    docking_algorithm: str = "glide"

    # Native Glide SDF property names
    _PROP_POSE_IDX = "i_i_glide_posenum"
    _PROP_DOCKING_SCORE = "r_i_docking_score"

    def _extract_molecule_name(self, mol: Chem.Mol) -> str:
        """Use the SDF molecule title (``_Name``)."""
        if mol.HasProp("_Name") and mol.GetProp("_Name").strip():
            return mol.GetProp("_Name").strip()
        raise ValueError(
            "Glide SDF record is missing the molecule name (_Name / header line)."
        )

    def _extract_pose_idx(self, mol: Chem.Mol) -> int:
        if mol.HasProp(self._PROP_POSE_IDX):
            return int(mol.GetProp(self._PROP_POSE_IDX))
        raise ValueError(
            f"Glide SDF record is missing the required property '{self._PROP_POSE_IDX}'."
        )

    def _extract_docking_score(self, mol: Chem.Mol) -> float:
        if mol.HasProp(self._PROP_DOCKING_SCORE):
            return float(mol.GetProp(self._PROP_DOCKING_SCORE))
        raise ValueError(
            f"Glide SDF record is missing the required property '{self._PROP_DOCKING_SCORE}'."
        )
