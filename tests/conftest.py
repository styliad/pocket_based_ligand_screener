"""Shared fixtures for pocket_ligand_screener tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "test_data"


@pytest.fixture()
def tmp_sdf(tmp_path: Path) -> Path:
    """Create a minimal multi-pose Glide-style SDF fixture.

    Two ligands ("aspirin" and "ibuprofen"), 3 poses each.
    """
    sdf_path = tmp_path / "glide_test.sdf"
    writer = Chem.SDWriter(str(sdf_path))

    smiles_map = {
        "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    }

    pose_configs = [
        ("aspirin", 1, 10, -7.5),
        ("aspirin", 1, 20, -6.8),
        ("aspirin", 1, 30, -5.2),
        ("ibuprofen", 2, 5, -8.1),
        ("ibuprofen", 2, 15, -7.0),
        ("ibuprofen", 2, 25, -6.3),
    ]

    for name, lig_idx, pose_idx, score in pose_configs:
        mol = Chem.MolFromSmiles(smiles_map[name])
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        mol.SetProp("_Name", name)
        mol.SetProp("s_m_entry_name", f"{name}.{lig_idx}")
        mol.SetProp("i_i_glide_lignum", str(lig_idx))
        mol.SetProp("i_i_glide_posenum", str(pose_idx))
        mol.SetProp("r_i_docking_score", str(score))
        mol.SetProp("r_i_glide_gscore", str(score - 0.01))
        writer.write(mol)

    writer.close()
    return sdf_path


@pytest.fixture()
def glide_real_sdf() -> Path | None:
    """Return path to the real Glide test SDF if available."""
    p = DATA_DIR / "glide_pose_data.sdf"
    return p if p.exists() else None
