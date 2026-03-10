import pytest
from pathlib import Path

import MDAnalysis as mda
import prolif as plf

from dock_ligand_annotator.io import (
    load_files,
    load_universe,
    load_protein_mol,
    load_docked_poses
)


@pytest.fixture
def config():
    return {
        "protein_ligand_pairs": [
            {"protein_file": "protein1.pdb", "ligand_file": "ligand1.sdf"},
            {"protein_file": "protein2.pdb", "ligand_file": "ligand2.sdf"}
        ]
    }


def test_load_files(config):
    result = load_files(config)
    assert len(result) == 2
    assert result[0] == (Path("protein1.pdb"), "ligand1.sdf")
    assert result[1] == (Path("protein2.pdb"), "ligand2.sdf")


def test_load_files_empty_config():
    config = {"protein_ligand_pairs": []}
    result = load_files(config)
    assert result == []


def test_load_files_missing_protein_file():
    config = {
        "protein_ligand_pairs": [
            {"ligand_file": "ligand1.sdf"}
        ]
    }
    with pytest.raises(KeyError):
        load_files(config)


def test_load_files_missing_ligand_file():
    config = {
        "protein_ligand_pairs": [
            {"protein_file": "protein1.pdb"}
        ]
    }
    with pytest.raises(KeyError):
        load_files(config)


def test_load_files_invalid_path(config):
    config["protein_ligand_pairs"][0]["protein_file"] = "/invalid_path/protein1.pdb"
    result = load_files(config)
    assert result[0] == (Path("/invalid_path/protein1.pdb"), "ligand1.sdf")


def test_load_universe(protein_file, universe):
    universe = load_universe(protein_file)
    assert isinstance(universe, mda.Universe)
    # assert len(u.atoms) == 4631


def test_load_protein_mol(universe):
    protein_mol = load_protein_mol(universe)
    assert isinstance(protein_mol, plf.Molecule)
    # assert len(protein_mol) == 4631


def test_load_docked_poses(ligand_file):
    docked_poses = load_docked_poses(ligand_file)
    assert isinstance(docked_poses, plf.sdf_supplier)
    assert len(list(docked_poses)) == 8
