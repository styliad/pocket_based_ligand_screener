import MDAnalysis as mda
import prolif as plf
from dock_ligand_annotator.mda_utils import (
    load_universe,
    load_protein_mol,
    load_docked_poses
)


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
