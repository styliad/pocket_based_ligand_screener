from pathlib import Path
from typing import Dict

import MDAnalysis as mda
import prolif as plf


def load_files(config: dict) -> Dict[str, str]:
    if config is None:
        return []
    protein_ligand_pairs = []
    for pair in config['protein_ligand_pairs']:
        protein_file = Path(pair['protein_file'])
        ligand_file = pair['ligand_file']
        protein_ligand_pairs.append((protein_file, ligand_file))
    return protein_ligand_pairs


def load_universe(protein_file: str) -> mda.Universe:
    u = mda.Universe(protein_file)
    u.atoms.guess_bonds()  # Important for correct functionality
    return u


def load_protein_mol(universe: mda.Universe) -> plf.Molecule:
    protein_mol = plf.Molecule.from_mda(universe)
    return protein_mol


def load_docked_poses(ligand_file: str) -> plf.sdf_supplier:
    docked_poses = plf.sdf_supplier(ligand_file)
    return docked_poses
