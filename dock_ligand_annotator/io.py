from pathlib import Path
from typing import Dict
import csv

import MDAnalysis as mda
import prolif as plf
import pandas as pd

from dock_ligand_annotator.types import InteractionList


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


def save_to_csv(interactions_list: InteractionList, filename: str) -> None:

    header = ['docked_ligand_index',
              'interaction_type',
              'ligand_atom_indices',
              'ligand_atom_types',
              'residue_name',
              'residue_number',
              'residue_atom_indices',
              'residue_atom_types',
              'residue_atom_bb_sc',
              'interaction_distance',
              'functional_groups']

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for interaction in interactions_list:
            writer.writerow(interaction)


def interactions_to_dataframe(interactions_list: InteractionList) -> pd.DataFrame:

    base_columns = ['docked_ligand_index',
                    'interaction_type',
                    'ligand_atom_indices',
                    'ligand_atom_types',
                    'residue_name',
                    'residue_number',
                    'residue_atom_indices',
                    'residue_atom_types',
                    'residue_atom_bb_sc',
                    'interaction_distance']

    has_fg = any(len(row) > len(base_columns) for row in interactions_list)
    columns = base_columns + ['functional_groups'] if has_fg else base_columns

    df = pd.DataFrame(interactions_list, columns=columns)
    return df