import csv
from pathlib import Path
from typing import Dict, List, Tuple, Union
from collections import namedtuple

import numpy as np
import MDAnalysis as mda
import prolif as plf
import pandas as pd

from dock_ligand_annotator.ifg import annotate_functional_groups

CONFIG_FILE = str(Path(__file__).resolve().parents[2] / "config" / "config.yaml")

Interaction = List[Union[int, str, Tuple[int], Tuple[str], float]]

Functional_group = Tuple[str]


def build_bb_sc_lookup(universe: mda.Universe) -> dict:
    """
    Pre-compute a mapping of atom index -> 'bb' or 'sc' for the entire protein.

    Call once per universe and pass the resulting dict to
    ``annotate_backbone_sidechain`` or use directly.

    Parameters:
    universe (MDAnalysis.Universe): The MDAnalysis Universe object.

    Returns:
    dict: A dictionary mapping atom index (int) to 'bb' or 'sc'.
    """
    bb_indices = set(universe.select_atoms('backbone').indices)
    return {
        atom.index: 'bb' if atom.index in bb_indices else 'sc'
        for atom in universe.atoms
    }


def annotate_backbone_sidechain(universe: mda.Universe, atom_index: int,
                                _lookup: dict = None) -> str:
    """
    Check if an atom is part of the backbone or sidechain.

    Parameters:
    universe (MDAnalysis.Universe): The MDAnalysis Universe object.
    atom_index (int): The index of the atom to check.
    _lookup (dict, optional): Pre-computed lookup from ``build_bb_sc_lookup``.
        If provided, avoids repeated ``select_atoms`` calls.

    Returns:
    str: 'bb' for backbone, 'sc' for sidechain, 'none' otherwise.
    """
    if _lookup is not None:
        return _lookup.get(atom_index, 'none')

    atom = universe.atoms[atom_index]

    is_backbone = atom in universe.select_atoms('backbone')
    is_sidechain = atom in universe.select_atoms('not backbone')

    if is_backbone:
        return 'bb'
    elif is_sidechain:
        return 'sc'
    else:
        return 'none'


def parse_prolif_interactions(fps_list: Dict[int, dict],
                              docked_ligands: plf.sdf_supplier, u: mda.Universe) -> Interaction:
    '''
    Parses ProLIF interaction fingerprints and extracts detailed interaction information.
    Args:
        fps_list (dict): A dictionary where keys are docked ligand indices and 
        values are dictionaries of residue pairs and their interactions.
        docked_ligands (iterable): An iterable containing ligand poses, where 
        each pose has a method `GetAtomWithIdx` to get atom information.
        u (MDAnalysis.Universe): An MDAnalysis Universe object containing the
        protein structure, where `u.atoms` provides access to atom information.
    Returns:
        list: A list of interactions, where each interaction is represented as
              a list containing:
            - docked_lgd_idx (int): Index of the docked ligand.
            - interaction_type (str): Type of interaction.
            - ligand_atom_idx (list): List of ligand atom indices involved in the interaction.
            - ligand_atom_types (tuple): Tuple of ligand atom types involved in the interaction.
            - residue_name (str): Name of the residue involved in the interaction.
            - residue_num (int): Residue number of the residue involved in the interaction.
            - residue_atom_idx (list): List of residue atom indices involved in the interaction.
            - residue_atom_types (tuple): Tuple of residue atom types involved in the interaction.
            - residue_atom_bb_sc (tuple): Tuple indicating whether each residue atom is in the backbone or sidechain.
            - interaction_distance (float): Distance of the interaction, rounded to three decimal places.
    '''
    interactions_list = []
    for docked_lgd_idx, fps in fps_list.items():
        for residue_pair, interactions in fps.items():
            for interaction_type, interaction_list in interactions.items():
                for interaction in interaction_list:
                    ligand_atom_idx = interaction['indices']['ligand']
                    ligand_atom_types = tuple([docked_ligands[docked_lgd_idx].GetAtomWithIdx(atm_idx).GetSymbol()
                                              for atm_idx in ligand_atom_idx])

                    residue_name = residue_pair[1].name
                    residue_num = residue_pair[1].number

                    residue_atom_idx = interaction['parent_indices']['protein']
                    residue_atom_types = tuple([u.atoms[atm_idx].type 
                                               for atm_idx in residue_atom_idx])
                    residue_atom_bb_sc = tuple([annotate_backbone_sidechain(u, atm_idx)
                                               for atm_idx in residue_atom_idx])

                    interaction_distance = np.round(interaction['distance'], 3)

                    interactions_list.append([docked_lgd_idx,
                                              interaction_type,
                                              ligand_atom_idx,
                                              ligand_atom_types,
                                              residue_name,
                                              residue_num,
                                              residue_atom_idx,
                                              residue_atom_types,
                                              residue_atom_bb_sc,
                                              interaction_distance])
    return interactions_list


def annotate_ligands(docked_poses: plf.sdf_supplier) -> Dict[int, namedtuple]:
    """
    Annotates functional groups for each docked ligand pose based the ifg RDKit module.

    Args:
        docked_poses (iterable): An iterable containing ligand poses.

    Returns:
        dict: A dictionary where keys are docked ligand indices and values are namedtuple (atom_idx, type, atom) of functional groups.
    """
    docked_ligand_fgs = {}
    for idx, pose in enumerate(docked_poses):
        docked_ligand_fgs[idx] = annotate_functional_groups(pose)
    return docked_ligand_fgs


def map_atom_indices_to_fg(docked_ligand_fgs: Dict[int, namedtuple]) -> Dict[int, Dict[int, Tuple[str]]]:
    """
    Maps atom indices to functional groups for each docked ligand.

    Args:
        docked_ligand_fgs (dict): A dictionary where keys are docked ligand indices and values are lists of functional groups.
    Returns:
        dict: A dictionary where keys are docked ligand indices and values are dictionaries mapping atom indices to functional groups.
    """
    atomidx_to_fg_mapper = {}

    for docked_lgd_idx, ifg_list in docked_ligand_fgs.items():
        atomidx_to_fg_mapper[docked_lgd_idx] = {}
        for ifg in ifg_list:
            for atom_id in ifg.atomIds:
                atomidx_to_fg_mapper[docked_lgd_idx][atom_id] = (ifg.atoms, ifg.type)
    return atomidx_to_fg_mapper


def annotate_fg(interactions_list: Interaction, docked_poses: plf.sdf_supplier) -> Interaction:
    """
    Appends functional group information to each interaction in the interactions list.

    Args:
        interactions_list (list): A list of interactions.
        docked_poses (iterable): An iterable containing ligand poses.

    Returns:
        list: The updated interactions list with functional group information appended.
    """
    docked_ligand_fgs = annotate_functional_groups(docked_poses)
    atomidx_to_fg_mapper = map_atom_indices_to_fg(docked_ligand_fgs)

    annotated_interactions = interactions_list.copy()
    parsed_fgs = []
    for interaction in annotated_interactions:
        if atomidx_to_fg_mapper.get(interaction[0]):
            for atom_idx in interaction[2]:
                try:
                    parsed_fgs.append(atomidx_to_fg_mapper[interaction[0]][atom_idx])
                except KeyError:
                    parsed_fgs.append('No_fg')
            interaction.append(tuple(parsed_fgs))
            parsed_fgs = []
    return annotated_interactions



def save_to_csv(interactions_list: Interaction, filename: str) -> None:

    header = ['docked_ligand_index',
              'interaction_type',
              'ligand_atom_indices',
              'ligand_atom_types',
              'residue_name',
              'residue_number',
              'residue_atom_indices',
              'residue_atom_types',
              'interaction_distance',
              'functional_groups']

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for interaction in interactions_list:
            writer.writerow(interaction)


def interactions_to_dataframe(interactions_list: Interaction) -> pd.DataFrame:

    columns = ['index',
               'docked_ligand_index',
               'interaction_type',
               'ligand_atom_indices',
               'ligand_atom_types',
               'residue_name',
               'residue_number',
               'residue_atom_indices',
               'residue_atom_types',
               'residue_atom_bb_sc',
               'interaction_distance']

    df = pd.DataFrame(interactions_list, columns=columns)
    return df
