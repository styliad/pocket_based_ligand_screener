from typing import Dict, Tuple
from collections import namedtuple

import prolif as plf

from dock_ligand_annotator.ifg import annotate_functional_groups
from dock_ligand_annotator.types import InteractionList


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


def annotate_fg(interactions_list: InteractionList, docked_poses: plf.sdf_supplier) -> InteractionList:
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

