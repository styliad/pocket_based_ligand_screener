import pytest
import MDAnalysis as mda
import prolif as plf

from dock_ligand_annotator.interaction_utils import annotate_backbone_sidechain,parse_prolif_interactions
    
    
from dock_ligand_annotator.io import save_to_csv,interactions_to_dataframe

from dock_ligand_annotator.functional_groups import(
    annotate_ligands,
    map_atom_indices_to_fg,
    annotate_fg
)


def test_annotate_backbone_sidechain():

    assert 1 == 1


def test_parse_prolif_interactions():

    assert 1 == 1


def test_annotate_ligands():

    assert 1 == 1


def test_map_atom_indices_to_fg():

    assert 1 == 1


def test_annotate_fg():

    assert 1 == 1


def test_save_to_csv():

    assert 1 == 1


def test_interactions_to_dataframe():

    assert 1 == 1
