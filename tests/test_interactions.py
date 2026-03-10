import pytest

from dock_ligand_annotator.interactions import Interactions
from dock_ligand_annotator.io import (
    load_protein_mol,
    load_docked_poses
)


@pytest.fixture(scope="module")
def setup_interactions(protein_file, ligand_file, universe):
    protein_mol = load_protein_mol(universe)
    docked_poses = load_docked_poses(ligand_file)
    interactions = Interactions(protein_mol, docked_poses)
    return interactions


def test_calculate(setup_interactions):
    interactions = setup_interactions
    fps_list = interactions.calculate()
    assert fps_list is not None
    assert isinstance(fps_list, dict)


def test_parse(setup_interactions, universe):
    interactions = setup_interactions
    fps_list = interactions.calculate()
    interactions_list = interactions.parse(fps_list, universe)
    assert interactions_list is not None
    assert isinstance(interactions_list, list)


def test_annotate(setup_interactions, universe):
    interactions = setup_interactions
    fps_list = interactions.calculate()
    interactions_list = interactions.parse(fps_list, universe)
    annotated_interactions = interactions.annotate(interactions_list)
    assert annotated_interactions is not None
    assert isinstance(annotated_interactions, list)


def test_to_csv(setup_interactions, universe, tmp_path):
    interactions = setup_interactions
    fps_list = interactions.calculate()
    interactions_list = interactions.parse(fps_list, universe)
    annotated_interactions = interactions.annotate(interactions_list)
    output_file = tmp_path / "annotated_interactions.csv"
    interactions.to_csv(annotated_interactions, output_file)
    assert output_file.exists()
    assert output_file.stat().st_size > 0  # Check that the file is not empty


def test_to_dataframe(setup_interactions, universe):
    interactions = setup_interactions
    fps_list = interactions.calculate()
    interactions_list = interactions.parse(fps_list, universe)
    annotated_interactions = interactions.annotate(interactions_list)
    df = interactions.to_dataframe(annotated_interactions)
    assert df is not None
    assert not df.empty
