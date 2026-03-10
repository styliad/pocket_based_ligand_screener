import yaml
from pathlib import Path
import pytest
from dock_ligand_annotator.config_utils import create_config_file, load_config

TMP_CONFIG_FILE = "tmp_config_file.yaml"
TEST_CONFIG_FILE = str(Path(__file__).resolve().parents[3] / "tests" / "test_data" / "test_config_file.yaml")

@pytest.fixture
def protein_ligand_dict():
    return {
        "protein1.pdb": "ligand1.sdf",
    }


@pytest.fixture
def working_dir_path():
    return "path/to/working/folder/"


def test_create_config_file(protein_ligand_dict, tmp_path, working_dir_path):
    output_file = tmp_path / TMP_CONFIG_FILE
    create_config_file(protein_ligand_dict, output_file, working_dir_path)
    assert output_file.exists()
    assert output_file.stat().st_size > 0  # Check that the file is not empty

    with open(output_file, 'r') as file:
        config_data = yaml.safe_load(file)

    assert "protein_ligand_pairs" in config_data
    assert len(config_data["protein_ligand_pairs"]) == len(protein_ligand_dict)
    for pair in config_data["protein_ligand_pairs"]:
        assert pair["protein_file"].replace(working_dir_path, "") in protein_ligand_dict
        assert pair["ligand_file"].replace(working_dir_path, "") == protein_ligand_dict[pair["protein_file"].replace(working_dir_path, "")]

def test_load_config(protein_ligand_dict, tmp_path, working_dir_path):
    loaded_config = load_config(TEST_CONFIG_FILE)

    assert loaded_config == {'protein_ligand_pairs':
                             [{'ligand_file': 'path/to/working/folder/ligand_1.sdf',
                                'protein_file': 'path/to/working/folder/protein_1.pdb'}]}
