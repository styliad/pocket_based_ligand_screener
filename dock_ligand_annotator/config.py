import yaml
from pathlib import Path
from typing import Dict

CONFIG_FILE = str(Path(__file__).resolve().parents[2] / "config" / "config.yaml")


def create_config_file(protein_ligand_dict: Dict[str, str],
                       config_file_path: str = CONFIG_FILE,
                       working_dir_path: str = None) -> None:
    """
    Creates a configuration file in YAML format containing protein-ligand
    pairs based on a Python dictionary.
    Args:
        protein_ligand_dict (dict): A dictionary where keys are protein 
        file paths and values are ligand file paths.
        config_file_path (str, optional): The path where the configuration 
        file will be saved. Defaults to "config.yaml".
        working_dir_part (str): The path to the working directory, where
        protein and ligands data are stored.
    Returns:
        None
    """

    if working_dir_path is None:
        protein_ligand_pairs = [
            {"protein_file": protein, "ligand_file": ligand}
            for protein, ligand in protein_ligand_dict.items()
        ]
    else:
        if working_dir_path.endswith("/") is False:
            working_dir_path += "/"
        working_dir_path = working_dir_path
        protein_ligand_pairs = [
            {"protein_file": working_dir_path + protein,
             "ligand_file": working_dir_path + ligand}
            for protein, ligand in protein_ligand_dict.items()
        ]

    config_data = {"protein_ligand_pairs": protein_ligand_pairs}

    with open(config_file_path, 'w') as config_file:
        yaml.dump(config_data, config_file, default_flow_style=False)


def load_config(config_file_path: str = CONFIG_FILE) -> Dict[str, str]:
    """
    Loads the configuration file in YAML format.
    Args:
        config_file_path (str, optional): The path to the configuration file.
        Defaults to "config.yaml".
    Returns:
        dict: The loaded configuration data.
    """
    with open(config_file_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    return config
