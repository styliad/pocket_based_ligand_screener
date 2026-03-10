from pathlib import Path
from typing import Dict


def load_files(config: dict) -> Dict[str, str]:
    if config is None:
        return []
    protein_ligand_pairs = []
    for pair in config['protein_ligand_pairs']:
        protein_file = Path(pair['protein_file'])
        ligand_file = pair['ligand_file']
        protein_ligand_pairs.append((protein_file, ligand_file))
    return protein_ligand_pairs
