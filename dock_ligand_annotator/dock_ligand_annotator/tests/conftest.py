from pathlib import Path
import pytest
import MDAnalysis as mda


@pytest.fixture(scope="session")
def protein_file():
    protein_file = Path(__file__).resolve().parents[3] / "data" / "dock_ligand_annotator" / "protein" / "example_protein_7JVQ.pdb"
    return protein_file


@pytest.fixture(scope="session")
def universe(protein_file):
    u = mda.Universe(protein_file)
    return u


@pytest.fixture(scope="session")
def ligand_file():
    ligand_file = str(Path(__file__).resolve().parents[3] / "data" / "dock_ligand_annotator" / "ligand" / "example_ligands.sdf")
    return ligand_file
