from typing import Dict
import prolif as plf
import pandas as pd
import MDAnalysis as mda

from dock_ligand_annotator.types import InteractionList
from dock_ligand_annotator.interaction_utils import parse_prolif_interactions
from dock_ligand_annotator.io import save_to_csv, interactions_to_dataframe
from dock_ligand_annotator.functional_groups import annotate_fg


class InteractionLists:
    """
    A class to calculate, parse, and annotate interactions between a protein and ligands.
    Attributes:
        protein (prolif.Molecule): The protein molecule.
        ligands (prolif.Molecule): The ligands.
    Methods:
        calculate: Calculate the fingerprint interactions between the protein and ligands.
        parse: Parse the fingerprint interactions and extract detailed interaction information.
        annotate: Annotate the interactions with functional group information.
        to_csv: Save the interactions to a CSV file.
        to_dataframe: Convert the interactions to a pandas DataFrame.
    """
    def __init__(self, protein: plf.Molecule, ligands: plf.sdf_supplier) -> None:
        self.protein = protein
        self.ligands = ligands

    def calculate(self) -> Dict[int, dict]:
        """
        Calculate the fingerprint interactions between the protein and ligands.

        Returns:
            List: A list of fingerprint interactions.
        """
        fp = plf.Fingerprint(count=True)
        fp.run_from_iterable(self.ligands, self.protein)
        fps_list = fp.ifp
        return fps_list

    def parse(self, fps_list: Dict[int, dict], universe: mda.Universe) -> InteractionList:
        interactions_list = parse_prolif_interactions(fps_list, self.ligands, universe)
        return interactions_list

    def annotate(self, interactions_list: InteractionList) -> InteractionList:
        fg_annotated_interactions = annotate_fg(interactions_list, self.ligands)
        return fg_annotated_interactions

    @staticmethod
    def to_csv(interactions: InteractionList, output_file) -> None:
        save_to_csv(interactions, output_file)

    @staticmethod
    def to_dataframe(interactions: InteractionList) -> pd.DataFrame:
        df = interactions_to_dataframe(interactions)
        return df
