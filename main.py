"""
main.py — End-to-end pipeline overview for pocket_ligand_screener.

This script shows the full workflow with the actual implementations that exist
and dummy/placeholder functions for modules that are not yet built.

Pipeline
--------
1. STANDARDISE  — convert docking software output SDF → standardised SDF
2. ANNOTATE     — calculate interactions + functional groups
3. SCREEN       — score poses against pocket data, select best pose  [TODO]

Run modes for screening (step 3):
    - "residue_contact": maximise common residue contacts between
      ligand pose and pocket (uses .csv input file)
    - "surface_overlap": maximise spatial overlap between ligand atom
      positions and pocket surface vertices (uses .npz input file)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from pocket_ligand_screener.standardiser import GlideStandardiser
from pocket_ligand_screener.screener import (
    ResidueContactScorer,
    SurfaceOverlapScorer,
    annotate_all_pockets,
    coords_from_mol,
    score_all_poses,
    select_best_pose_per_pocket,
)

# ---------------------------------------------------------------------------
# Paths (example — would come from config in production)
# ---------------------------------------------------------------------------
RAW_DOCKING_SDF = Path("data/raw_docking_poses.sdf")
STANDARDISED_SDF = Path("result/standardised_poses.sdf")
PROTEIN_PDB = Path("data/protein.pdb")
ANNOTATED_CSV = Path("result/annotated_interactions.csv")
POCKET_CONTACTS_CSV = Path("data/output4.csv")  # residue contact mode
POCKET_VERTICES_NPZ = Path("data/pockets.npz")  # surface overlap mode
OUTPUT_CSV = Path("result/selected_poses.csv")
OUTPUT_SDF = Path("result/selected_poses.sdf")


# ===================================================================
# STEP 1: STANDARDISE — convert raw docking output → standardised SDF
# ===================================================================
# STATUS: DONE (pocket_ligand_screener.standardiser)

def step1_standardise(raw_sdf: Path, output_sdf: Path) -> Path:
    """Standardise raw docking SDF to unified format.

    Adds: molecule_name, ligand_idx, pose_idx, docking_score, docking_algorithm
    Streaming — constant memory for >10k poses.
    """
    standardiser = GlideStandardiser(raw_sdf)
    standardiser.standardise(output_sdf)
    print(f"  Standardised {len(standardiser)} poses → {output_sdf}")
    return output_sdf


# ===================================================================
# STEP 2: ANNOTATE — calculate interactions + functional groups
# ===================================================================
# STATUS: DONE (dock_ligand_annotator.interactions.Interactions)
# Not directly called here — runs separately and produces annotated_interactions.csv
#
# Uses: ProLIF for interaction fingerprints, IFG for functional groups
# Input: protein PDB + standardised SDF
# Output: annotated_interactions.csv with columns:
#   docked_ligand_index, interaction_type,
#   ligand_atom_indices, ligand_atom_types,
#   residue_name, residue_number,
#   residue_atom_indices, residue_atom_types, residue_atom_bb_sc,
#   interaction_distance, functional_groups

def step2_annotate(protein_pdb: Path, standardised_sdf: Path, output_csv: Path) -> Path:
    """Calculate interactions and annotate functional groups.

    This wraps dock_ligand_annotator's Interactions class.
    """
    # --- EXISTING CODE (dock_ligand_annotator) ---
    # from dock_ligand_annotator.mda_utils import load_universe, load_protein_mol, load_docked_poses
    # from dock_ligand_annotator.classes.interactions import Interactions
    #
    # u = load_universe(str(protein_pdb))
    # protein_mol = load_protein_mol(u)
    # docked_poses = load_docked_poses(str(standardised_sdf))
    #
    # interactions = Interactions(protein_mol, docked_poses)
    # fp_list = interactions.calculate()
    # parsed = interactions.parse(fp_list, u)
    # annotated = interactions.annotate(parsed)
    # interactions.to_csv(annotated, str(output_csv))

    print(f"  Annotated interactions → {output_csv}")
    return output_csv


# ===================================================================
# STEP 3: SCREEN — score poses against pocket, select best per ligand
# ===================================================================
# STATUS: TODO — all functions below are placeholders

# ---- 3a. Load pocket data ----

def load_pocket_residue_contacts(pocket_csv: Path) -> dict:
    """Load pocket residue contacts from pre-calculate residue contact csv file.

    Returns a dict keyed by pocket name, each value being a
    ``ResidueContactScorer`` ready to score poses against that pocket.

    CSV columns: surface, pocket, protein, residue_type, residue_number,
                 chain, atom_name, distance_angstrom
    """
    df = pd.read_csv(pocket_csv)
    pocket_names = df["pocket"].unique()

    scorers = {}
    for pocket_name in pocket_names:
        scorers[pocket_name] = ResidueContactScorer(pocket_csv, pocket_name=pocket_name)

    print(f"  Loaded {len(scorers)} pocket(s): {list(scorers.keys())}")
    return scorers


def load_pocket_surface_vertices(
    pocket_npz: Path,
    distance_cutoff: float = 2.5,
) -> SurfaceOverlapScorer:
    """Load pocket surface vertices from pockets.npz.

    Returns a ``SurfaceOverlapScorer`` with pre-built KD-trees for
    every pocket found in the NPZ metadata.
    """
    scorer = SurfaceOverlapScorer(pocket_npz, distance_cutoff=distance_cutoff)
    print(f"  Loaded {len(scorer.pockets)} pocket surface(s): {list(scorer.pockets.keys())}")
    return scorer


# ---- 3b. Score each pose against pocket ----

def score_pose_residue_contacts(
    pose_interactions: pd.DataFrame,
    scorer: ResidueContactScorer,
) -> float:
    """Score a single pose by counting shared residue contacts.

    Compares (residue_name, residue_number) from annotated_interactions.csv
    against pocket residues from output4.csv.

    Returns the number of common residue contacts.
    """
    return scorer.score(pose_interactions)


def score_pose_surface_overlap(
    pose_mol: "Chem.Mol",
    surface_scorer: SurfaceOverlapScorer,
    pocket_name: str,
) -> float:
    """Score a single pose by spatial overlap with pocket surface.

    Returns the fraction of pocket vertices covered by the ligand.
    """
    coords = coords_from_mol(pose_mol)
    return surface_scorer.score(coords, pocket_name=pocket_name)


# ---- 3c. Select best pose per ligand ----

def select_best_poses(
    standardised_sdf: Path,
    annotated_csv: Path,
    residue_scorers: dict,
    surface_scorer: SurfaceOverlapScorer | None = None,
    alpha: float = 1.0,
    beta: float = 0.3,
    residue_weight: float = 0.6,
    surface_weight: float = 0.4,
    rank_by: str = "combined_score",
) -> "pd.DataFrame":
    """Score all poses and select the best per pocket.

    Parameters
    ----------
    standardised_sdf : Path
        Standardised SDF with molecule_name, ligand_idx, pose_idx.
    annotated_csv : Path
        Annotated interactions CSV from step 2.
    residue_scorers : dict
        ``{pocket_name: ResidueContactScorer}``.
    surface_scorer : SurfaceOverlapScorer, optional
        If provided, surface overlap scores are computed and combined.
    alpha, beta : float
        Tversky index parameters.
    residue_weight, surface_weight : float
        Weights for the combined score.
    rank_by : str
        Column to maximise when selecting the best pose.

    Returns
    -------
    pd.DataFrame
        Scores DataFrame with the best pose per pocket.
    """
    interactions_df = pd.read_csv(annotated_csv)

    # Load mol objects only if surface scoring is requested
    sdf_supplier = None
    if surface_scorer is not None:
        sdf_supplier = Chem.SDMolSupplier(str(standardised_sdf), removeHs=False)

    scores_df = score_all_poses(
        interactions_df=interactions_df,
        residue_scorers=residue_scorers,
        surface_scorer=surface_scorer,
        sdf_supplier=sdf_supplier,
        alpha=alpha,
        beta=beta,
        residue_weight=residue_weight,
        surface_weight=surface_weight,
    )

    best_df = select_best_pose_per_pocket(scores_df, rank_by=rank_by)

    print(f"  Scored {len(scores_df)} (pose, pocket) pairs")
    print(f"  Selected {len(best_df)} best pose(s)")
    return scores_df, best_df


# ---- 3d. Export results ----

def export_results(
    scores_df: "pd.DataFrame",
    best_df: "pd.DataFrame",
    interactions_df: "pd.DataFrame",
    residue_scorers: dict,
    standardised_sdf: Path,
    output_csv: Path,
    output_sdf: Path,
) -> None:
    """Write the final CSV and filtered SDF for selected poses.

    CSV contains: full scores table + annotated interactions for winning
    poses with pocket_name column.

    SDF contains only the Mol objects for the selected poses.
    """
    # Save full scores
    scores_df.to_csv(output_csv, index=False)
    print(f"  Scores CSV → {output_csv}")

    # Annotate interactions with pocket names and save
    annotated = annotate_all_pockets(interactions_df, residue_scorers)
    annotated_csv = output_csv.parent / "annotated_interactions_with_pockets.csv"
    annotated.to_csv(annotated_csv, index=False)
    print(f"  Annotated interactions → {annotated_csv}")

    # Extract winning pose mol objects to a new SDF
    winning_indices = set(best_df["docked_ligand_index"].astype(int).tolist())
    supplier = Chem.SDMolSupplier(str(standardised_sdf), removeHs=False)
    writer = Chem.SDWriter(str(output_sdf))
    for i, mol in enumerate(supplier):
        if mol is not None and i in winning_indices:
            writer.write(mol)
    writer.close()
    print(f"  Selected poses SDF → {output_sdf}")


# ===================================================================
# PIPELINE
# ===================================================================

def run_pipeline(
    raw_sdf: Path,
    protein_pdb: Path,
    pocket_contacts_csv: Path = None,
    pocket_vertices_npz: Path = None,
    mode: str = "residue_contact",
    distance_cutoff: float = 2.5,
    output_dir: Path = Path("result"),
) -> None:
    """Run the full standardise → annotate → screen pipeline.

    Parameters
    ----------
    raw_sdf : Path
        Raw docking output SDF (e.g. from Glide).
    protein_pdb : Path
        Protein structure file.
    pocket_contacts_csv : Path, optional
        Pocket residue contacts CSV (output4.csv). Required for residue_contact mode.
    pocket_vertices_npz : Path, optional
        Pocket surface vertices NPZ (pockets.npz). Required for surface_overlap mode.
    mode : str
        "residue_contact" or "surface_overlap".
    distance_cutoff : float
        Cutoff in angstroms for surface overlap mode.
    output_dir : Path
        Directory for all output files.
    """
    output_dir = Path(output_dir)
    standardised_sdf = output_dir / "standardised_poses.sdf"
    annotated_csv = output_dir / "annotated_interactions.csv"
    output_csv = output_dir / "selected_poses.csv"
    output_sdf = output_dir / "selected_poses.sdf"

    # Step 1: Standardise
    print("Step 1: Standardising docking poses...")
    step1_standardise(raw_sdf, standardised_sdf)

    # Step 2: Annotate interactions + functional groups
    print("Step 2: Annotating interactions and functional groups...")
    step2_annotate(protein_pdb, standardised_sdf, annotated_csv)

    # Step 3: Screen and select best pose per ligand
    print(f"Step 3: Screening poses (mode={mode})...")

    # Always load residue scorers (required for all modes)
    if pocket_contacts_csv is None:
        raise ValueError("pocket_contacts_csv is required")
    residue_scorers = load_pocket_residue_contacts(pocket_contacts_csv)

    # Optionally load surface scorer
    surface_scorer = None
    if mode in ("surface_overlap", "combined"):
        if pocket_vertices_npz is None:
            raise ValueError("pocket_vertices_npz is required for surface_overlap/combined mode")
        surface_scorer = load_pocket_surface_vertices(pocket_vertices_npz, distance_cutoff)
    elif mode != "residue_contact":
        raise ValueError(
            f"Unknown mode: {mode!r}. Use 'residue_contact', 'surface_overlap', or 'combined'."
        )

    scores_df, best_df = select_best_poses(
        standardised_sdf,
        annotated_csv,
        residue_scorers=residue_scorers,
        surface_scorer=surface_scorer,
    )

    # Step 4: Export
    print("Step 4: Exporting results...")
    interactions_df = pd.read_csv(annotated_csv)
    export_results(
        scores_df, best_df, interactions_df, residue_scorers,
        standardised_sdf, output_csv, output_sdf,
    )

    print(f"Done. Output: {output_csv}, {output_sdf}")


# ===================================================================
# Entry point
# ===================================================================

if __name__ == "__main__":
    run_pipeline(
        raw_sdf=RAW_DOCKING_SDF,
        protein_pdb=PROTEIN_PDB,
        pocket_contacts_csv=POCKET_CONTACTS_CSV,
        pocket_vertices_npz=POCKET_VERTICES_NPZ,
        mode="residue_contact",
        output_dir=Path("result"),
    )
