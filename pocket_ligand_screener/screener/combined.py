"""Combined scoring: residue contacts + surface overlap.

Merges orthogonal signals into a single ranking score per
(pose, pocket) pair and selects the best pose per ligand.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import pandas as pd

from pocket_ligand_screener.screener.residue_contact import (
    ResidueContactScorer,
    _extract_pose_residues,
    annotate_all_pockets,
)
from pocket_ligand_screener.screener.surface_overlap import (
    SurfaceOverlapScorer,
    coords_from_mol,
)

logger = logging.getLogger(__name__)


def score_all_poses(
    interactions_df: pd.DataFrame,
    residue_scorers: Dict[str, ResidueContactScorer],
    surface_scorer: Optional[SurfaceOverlapScorer] = None,
    sdf_supplier: Optional[object] = None,
    alpha: float = 1.0,
    beta: float = 0.3,
    residue_weight: float = 0.6,
    surface_weight: float = 0.4,
) -> pd.DataFrame:
    """Score every (pose, pocket) combination and return a results DataFrame.

    Parameters
    ----------
    interactions_df : pd.DataFrame
        Full annotated interactions CSV (all poses).
    residue_scorers : dict
        ``{pocket_name: ResidueContactScorer}`` from :func:`load_pocket_residue_contacts`.
    surface_scorer : SurfaceOverlapScorer, optional
        If provided, surface overlap scores are computed and combined.
    sdf_supplier : iterable of rdkit.Chem.Mol, optional
        Mol objects indexed by ``docked_ligand_index``. Required when
        ``surface_scorer`` is provided.
    alpha, beta : float
        Tversky parameters (see ``ResidueContactScorer.score_tversky``).
    residue_weight, surface_weight : float
        Weights for the combined score (normalised internally).

    Returns
    -------
    pd.DataFrame
        One row per (pose, pocket) with columns:
        ``docked_ligand_index``, ``pocket_name``,
        ``residue_count``, ``residue_coverage``, ``residue_jaccard``,
        ``residue_tversky``, ``surface_coverage``, ``combined_score``.
    """
    w_total = residue_weight + surface_weight
    w_res = residue_weight / w_total
    w_surf = surface_weight / w_total

    # Pre-index mol objects if available
    mol_map: Dict[int, object] = {}
    if surface_scorer is not None and sdf_supplier is not None:
        for i, mol in enumerate(sdf_supplier):
            if mol is not None:
                mol_map[i] = mol

    rows = []
    for pose_idx, pose_df in interactions_df.groupby("docked_ligand_index"):
        for pocket_name, scorer in residue_scorers.items():
            res_scores = scorer.score_all(pose_df, alpha=alpha, beta=beta)

            surf_cov = 0.0
            if surface_scorer is not None and pocket_name in surface_scorer.pockets:
                mol = mol_map.get(int(pose_idx))
                if mol is not None:
                    coords = coords_from_mol(mol)
                    surf_cov = surface_scorer.score(coords, pocket_name=pocket_name)

            combined = w_res * res_scores["tversky"] + w_surf * surf_cov

            rows.append({
                "docked_ligand_index": pose_idx,
                "pocket_name": pocket_name,
                "residue_count": res_scores["count"],
                "residue_coverage": res_scores["coverage"],
                "residue_jaccard": res_scores["jaccard"],
                "residue_tversky": res_scores["tversky"],
                "surface_coverage": surf_cov,
                "combined_score": combined,
            })

    return pd.DataFrame(rows)


def select_best_pose(
    scores_df: pd.DataFrame,
    rank_by: str = "combined_score",
    aggregation: str = "sum",
) -> pd.DataFrame:
    """Select the best pose based on performance across all pockets.

    For each pose, the ``rank_by`` scores from all pockets are aggregated
    (summed by default) into a single ``aggregated_score``. The pose with
    the highest aggregated score is selected.

    Parameters
    ----------
    scores_df : pd.DataFrame
        Output of :func:`score_all_poses` — one row per (pose, pocket).
    rank_by : str
        Column to aggregate across pockets.
    aggregation : str
        How to combine per-pocket scores: ``"sum"``, ``"mean"``, or ``"max"``.

    Returns
    -------
    pd.DataFrame
        All rows from ``scores_df`` for the winning pose (one row per
        pocket), with an added ``aggregated_score`` column.
    """
    agg_funcs = {"sum": "sum", "mean": "mean", "max": "max"}
    if aggregation not in agg_funcs:
        raise ValueError(
            f"Unknown aggregation {aggregation!r}. Use 'sum', 'mean', or 'max'."
        )

    pose_agg = (
        scores_df
        .groupby("docked_ligand_index")[rank_by]
        .agg(agg_funcs[aggregation])
        .rename("aggregated_score")
    )

    best_pose_idx = pose_agg.idxmax()
    result = scores_df[scores_df["docked_ligand_index"] == best_pose_idx].copy()
    result["aggregated_score"] = pose_agg.loc[best_pose_idx]
    return result.reset_index(drop=True)
