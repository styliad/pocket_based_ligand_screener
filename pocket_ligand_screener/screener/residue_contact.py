"""Score docking poses by residue contact overlap with a pocket.

Compares residue contacts from annotated interaction data (produced by
dock_ligand_annotator) against pocket residue atoms from a pocket
definition CSV.

Pocket CSV columns:
    surface, pocket, protein, residue_type, residue_number,
    chain, atom_name, distance_angstrom

Annotated interactions CSV columns:
    docked_ligand_index, interaction_type, ligand_atom_indices,
    ligand_atom_types, residue_name, residue_number,
    residue_atom_indices, residue_atom_types, residue_atom_bb_sc,
    interaction_distance, functional_groups
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

ResidueKey = Tuple[str, int]  # (residue_type, residue_number)


def _parse_residue_number(series: pd.Series) -> pd.Series:
    """Convert residue_number values that may be stored as tuple-strings."""
    def _to_int(val):
        if isinstance(val, (int, float)):
            return int(val)
        s = str(val).strip()
        if s.startswith("("):
            return int(ast.literal_eval(s)[0])
        return int(s)
    return series.map(_to_int)


def _extract_pose_residues(pose_interactions: pd.DataFrame) -> Set[ResidueKey]:
    """Extract unique (residue_name, residue_number) from interaction rows."""
    if pose_interactions.empty:
        return set()
    return set(
        zip(
            pose_interactions["residue_name"],
            _parse_residue_number(pose_interactions["residue_number"]),
        )
    )


class ResidueContactScorer:
    """Score poses by residue contact overlap with a pocket.

    Supports multiple scoring metrics: raw count, coverage fraction,
    Jaccard similarity, and Tversky index.

    Parameters
    ----------
    pocket_csv : str or Path
        Path to pocket contacts CSV (csv format)
    pocket_name : str, optional
        Name of the pocket to use. If *None*, all rows in the CSV are used.
        Matched against the ``pocket`` column.

    Attributes
    ----------
    pocket_residues : set of (str, int)
        Unique (residue_type, residue_number) pairs defining the pocket.
    """

    def __init__(self, pocket_csv: str | Path, pocket_name: str | None = None) -> None:
        self.pocket_csv = Path(pocket_csv)
        self.pocket_name = pocket_name
        self.pocket_residues: Set[ResidueKey] = self._load_pocket_residues()

    def _load_pocket_residues(self) -> Set[ResidueKey]:
        """Load unique (residue_type, residue_number) pairs from the pocket CSV."""
        df = pd.read_csv(self.pocket_csv)

        if self.pocket_name is not None:
            df = df[df["pocket"] == self.pocket_name]
            if df.empty:
                raise ValueError(
                    f"No rows found for pocket {self.pocket_name!r} "
                    f"in {self.pocket_csv}"
                )

        residues = set(
            zip(df["residue_type"], _parse_residue_number(df["residue_number"]))
        )

        logger.info(
            "Loaded %d unique residues for pocket %s from %s",
            len(residues),
            self.pocket_name or "(all)",
            self.pocket_csv.name,
        )
        return residues

    # ------------------------------------------------------------------
    # Scoring methods
    # ------------------------------------------------------------------

    def score(self, pose_interactions: pd.DataFrame) -> float:
        """Score a pose by counting shared residue contacts (raw count).

        Parameters
        ----------
        pose_interactions : pd.DataFrame
            Rows from the pose interactions CSV for a single pose.

        Returns
        -------
        float
            Number of unique (residue_name, residue_number) contacts
            that overlap with the pocket definition.
        """
        pose_residues = _extract_pose_residues(pose_interactions)
        return float(len(pose_residues & self.pocket_residues))

    def score_coverage(self, pose_interactions: pd.DataFrame) -> float:
        """Fraction of pocket residues contacted by the pose.

        Returns
        -------
        float
            ``|P ∩ L| / |P|``, in [0, 1]. Returns 0 if the pocket is empty.
        """
        if not self.pocket_residues:
            return 0.0
        pose_residues = _extract_pose_residues(pose_interactions)
        return len(pose_residues & self.pocket_residues) / len(self.pocket_residues)

    def score_jaccard(self, pose_interactions: pd.DataFrame) -> float:
        """Jaccard similarity between pose and pocket residue sets.

        Returns
        -------
        float
            ``|P ∩ L| / |P ∪ L|``, in [0, 1].
        """
        pose_residues = _extract_pose_residues(pose_interactions)
        intersection = len(pose_residues & self.pocket_residues)
        union = len(pose_residues | self.pocket_residues)
        return intersection / union if union > 0 else 0.0

    def score_tversky(
        self,
        pose_interactions: pd.DataFrame,
        alpha: float = 1.0,
        beta: float = 0.3,
    ) -> float:
        """Tversky index between pose and pocket residue sets.

        .. math::

            T = \\frac{|P \\cap L|}
                      {|P \\cap L| + \\alpha |P \\setminus L| + \\beta |L \\setminus P|}

        Parameters
        ----------
        alpha : float
            Weight for pocket residues missed by the pose
            (higher → penalise poor pocket coverage).
        beta : float
            Weight for pose contacts outside the pocket
            (higher → penalise poses that extend beyond the pocket).

        Returns
        -------
        float
            Tversky index in [0, 1].
        """
        pose_residues = _extract_pose_residues(pose_interactions)
        intersection = len(pose_residues & self.pocket_residues)
        pocket_only = len(self.pocket_residues - pose_residues)
        pose_only = len(pose_residues - self.pocket_residues)
        denom = intersection + alpha * pocket_only + beta * pose_only
        return intersection / denom if denom > 0 else 0.0

    def score_all(
        self,
        pose_interactions: pd.DataFrame,
        alpha: float = 1.0,
        beta: float = 0.3,
    ) -> Dict[str, float]:
        """Compute all scoring metrics at once for a single pose.

        Returns
        -------
        dict
            Keys: ``count``, ``coverage``, ``jaccard``, ``tversky``.
        """
        pose_residues = _extract_pose_residues(pose_interactions)
        intersection = len(pose_residues & self.pocket_residues)
        union = len(pose_residues | self.pocket_residues)
        pocket_only = len(self.pocket_residues - pose_residues)
        pose_only = len(pose_residues - self.pocket_residues)

        n_pocket = len(self.pocket_residues)
        tversky_denom = intersection + alpha * pocket_only + beta * pose_only

        return {
            "count": float(intersection),
            "coverage": intersection / n_pocket if n_pocket > 0 else 0.0,
            "jaccard": intersection / union if union > 0 else 0.0,
            "tversky": intersection / tversky_denom if tversky_denom > 0 else 0.0,
        }

    # ------------------------------------------------------------------
    # Pocket annotation
    # ------------------------------------------------------------------

    def annotate_interactions(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """Add a ``pocket_name`` column to an interactions DataFrame.

        Each row gets this scorer's ``pocket_name`` if its
        ``(residue_name, residue_number)`` is in the pocket residue set,
        otherwise ``None``.

        Parameters
        ----------
        interactions_df : pd.DataFrame
            Full annotated interactions (all poses).

        Returns
        -------
        pd.DataFrame
            Copy with an added ``pocket_name`` column.
        """
        df = interactions_df.copy()
        residue_numbers = _parse_residue_number(df["residue_number"])
        keys = list(zip(df["residue_name"], residue_numbers))
        df["pocket_name"] = [
            self.pocket_name if k in self.pocket_residues else None
            for k in keys
        ]
        return df


def annotate_all_pockets(
    interactions_df: pd.DataFrame,
    scorers: Dict[str, ResidueContactScorer],
) -> pd.DataFrame:
    """Annotate interactions with pocket membership across multiple pockets.

    Each interaction row can belong to zero or more pockets. The result
    contains a ``pocket_name`` column. Rows matching multiple pockets are
    duplicated (one copy per pocket). Rows matching no pocket get
    ``pocket_name = None`` and appear once.

    Parameters
    ----------
    interactions_df : pd.DataFrame
        Full annotated interactions CSV.
    scorers : dict
        Mapping of pocket_name → ``ResidueContactScorer``.

    Returns
    -------
    pd.DataFrame
        Annotated DataFrame with ``pocket_name`` column.
    """
    residue_numbers = _parse_residue_number(interactions_df["residue_number"])
    keys = list(zip(interactions_df["residue_name"], residue_numbers))

    # Build a mapping: row index → list of pocket names
    row_pockets: Dict[int, List[str]] = {i: [] for i in range(len(interactions_df))}
    for pocket_name, scorer in scorers.items():
        for i, k in enumerate(keys):
            if k in scorer.pocket_residues:
                row_pockets[i].append(pocket_name)

    # Expand rows
    new_rows = []
    for i, row in enumerate(interactions_df.itertuples(index=False)):
        pockets = row_pockets[i] or [None]
        for pname in pockets:
            new_row = row._asdict()
            new_row["pocket_name"] = pname
            new_rows.append(new_row)

    return pd.DataFrame(new_rows)
