"""Pre-scoring filter: reject poses that lack required ProLIF interactions.

Allows specifying required interactions (by type, residue, or both) that a
pose *must* exhibit before it proceeds to scoring.  Poses that fail the
check are excluded from the scoring DataFrame entirely.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Set

import pandas as pd

from pocket_ligand_screener.screener.residue_contact import _parse_residue_number

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RequiredInteraction:
    """A single required interaction constraint.

    Any field left as *None* is treated as a wildcard ("any").
    """

    interaction_type: Optional[str] = None
    residue_name: Optional[str] = None
    residue_number: Optional[int] = None

    def is_satisfied_by(self, pose_df: pd.DataFrame) -> bool:
        """Return True if at least one row in *pose_df* satisfies this constraint."""
        mask = pd.Series(True, index=pose_df.index)

        if self.interaction_type is not None:
            mask &= pose_df["interaction_type"] == self.interaction_type

        if self.residue_name is not None:
            mask &= pose_df["residue_name"] == self.residue_name

        if self.residue_number is not None:
            mask &= _parse_residue_number(pose_df["residue_number"]) == self.residue_number

        return bool(mask.any())


class InteractionFilter:
    """Filter docking poses by required ProLIF interactions.

    All constraints must be satisfied (logical AND) for a pose to pass.

    Parameters
    ----------
    required : list of RequiredInteraction
        Constraints that every pose must satisfy.
    """

    def __init__(self, required: List[RequiredInteraction]) -> None:
        self.required = required

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def passes(self, pose_df: pd.DataFrame) -> bool:
        """Check whether a single pose satisfies all required interactions."""
        return all(req.is_satisfied_by(pose_df) for req in self.required)

    def filter(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """Remove poses that do not satisfy all required interactions.

        Parameters
        ----------
        interactions_df : pd.DataFrame
            Full annotated interactions (multi-pose).

        Returns
        -------
        pd.DataFrame
            Subset containing only rows for poses that pass.
        """
        passing_indices: Set[int] = set()

        for pose_idx, pose_df in interactions_df.groupby("docked_ligand_index"):
            if self.passes(pose_df):
                passing_indices.add(pose_idx)

        n_total = interactions_df["docked_ligand_index"].nunique()
        n_pass = len(passing_indices)
        logger.info(
            "InteractionFilter: %d / %d poses passed (%d removed)",
            n_pass,
            n_total,
            n_total - n_pass,
        )

        return interactions_df[
            interactions_df["docked_ligand_index"].isin(passing_indices)
        ].reset_index(drop=True)
