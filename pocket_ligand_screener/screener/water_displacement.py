"""Score docking poses by displacement of thermodynamically unfavourable waters.

Given classified water sites (from any :mod:`water_annotator` backend), a
ligand "displaces" a water when any of its heavy-atom coordinates fall within
a distance cutoff of that site.  The score is the **count** of displaced
unhappy + very-unhappy waters.
"""

from __future__ import annotations

from typing import Sequence

from loguru import logger

import numpy as np
from scipy.spatial import cKDTree

from water_annotator.base import WaterCategory, WaterSite, classify_water

# Default categories that count as "displaceable" targets.
_DEFAULT_TARGET_CATEGORIES = frozenset({WaterCategory.VERY_UNHAPPY, WaterCategory.UNHAPPY})


class WaterDisplacementScorer:
    """Score poses by the number of unfavourable waters they displace.

    Parameters
    ----------
    water_sites : sequence of WaterSite
        Hydration sites (e.g. from ``WaterMapAnnotator.sites`` or
        ``WaterFLAPAnnotator.sites``).
    displacement_cutoff : float
        A water site is considered displaced if any ligand atom is within
        this distance (Å).  Default 2.0 Å.
    target_categories : frozenset of WaterCategory, optional
        Which thermodynamic categories count towards the score.
        Default: ``{VERY_UNHAPPY, UNHAPPY}``.
    very_unhappy_threshold : float
        Threshold passed to :func:`classify_water`.
    unhappy_threshold : float
        Threshold passed to :func:`classify_water`.
    happy_threshold : float
        Threshold passed to :func:`classify_water`.
    """

    def __init__(
        self,
        water_sites: Sequence[WaterSite],
        displacement_cutoff: float = 2.0,
        target_categories: frozenset[WaterCategory] | None = None,
        very_unhappy_threshold: float = 3.5,
        unhappy_threshold: float = 2.0,
        happy_threshold: float = -1.0,
    ) -> None:
        self.displacement_cutoff = displacement_cutoff
        self.target_categories = (
            target_categories if target_categories is not None else _DEFAULT_TARGET_CATEGORIES
        )

        # Classify each site and separate targets from non-targets.
        self._all_sites = list(water_sites)
        self._target_sites: list[WaterSite] = []
        self._target_coords: np.ndarray | None = None

        for site in self._all_sites:
            cat = classify_water(
                site.dG,
                very_unhappy=very_unhappy_threshold,
                unhappy=unhappy_threshold,
                happy=happy_threshold,
            )
            if cat in self.target_categories:
                self._target_sites.append(site)

        if self._target_sites:
            self._target_coords = np.array(
                [[s.x, s.y, s.z] for s in self._target_sites], dtype=np.float64,
            )
            self._tree = cKDTree(self._target_coords)
        else:
            self._target_coords = np.empty((0, 3), dtype=np.float64)
            self._tree = None

        logger.info(
            "WaterDisplacementScorer: %d target waters out of %d total "
            "(cutoff=%.1f Å, categories=%s)",
            len(self._target_sites),
            len(self._all_sites),
            self.displacement_cutoff,
            {c.value for c in self.target_categories},
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_target_waters(self) -> int:
        """Number of water sites in the target categories."""
        return len(self._target_sites)

    @property
    def target_sites(self) -> list[WaterSite]:
        """Water sites in the target categories."""
        return list(self._target_sites)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def displaced_indices(self, ligand_coords: np.ndarray) -> set[int]:
        """Indices (into ``target_sites``) of waters displaced by the ligand."""
        if self._tree is None or len(ligand_coords) == 0:
            return set()
        ligand_tree = cKDTree(np.asarray(ligand_coords, dtype=np.float64))
        pairs = self._tree.query_ball_tree(ligand_tree, r=self.displacement_cutoff)
        return {i for i, hits in enumerate(pairs) if hits}

    def score(self, ligand_coords: np.ndarray) -> int:
        """Number of target waters displaced by the ligand."""
        return len(self.displaced_indices(ligand_coords))

    def score_fraction(self, ligand_coords: np.ndarray) -> float:
        """Fraction of target waters displaced by the ligand, in [0, 1]."""
        if self.n_target_waters == 0:
            return 0.0
        return self.score(ligand_coords) / self.n_target_waters
