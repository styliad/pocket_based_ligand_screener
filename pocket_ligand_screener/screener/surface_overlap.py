"""Score docking poses by spatial overlap with pocket surface vertices.

Loads pocket surface meshes from a ``pockets.npz`` file (vertices + metadata)
and scores ligand poses by the fraction of pocket vertices that fall within
a distance cutoff of any ligand atom.

NPZ structure:
    - ``metadata``: JSON string with a ``pockets`` list, each entry having
      ``name``, ``num_vertices``, ``num_triangles``.
    - ``<pocket_name>``: (N, 3) float array of vertex coordinates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
from loguru import logger
from scipy.spatial import cKDTree


class SurfaceOverlapScorer:
    """Score poses by spatial overlap with pocket surface vertices.

    Parameters
    ----------
    pocket_npz : str or Path
        Path to ``pockets.npz`` file.
    pocket_name : str, optional
        Score against a single pocket. If *None*, scores are computed
        for every pocket found in the NPZ metadata.
    distance_cutoff : float
        Distance in angstroms within which a pocket vertex is considered
        "covered" by a ligand atom. Default 2.5 Å.

    Attributes
    ----------
    pockets : dict of str → numpy.ndarray
        Mapping of pocket name to (N, 3) vertex arrays.
    trees : dict of str → cKDTree
        Pre-built KD-trees for each pocket's vertices.
    """

    def __init__(
        self,
        pocket_npz: str | Path,
        pocket_name: str | None = None,
        distance_cutoff: float = 2.5,
    ) -> None:
        self.pocket_npz = Path(pocket_npz)
        self.pocket_name = pocket_name
        self.distance_cutoff = distance_cutoff
        self.pockets: Dict[str, np.ndarray] = {}
        self.trees: Dict[str, cKDTree] = {}
        self._load(pocket_name)

    def _load(self, pocket_name: str | None) -> None:
        npz = np.load(self.pocket_npz, allow_pickle=True)
        metadata = json.loads(str(npz["metadata"]))

        for entry in metadata["pockets"]:
            name = entry["name"]
            if pocket_name is not None and name != pocket_name:
                continue
            vertices = npz[name].astype(np.float64)
            self.pockets[name] = vertices
            self.trees[name] = cKDTree(vertices)
            logger.info(
                "Loaded pocket %s: %d vertices", name, len(vertices)
            )

        if pocket_name is not None and pocket_name not in self.pockets:
            raise ValueError(
                f"Pocket {pocket_name!r} not found in {self.pocket_npz}. "
                f"Available: {[p['name'] for p in metadata['pockets']]}"
            )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(
        self,
        ligand_coords: np.ndarray,
        pocket_name: str | None = None,
    ) -> float:
        """Fraction of pocket vertices covered by the ligand.

        Parameters
        ----------
        ligand_coords : numpy.ndarray
            (A, 3) array of ligand atom coordinates.
        pocket_name : str, optional
            Pocket to score against. Required if the scorer was loaded
            with multiple pockets.

        Returns
        -------
        float
            Fraction of pocket vertices within ``distance_cutoff`` of
            any ligand atom, in [0, 1].
        """
        name = self._resolve_pocket_name(pocket_name)
        tree = self.trees[name]
        n_vertices = len(self.pockets[name])

        indices = tree.query_ball_point(ligand_coords, r=self.distance_cutoff)
        covered: set = set()
        for idx_list in indices:
            covered.update(idx_list)
        return len(covered) / n_vertices if n_vertices > 0 else 0.0

    def score_count(
        self,
        ligand_coords: np.ndarray,
        pocket_name: str | None = None,
    ) -> int:
        """Raw count of pocket vertices covered by the ligand.

        Parameters
        ----------
        ligand_coords : numpy.ndarray
            (A, 3) array of ligand atom coordinates.
        pocket_name : str, optional
            Pocket to score against.

        Returns
        -------
        int
            Number of pocket vertices within ``distance_cutoff``.
        """
        name = self._resolve_pocket_name(pocket_name)
        tree = self.trees[name]

        indices = tree.query_ball_point(ligand_coords, r=self.distance_cutoff)
        covered: set = set()
        for idx_list in indices:
            covered.update(idx_list)
        return len(covered)

    def score_all_pockets(
        self,
        ligand_coords: np.ndarray,
    ) -> Dict[str, float]:
        """Score the ligand against every loaded pocket.

        Returns
        -------
        dict
            Mapping of pocket_name → coverage fraction.
        """
        return {
            name: self.score(ligand_coords, pocket_name=name)
            for name in self.pockets
        }

    def _resolve_pocket_name(self, pocket_name: str | None) -> str:
        if pocket_name is not None:
            if pocket_name not in self.pockets:
                raise ValueError(
                    f"Pocket {pocket_name!r} not loaded. "
                    f"Available: {list(self.pockets.keys())}"
                )
            return pocket_name
        if len(self.pockets) == 1:
            return next(iter(self.pockets))
        raise ValueError(
            "Multiple pockets loaded; specify pocket_name. "
            f"Available: {list(self.pockets.keys())}"
        )


def coords_from_mol(mol: "rdkit.Chem.rdchem.Mol") -> np.ndarray:
    """Extract (N, 3) atom coordinate array from an RDKit Mol."""
    conf = mol.GetConformer()
    return np.array(
        [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())],
        dtype=np.float64,
    )
