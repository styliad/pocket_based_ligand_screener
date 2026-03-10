"""Base class for docking pose standardisers.

Each docking software subclass maps its native SDF properties
to the standardised column set:
    molecule_name, ligand_idx, pose_idx, docking_score, docking_algorithm

The standardisation pipeline is **streaming**: input molecules are read one
at a time via ``ForwardSDMolSupplier`` and written immediately to the output
SDF.  Only lightweight metadata records are kept in memory, so this scales
to arbitrarily large files with constant memory overhead.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem

logger = logging.getLogger(__name__)

# Standardised property names written into output SDF files.
STANDARD_PROPS = (
    "molecule_name",
    "ligand_idx",
    "pose_idx",
    "docking_score",
    "docking_algorithm",
)


@dataclass(frozen=True)
class StandardisedPoseRecord:
    """Lightweight metadata for a single standardised pose (no Mol object)."""

    molecule_name: str
    ligand_idx: int
    pose_idx: int
    docking_score: float
    docking_algorithm: str


class BaseStandardiser(ABC):
    """Abstract base for docking software-specific SDF standardisers.

    Subclasses must implement:
        - ``_extract_molecule_name``
        - ``_extract_pose_idx``
        - ``_extract_docking_score``
        - ``docking_algorithm``  (class-level string attribute)

    The base class handles:
        - Streaming SDF read → write (constant memory, safe for large pose files)
        - ``ligand_idx`` assignment (sequential per unique molecule name)
        - Writing the standardised SDF with the five canonical properties
    """

    docking_algorithm: str = ""

    def __init__(self, sdf_path: str | Path) -> None:
        self.sdf_path = Path(sdf_path)
        if not self.sdf_path.exists():
            raise FileNotFoundError(f"SDF file not found: {self.sdf_path}")
        self._records: list[StandardisedPoseRecord] = []

    # ------------------------------------------------------------------
    # Abstract hooks for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _extract_molecule_name(self, mol: Chem.Mol) -> str:
        """Return a human-readable name for the molecule."""

    @abstractmethod
    def _extract_pose_idx(self, mol: Chem.Mol) -> int:
        """Return the pose index from the native SDF properties."""

    @abstractmethod
    def _extract_docking_score(self, mol: Chem.Mol) -> float:
        """Return the docking score from the native SDF properties."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _stream_mols(self):
        """Yield *(record_idx, mol)* by streaming with ForwardSDMolSupplier."""
        with open(self.sdf_path, "rb") as fh:
            supplier = Chem.ForwardSDMolSupplier(fh, removeHs=False)
            for record_idx, mol in enumerate(supplier):
                if mol is None:
                    logger.warning(
                        "Skipping unreadable record at index %d", record_idx
                    )
                    continue
                yield record_idx, mol

    def _stamp_mol(
        self, mol: Chem.Mol, record: StandardisedPoseRecord
    ) -> Chem.RWMol:
        """Set the five standardised properties on a writable copy of *mol*."""
        rw = Chem.RWMol(mol)
        rw.SetProp("molecule_name", record.molecule_name)
        rw.SetIntProp("ligand_idx", record.ligand_idx)
        rw.SetIntProp("pose_idx", record.pose_idx)
        rw.SetDoubleProp("docking_score", record.docking_score)
        rw.SetProp("docking_algorithm", record.docking_algorithm)
        return rw

    def _make_record(
        self,
        mol: Chem.Mol,
        name_to_ligidx: dict[str, int],
    ) -> StandardisedPoseRecord:
        """Build a record from *mol*, updating *name_to_ligidx* in place."""
        molecule_name = self._extract_molecule_name(mol)

        if molecule_name not in name_to_ligidx:
            name_to_ligidx[molecule_name] = len(name_to_ligidx)
        ligand_idx = name_to_ligidx[molecule_name]

        return StandardisedPoseRecord(
            molecule_name=molecule_name,
            ligand_idx=ligand_idx,
            pose_idx=self._extract_pose_idx(mol),
            docking_score=self._extract_docking_score(mol),
            docking_algorithm=self.docking_algorithm,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def standardise(self, output_path: str | Path) -> Path:
        """Stream-read the input SDF, standardise, and stream-write the output.

        Only lightweight :class:`StandardisedPoseRecord` objects (no ``Mol``)
        are kept in memory, so this scales to arbitrarily large files.

        Parameters
        ----------
        output_path:
            Path for the standardised SDF.

        Returns
        -------
        Path
            Resolved *output_path*.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        name_to_ligidx: dict[str, int] = {}
        records: list[StandardisedPoseRecord] = []

        writer = Chem.SDWriter(str(output_path))
        try:
            for _idx, mol in self._stream_mols():
                record = self._make_record(mol, name_to_ligidx)
                records.append(record)
                writer.write(self._stamp_mol(mol, record))
        finally:
            writer.close()

        self._records = records

        logger.info(
            "Standardised %d poses (%d unique ligands) from %s -> %s",
            len(records),
            len(name_to_ligidx),
            self.sdf_path.name,
            output_path.name,
        )
        return output_path

    @property
    def records(self) -> list[StandardisedPoseRecord]:
        """Pose metadata collected during the last ``standardise()`` call."""
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"sdf_path='{self.sdf_path}', "
            f"n_poses={len(self._records)})"
        )
