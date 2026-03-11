"""Base class for water site annotators.

Each software subclass parses its native output to extract hydration-site
coordinates and free-energy values, then the base class writes a classified
PDB where each water is labelled by thermodynamic category:

    WVU  – very unhappy   (dG > 3.5 kcal/mol)
    WUN  – unhappy         (2.0 <= dG <= 3.5 kcal/mol)
    WBL  – bulk-like       (-1.0 < dG < 2.0 kcal/mol)
    WHP  – happy           (dG <= -1.0 kcal/mol)

The dG value is stored in the B-factor column.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# Default classification thresholds (kcal/mol).
_VERY_UNHAPPY_THRESHOLD = 3.5
_UNHAPPY_THRESHOLD = 2.0
_HAPPY_THRESHOLD = -1.0


class WaterCategory(Enum):
    """Thermodynamic classification for a hydration site."""

    VERY_UNHAPPY = "WVU"
    UNHAPPY = "WUN"
    BULK_LIKE = "WBL"
    HAPPY = "WHP"


@dataclass(frozen=True)
class WaterSite:
    """A single hydration site with coordinates and thermodynamic data."""

    site_id: int
    x: float
    y: float
    z: float
    dG: float
    occupancy: float = 1.0


def classify_water(
    dG: float,
    very_unhappy: float = _VERY_UNHAPPY_THRESHOLD,
    unhappy: float = _UNHAPPY_THRESHOLD,
    happy: float = _HAPPY_THRESHOLD,
) -> WaterCategory:
    """Return the thermodynamic category for a given dG value."""
    if dG > very_unhappy:
        return WaterCategory.VERY_UNHAPPY
    if dG >= unhappy:
        return WaterCategory.UNHAPPY
    if dG > happy:
        return WaterCategory.BULK_LIKE
    return WaterCategory.HAPPY


def _format_hetatm(
    serial: int,
    resname: str,
    chain: str,
    resseq: int,
    x: float,
    y: float,
    z: float,
    occupancy: float,
    bfactor: float,
) -> str:
    """Format a single HETATM line in PDB format (v3.0)."""
    return (
        f"HETATM{serial:5d}    O {resname:>3s} {chain:1s}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{bfactor:6.2f}           O"
    )


_PDB_REMARKS = """\
REMARK 888
REMARK 888 WRITTEN BY WATER_ANNOTATOR
REMARK 999 CONVERTED FOR ROBUSTER VISUALIZATION
REMARK 999 ALL ATOMS SET TO ELEMENT O
REMARK 999 ORIGINAL CLASS ENCODED IN RESNAME: WVU/WBL/WHP/WUN
REMARK 999 VERY 'UNHAPPY' WATER (3.5<\u0394G kcal/mol): WVU
REMARK 999 'UNHAPPY' WATER (2.0<\u0394G<3.5 kcal/mol): WUN
REMARK 999 BULK SOLVENT-LIKE WATER (-1.0<\u0394G<2.0 kcal/mol): WBL
REMARK 999 'HAPPY' WATER (\u0394G<-1.0 kcal/mol): WHP
REMARK 999 \u0394G STORED IN B-FACTOR\
"""


class BaseWaterAnnotator(ABC):
    """Abstract base for software-specific water annotators.

    Subclasses must implement:
        - ``_parse_water_sites`` – read input files and return water sites
        - ``software_name``      – class-level string attribute

    The base class handles:
        - Classification of waters by free-energy thresholds
        - Writing the annotated PDB file
    """

    software_name: str = ""

    def __init__(
        self,
        *,
        very_unhappy_threshold: float = _VERY_UNHAPPY_THRESHOLD,
        unhappy_threshold: float = _UNHAPPY_THRESHOLD,
        happy_threshold: float = _HAPPY_THRESHOLD,
    ) -> None:
        self.very_unhappy_threshold = very_unhappy_threshold
        self.unhappy_threshold = unhappy_threshold
        self.happy_threshold = happy_threshold
        self._sites: list[WaterSite] = []

    @abstractmethod
    def _parse_water_sites(self) -> list[WaterSite]:
        """Parse input files and return a list of hydration sites."""

    def _classify(self, dG: float) -> WaterCategory:
        return classify_water(
            dG,
            very_unhappy=self.very_unhappy_threshold,
            unhappy=self.unhappy_threshold,
            happy=self.happy_threshold,
        )

    def annotate(self, output_path: str | Path, title: str = "") -> Path:
        """Parse water sites, classify, and write an annotated PDB.

        Parameters
        ----------
        output_path:
            Destination PDB file path.
        title:
            Optional TITLE record for the PDB header.

        Returns
        -------
        Path
            Resolved *output_path*.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sites = self._parse_water_sites()
        self._sites = sites

        lines: list[str] = []
        if title:
            lines.append(f"TITLE     {title}")
        lines.append("REMARK   4      COMPLIES WITH FORMAT V. 3.0, 1-DEC-2006")
        lines.append(_PDB_REMARKS)

        for idx, site in enumerate(sites, start=1):
            category = self._classify(site.dG)
            lines.append(
                _format_hetatm(
                    serial=idx,
                    resname=category.value,
                    chain="A",
                    resseq=site.site_id,
                    x=site.x,
                    y=site.y,
                    z=site.z,
                    occupancy=site.occupancy,
                    bfactor=site.dG,
                )
            )

        lines.append("END")
        output_path.write_text("\n".join(lines) + "\n")

        n_by_cat = {}
        for site in sites:
            cat = self._classify(site.dG)
            n_by_cat[cat.value] = n_by_cat.get(cat.value, 0) + 1
        logger.info(
            "Wrote %d water sites (%s) to %s",
            len(sites),
            ", ".join(f"{k}={v}" for k, v in sorted(n_by_cat.items())),
            output_path.name,
        )
        return output_path

    @property
    def sites(self) -> list[WaterSite]:
        """Water sites from the last ``annotate()`` call."""
        return list(self._sites)

    def __len__(self) -> int:
        return len(self._sites)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_sites={len(self._sites)})"
        )
