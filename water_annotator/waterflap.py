"""Water annotator for WaterFLAP output.

WaterFLAP produces hydration-site predictions as a PDB file where each
HETATM record represents a water site.  The free-energy value (ΔG) is
stored in the B-factor column and the atom element type encodes the
original WaterFLAP classification.

This annotator reads the raw WaterFLAP PDB, classifies each site by its
ΔG value, and writes a standardised PDB with uniform oxygen atoms and
residue names encoding the thermodynamic category.

The classification thresholds are shared with the base annotator.
"""

from __future__ import annotations

from pathlib import Path

from water_annotator.base import BaseWaterAnnotator, WaterSite


def _parse_waterflap_pdb(pdb_path: Path) -> list[WaterSite]:
    """Read a WaterFLAP PDB and extract water sites.

    Each HETATM record is a hydration site.  The site ID is taken from
    the residue sequence number and the ΔG from the B-factor column.
    """
    sites: list[WaterSite] = []
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("HETATM"):
                continue
            site_id = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            occupancy = float(line[54:60])
            dG = float(line[60:66])
            sites.append(
                WaterSite(
                    site_id=site_id,
                    x=x,
                    y=y,
                    z=z,
                    dG=dG,
                    occupancy=occupancy,
                )
            )
    return sites


class WaterFLAPAnnotator(BaseWaterAnnotator):
    """Annotate hydration sites from WaterFLAP output.

    Parameters
    ----------
    pdb_path:
        WaterFLAP output PDB file containing HETATM records for each
        predicted hydration site, with ΔG in the B-factor column.
    """

    software_name: str = "waterflap"

    def __init__(
        self,
        pdb_path: str | Path,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.pdb_path = Path(pdb_path)
        if not self.pdb_path.exists():
            raise FileNotFoundError(f"PDB file not found: {self.pdb_path}")

    def _parse_water_sites(self) -> list[WaterSite]:
        return _parse_waterflap_pdb(self.pdb_path)
