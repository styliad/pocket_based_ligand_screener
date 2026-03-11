"""Water annotator for Schrödinger WaterMap output.

WaterMap produces hydration-site thermodynamic data as a CSV export and
site coordinates in a separate PDB file.  This annotator merges the two
sources by site index and writes a classified PDB.

Expected CSV columns (as exported from Maestro)::

    Site, Entry ID, Occupancy, Overlap, dH, -TdS, dG, #HB(WW), #HB(PW), #HB(LW)
"""

from __future__ import annotations

import csv
from pathlib import Path

from water_annotator.base import BaseWaterAnnotator, WaterSite
from water_annotator.standardiser import validate_watermap_csv

# CSV column names (after stripping whitespace).
_COL_SITE = "Site"
_COL_OCCUPANCY = "Occupancy"
_COL_DG = "dG"


def _parse_watermap_csv(csv_path: Path) -> dict[int, dict]:
    """Read WaterMap CSV and return ``{site_id: {dG, occupancy}}``."""
    rows: dict[int, dict] = {}
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # Strip whitespace from keys (Maestro exports can have spaces).
            row = {k.strip(): v.strip() for k, v in row.items()}
            site_id = int(row[_COL_SITE])
            rows[site_id] = {
                "dG": float(row[_COL_DG]),
                "occupancy": float(row[_COL_OCCUPANCY]),
            }
    return rows


def _parse_water_pdb(pdb_path: Path) -> list[dict]:
    """Read a PDB with one oxygen per hydration site, ordered by site.

    Returns a list of ``{serial, x, y, z}`` dicts, one per HETATM/ATOM
    record, in file order.
    """
    waters: list[dict] = []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith(("HETATM", "ATOM  ")):
                waters.append(
                    {
                        "serial": int(line[6:11]),
                        "x": float(line[30:38]),
                        "y": float(line[38:46]),
                        "z": float(line[46:54]),
                    }
                )
    return waters


class WaterMapAnnotator(BaseWaterAnnotator):
    """Annotate hydration sites from Schrödinger WaterMap output.

    Parameters
    ----------
    csv_path:
        WaterMap CSV export with thermodynamic data per site.
    water_pdb_path:
        PDB file containing one oxygen atom per hydration site, in the
        same order as the CSV site numbering (site 1 = first HETATM, etc.).
    """

    software_name: str = "watermap"

    def __init__(
        self,
        csv_path: str | Path,
        water_pdb_path: str | Path,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.csv_path = Path(csv_path)
        self.water_pdb_path = Path(water_pdb_path)
        if not self.water_pdb_path.exists():
            raise FileNotFoundError(f"PDB file not found: {self.water_pdb_path}")
        validate_watermap_csv(self.csv_path)

    def _parse_water_sites(self) -> list[WaterSite]:
        csv_data = _parse_watermap_csv(self.csv_path)
        pdb_waters = _parse_water_pdb(self.water_pdb_path)

        if len(pdb_waters) != len(csv_data):
            raise ValueError(
                f"Site count mismatch: CSV has {len(csv_data)} sites, "
                f"PDB has {len(pdb_waters)} HETATM records."
            )

        sites: list[WaterSite] = []
        for idx, water in enumerate(pdb_waters):
            site_id = idx + 1  # 1-based
            if site_id not in csv_data:
                raise ValueError(
                    f"Site {site_id} present in PDB but missing from CSV."
                )
            data = csv_data[site_id]
            sites.append(
                WaterSite(
                    site_id=site_id,
                    x=water["x"],
                    y=water["y"],
                    z=water["z"],
                    dG=data["dG"],
                    occupancy=data["occupancy"],
                )
            )
        return sites
