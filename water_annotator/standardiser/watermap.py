"""Standardiser for Schrödinger WaterMap CSV exports."""

from __future__ import annotations

from pathlib import Path

from water_annotator.standardiser.base import BaseCSVStandardiser


class WaterMapCSVStandardiser(BaseCSVStandardiser):
    """Standardiser for Schrödinger WaterMap CSV exports."""

    @property
    def expected_columns(self) -> frozenset[str]:
        return frozenset(
            {
                "Site",
                "Entry ID",
                "Occupancy",
                "Overlap",
                "dH",
                "-TdS",
                "dG",
                "#HB(WW)",
                "#HB(PW)",
                "#HB(LW)",
            }
        )


#: Expected WaterMap columns (kept for direct imports).
EXPECTED_COLUMNS = WaterMapCSVStandardiser().expected_columns

_watermap_standardiser = WaterMapCSVStandardiser()


def validate_watermap_csv(csv_path: str | Path) -> list[str]:
    """Validate a WaterMap CSV — shortcut using :class:`WaterMapCSVStandardiser`."""
    return _watermap_standardiser.validate(csv_path)
