"""Base standardiser for water-analysis CSV inputs.

Each software tool (WaterMap, GIST, etc.) exports hydration-site data as CSV
with its own column schema.  A standardiser validates that an input CSV
contains the columns expected by the corresponding downstream annotator.
"""

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from pathlib import Path


class CSVValidationError(ValueError):
    """Raised when a water-analysis CSV does not match the expected schema."""


class BaseCSVStandardiser(ABC):
    """Base class for CSV column validation.

    Subclasses define ``expected_columns`` and optionally override
    ``_post_validate`` for additional checks beyond column presence.
    """

    @property
    @abstractmethod
    def expected_columns(self) -> frozenset[str]:
        """Column names that must be present in the CSV."""

    def validate(self, csv_path: str | Path) -> list[str]:
        """Validate that *csv_path* has the required columns.

        Parameters
        ----------
        csv_path:
            Path to the CSV file to validate.

        Returns
        -------
        list[str]
            The column names found in the CSV (stripped of whitespace).

        Raises
        ------
        FileNotFoundError
            If *csv_path* does not exist.
        CSVValidationError
            If columns are missing or the file is empty / unreadable.
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                raise CSVValidationError(
                    f"Cannot read column headers from {csv_path}. "
                    "The file may be empty or not a valid CSV."
                )
            columns = [c.strip() for c in reader.fieldnames]

        found = set(columns)
        missing = self.expected_columns - found
        if missing:
            raise CSVValidationError(
                f"Missing required columns in {csv_path.name}: "
                f"{', '.join(sorted(missing))}. "
                f"Expected columns: {', '.join(sorted(self.expected_columns))}."
            )

        self._post_validate(csv_path, columns)
        return columns

    def _post_validate(
        self, csv_path: Path, columns: list[str]
    ) -> None:
        """Hook for additional validation after column checks.

        Override in subclasses to add software-specific checks.
        Does nothing by default.
        """
