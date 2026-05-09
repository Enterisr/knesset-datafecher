from __future__ import annotations

import os
from collections import defaultdict

from .logger_config import get_logger


class DuplicateFilesError(RuntimeError):
    """Raised when duplicate output filenames are detected across partitions."""


class DuplicateFileChecker:
    """Check duplicate JSON files after data fetching."""

    def __init__(self, output_folder: str = "committee_data"):
        self.output_folder = output_folder
        self.logger = get_logger(__name__)

    def check_for_duplicates(self) -> bool:
        if not os.path.exists(self.output_folder):
            self.logger.warning("Output folder '%s' does not exist", self.output_folder)
            return True

        duplicates = self._find_duplicate_files()
        if duplicates:
            self._log_duplicates(duplicates)
            raise DuplicateFilesError(
                f"Duplicate protocol JSON filenames detected in '{self.output_folder}'"
            )

        self.logger.info("No duplicate JSON files found")
        return True

    def _find_duplicate_files(self) -> dict[str, list[str]]:
        file_map: dict[str, list[str]] = defaultdict(list)

        for root, _dirs, files in os.walk(self.output_folder):
            for file in files:
                if file.endswith(".json"):
                    full_path = os.path.join(root, file)
                    file_map[file].append(full_path)

        return {filename: paths for filename, paths in file_map.items() if len(paths) > 1}

    def _log_duplicates(self, duplicates: dict[str, list[str]]) -> None:
        self.logger.error("DUPLICATE FILES DETECTED!")
        self.logger.error(
            "Found %d duplicate filename(s) (%d total files)",
            len(duplicates),
            sum(len(paths) for paths in duplicates.values()),
        )
        for filename, paths in duplicates.items():
            self.logger.error("Duplicate file '%s' found in %d locations:", filename, len(paths))
            for path in paths:
                self.logger.error("  - %s", path)


def check_for_duplicate_files(output_folder: str = "committee_data") -> bool:
    checker = DuplicateFileChecker(output_folder)
    return checker.check_for_duplicates()
