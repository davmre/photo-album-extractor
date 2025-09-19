"""
Centralized validation state management for the application.

This module provides a global ValidationManager that serves as the single source
of truth for all validation state throughout the application.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal

from core.bounding_box import BoundingBox
from core.bounding_box_storage import BoundingBoxStorage
from core.validation_utils import (
    FileValidationSummary,
    ValidationIssue,
    validate_bounding_box,
    validate_file_bounding_boxes,
)


class ValidationManager(QObject):
    """
    Centralized manager for validation state across the application.

    Maintains cached validation results and coordinates updates when
    validation settings change or bounding boxes are modified.
    """

    # Signals emitted when validation state changes
    box_validation_changed = pyqtSignal(str, str, str)  # directory, filename, box_id
    file_validation_changed = pyqtSignal(str, str)  # directory, filename
    directory_validation_changed = pyqtSignal(str)  # directory

    def __init__(self):
        super().__init__()

        # Cache structure: {(directory, filename, box_id): List[ValidationIssue]}
        self._box_validation_cache: dict[
            tuple[str, str, str], list[ValidationIssue]
        ] = {}

        # Cache structure: {(directory, filename): FileValidationSummary}
        self._file_validation_cache: dict[tuple[str, str], FileValidationSummary] = {}

        # Cache structure: {directory: Dict[filename, FileValidationSummary]}
        self._directory_validation_cache: dict[
            str, dict[str, FileValidationSummary]
        ] = {}

    def get_box_validation(
        self, directory: Path, filename: str, box_data: BoundingBox
    ) -> list[ValidationIssue]:
        """
        Get validation issues for a specific bounding box.

        Args:
            directory: Directory containing the image
            filename: Image filename
            box_data: BoundingBoxData to validate

        Returns:
            List of validation issues for the box
        """
        cache_key = (str(directory), filename, box_data.box_id)

        if cache_key not in self._box_validation_cache:
            # Validate and cache the result
            issues = validate_bounding_box(box_data)
            self._box_validation_cache[cache_key] = issues

        return self._box_validation_cache[cache_key]

    def get_file_validation(
        self, directory: Path, filename: str, storage: BoundingBoxStorage | None = None
    ) -> FileValidationSummary:
        """
        Get validation summary for all bounding boxes in a file.

        Args:
            directory: Directory containing the image
            filename: Image filename
            storage: Optional BoundingBoxStorage instance to use

        Returns:
            FileValidationSummary for the file
        """
        cache_key = (str(directory), filename)

        if cache_key not in self._file_validation_cache:
            # Create storage if not provided
            if storage is None:
                storage = BoundingBoxStorage(directory)

            # Validate and cache the result
            summary = validate_file_bounding_boxes(directory, filename, storage)
            self._file_validation_cache[cache_key] = summary

        return self._file_validation_cache[cache_key]

    def get_directory_validation(
        self, directory: Path, storage: BoundingBoxStorage | None = None
    ) -> dict[str, FileValidationSummary]:
        """
        Get validation summaries for all files in a directory.

        Args:
            directory: Directory to validate
            storage: Optional BoundingBoxStorage instance to use

        Returns:
            Dict mapping filename to FileValidationSummary
        """
        if str(directory) not in self._directory_validation_cache:
            # Create storage if not provided
            if storage is None:
                storage = BoundingBoxStorage(directory)

            # Load all filenames from storage
            filenames = storage.load_image_filenames()

            # Validate each file and build cache
            file_summaries = {}
            for filename in filenames:
                summary = self.get_file_validation(directory, filename, storage)
                file_summaries[filename] = summary

            self._directory_validation_cache[str(directory)] = file_summaries

        return self._directory_validation_cache[str(directory)]

    def invalidate_box(self, directory: Path, filename: str, box_id: str) -> None:
        """
        Invalidate cached validation for a specific bounding box.

        Args:
            directory: Directory containing the image
            filename: Image filename
            box_id: ID of the bounding box to invalidate
        """
        cache_key = (str(directory), filename, box_id)

        if cache_key in self._box_validation_cache:
            del self._box_validation_cache[cache_key]

        # Also invalidate the file-level cache since it depends on box validation
        self.invalidate_file(directory, filename)

        # Emit signal that box validation changed
        self.box_validation_changed.emit(str(directory), filename, box_id)

    def invalidate_file(self, directory: Path, filename: str) -> None:
        """
        Invalidate cached validation for all boxes in a file.

        Args:
            directory: Directory containing the image
            filename: Image filename
        """
        file_cache_key = (str(directory), filename)

        # Remove file-level cache
        if file_cache_key in self._file_validation_cache:
            del self._file_validation_cache[file_cache_key]

        # Remove all box-level caches for this file
        keys_to_remove = [
            key
            for key in self._box_validation_cache.keys()
            if key[0] == str(directory) and key[1] == filename
        ]
        for key in keys_to_remove:
            del self._box_validation_cache[key]

        # Update directory cache if it exists
        if str(directory) in self._directory_validation_cache:
            if filename in self._directory_validation_cache[str(directory)]:
                del self._directory_validation_cache[str(directory)][filename]

        # Emit signal that file validation changed
        self.file_validation_changed.emit(str(directory), filename)

    def invalidate_directory(self, directory: Path) -> None:
        """
        Invalidate cached validation for an entire directory.

        Args:
            directory: Directory to invalidate
        """
        # Remove directory-level cache
        if str(directory) in self._directory_validation_cache:
            del self._directory_validation_cache[str(directory)]

        # Remove all file-level caches for this directory
        file_keys_to_remove = [
            key
            for key in self._file_validation_cache.keys()
            if key[0] == str(directory)
        ]
        for key in file_keys_to_remove:
            del self._file_validation_cache[key]

        # Remove all box-level caches for this directory
        box_keys_to_remove = [
            key for key in self._box_validation_cache.keys() if key[0] == str(directory)
        ]
        for key in box_keys_to_remove:
            del self._box_validation_cache[key]

        # Emit signal that directory validation changed
        self.directory_validation_changed.emit(str(directory))

    def invalidate_all(self) -> None:
        """
        Invalidate all cached validation results.

        This is typically called when validation settings change,
        requiring all validation to be recalculated.
        """
        directories = list(self._directory_validation_cache.keys())

        # Clear all caches
        self._box_validation_cache.clear()
        self._file_validation_cache.clear()
        self._directory_validation_cache.clear()

        # Emit signals for all affected directories
        for directory in directories:
            self.directory_validation_changed.emit(directory)

    def update_box_validation(
        self,
        directory: Path,
        filename: str,
        box_data: BoundingBox,
        storage: BoundingBoxStorage | None = None,
    ) -> None:
        """
        Force update validation for a specific bounding box.

        Args:
            directory: Directory containing the image
            filename: Image filename
            box_data: Updated BoundingBoxData
            storage: Optional BoundingBoxStorage instance to use
        """
        # Invalidate and recalculate
        self.invalidate_box(directory, filename, box_data.box_id)

        # Force recalculation by accessing the validation
        self.get_box_validation(directory, filename, box_data)
        self.get_file_validation(directory, filename, storage)

    def update_file_validation(
        self, directory: Path, filename: str, storage: BoundingBoxStorage | None = None
    ) -> None:
        """
        Force update validation for all boxes in a file.

        Args:
            directory: Directory containing the image
            filename: Image filename
            storage: Optional BoundingBoxStorage instance to use
        """
        # Invalidate and recalculate
        self.invalidate_file(directory, filename)

        # Force recalculation by accessing the validation
        self.get_file_validation(directory, filename, storage)

    def get_cache_stats(self) -> dict[str, int]:
        """
        Get statistics about current cache usage.

        Returns:
            Dict with cache size information
        """
        return {
            "box_validations": len(self._box_validation_cache),
            "file_validations": len(self._file_validation_cache),
            "directories": len(self._directory_validation_cache),
        }


# Global validation manager instance
validation_manager = ValidationManager()
