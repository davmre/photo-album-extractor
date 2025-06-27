"""
Validation utilities for file-level validation operations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

from core.bounding_box_data import Severity
from core.bounding_box_storage import BoundingBoxStorage


class FileValidationSeverity(Enum):
    """File-level validation severity based on worst issue in the file."""

    CLEAN = "clean"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class FileValidationSummary:
    """Summary of validation issues for a single file."""

    file_path: str
    severity: FileValidationSeverity
    error_count: int
    warning_count: int


def validate_file_bounding_boxes(
    directory: str, filename: str, storage: BoundingBoxStorage | None = None
) -> FileValidationSummary:
    """
    Validate all bounding boxes in a single image file.

    Args:
        directory: Directory containing the image and storage data
        filename: Image filename (without path)
        storage: Optional BoundingBoxStorage object to use (avoids creating new one)

    Returns:
        FileValidationSummary with counts and worst severity
    """
    if storage is None:
        storage = BoundingBoxStorage(directory)
    bounding_boxes = storage.load_bounding_boxes(filename)

    error_count = 0
    warning_count = 0

    for box in bounding_boxes:
        issues = box.validate()
        for issue in issues:
            if issue.severity == Severity.ERROR:
                error_count += 1
            else:
                warning_count += 1

    # Determine overall severity
    if error_count > 0:
        severity = FileValidationSeverity.ERROR
    elif warning_count > 0:
        severity = FileValidationSeverity.WARNING
    else:
        severity = FileValidationSeverity.CLEAN

    file_path = os.path.join(directory, filename)
    return FileValidationSummary(
        file_path=file_path,
        severity=severity,
        error_count=error_count,
        warning_count=warning_count,
    )


def validate_directory_files(
    directory: str, storage: BoundingBoxStorage
) -> dict[str, FileValidationSummary]:
    """
    Validate all files with bounding box data in a directory.

    Args:
        directory: Directory to scan for image files with bounding box data
        storage: Optional BoundingBoxStorage object to use (avoids creating new one)

    Returns:
        Dict mapping filename to FileValidationSummary
    """
    if not os.path.isdir(directory):
        return {}

    filenames = storage.load_image_filenames()

    results = {}
    for filename in filenames:
        results[filename] = validate_file_bounding_boxes(directory, filename, storage)

    return results


def get_validation_icon_text(summary: FileValidationSummary) -> str:
    """
    Get the validation icon text to display next to a filename.

    Args:
        summary: FileValidationSummary for the file

    Returns:
        String with appropriate emoji icons, or empty string if clean
    """
    if summary.severity == FileValidationSeverity.ERROR:
        return "🚨"
    elif summary.severity == FileValidationSeverity.WARNING:
        return "⚠️"
    else:
        return ""
