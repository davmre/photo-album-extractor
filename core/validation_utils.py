"""
Validation utilities for file-level validation operations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

import numpy as np

from core import date_utils, geometry
from core.bounding_box_storage import BoundingBoxStorage


class Severity(Enum):
    """Validation issue severity levels."""

    ERROR = "error"
    WARNING = "warning"


COMMON_ASPECT_RATIOS = np.array([4 / 6, 6 / 4, 5 / 7, 7 / 5])
print(COMMON_ASPECT_RATIOS)


@dataclass
class ValidationIssue:
    """Represents a validation issue with a bounding box."""

    type: str
    severity: Severity
    message: str


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


def validate_bounding_box(box) -> list[ValidationIssue]:
    """Validate the bounding box and return any issues found."""
    issues = []

    # Check for unparseable date hint (ERROR)
    if box.attributes.date_hint.strip():
        parsed_date = date_utils.parse_flexible_date_as_datetime(
            box.attributes.date_hint
        )
        if parsed_date is None:
            issues.append(
                ValidationIssue(
                    type="unparseable_date",
                    severity=Severity.ERROR,
                    message="Date hint cannot be parsed",
                )
            )

    # If marked as good, skip checks for warnings (everything below this point).
    if box.marked_as_good:
        return issues

    # Check for date inconsistency (WARNING)
    if box.attributes.date_inconsistent:
        issues.append(
            ValidationIssue(
                type="date_inconsistent",
                severity=Severity.WARNING,
                message="Dates out of order: preceding photos have later dates than this one.",
            )
        )

    # Check if bounding box is not rectangular (WARNING)
    if not box.is_rectangle():
        issues.append(
            ValidationIssue(
                type="non_rectangular",
                severity=Severity.WARNING,
                message="Bounding box is not rectangular",
            )
        )

    # Check if aspect ratio is non-standard.
    width, height = geometry.dimension_bounds(box.corners)
    aspect_ratio_reciprocal = height / width
    if np.min(np.abs(aspect_ratio_reciprocal * COMMON_ASPECT_RATIOS - 1)) > 0.02:
        issues.append(
            ValidationIssue(
                type="nonstandard_aspect",
                severity=Severity.WARNING,
                message=f"Aspect ratio {width / height: .2f} is not a standard photo size; is the bounding box correct?",
            )
        )

    return issues


def validate_file_bounding_boxes(
    directory: str, filename: str, storage: BoundingBoxStorage
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
    bounding_boxes = storage.get_bounding_boxes(filename)

    error_count = 0
    warning_count = 0

    for box in bounding_boxes:
        issues = validate_bounding_box(box)
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
