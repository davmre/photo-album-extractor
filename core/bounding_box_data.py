from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum

from core import date_utils
from core.geometry import is_rectangle
from core.photo_types import BoundingBoxAny, QuadArray, bounding_box_as_array

# =============================================================================
# Validation
# =============================================================================


class Severity(Enum):
    """Validation issue severity levels."""

    ERROR = "error"
    WARNING = "warning"


@dataclass
class ValidationIssue:
    """Represents a validation issue with a bounding box."""

    type: str
    severity: Severity
    message: str


# =============================================================================
# Photo Attributes
# =============================================================================


@dataclass
class PhotoAttributes:
    """Attributes associated with a photo bounding box."""

    date_hint: str = ""
    exif_date: str = ""
    date_inconsistent: bool = False
    comments: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> PhotoAttributes:
        return cls(
            date_hint=data.get(
                "date_hint",
                # Try old names for backwards compatibility
                data.get("date_string", data.get("date_time", "")),
            ),
            exif_date=data.get("exif_date", ""),
            comments=data.get("comments", ""),
            date_inconsistent=data.get("date_inconsistent", "false").lower() == "true",
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "date_hint": self.date_hint,
            "exif_date": self.exif_date,
            "comments": self.comments,
            "date_inconsistent": str(self.date_inconsistent),
        }

    def __bool__(self) -> bool:
        return bool(self.date_hint or self.comments)

    def copy(self) -> PhotoAttributes:
        return PhotoAttributes(
            date_hint=self.date_hint,
            exif_date=self.exif_date,
            date_inconsistent=self.date_inconsistent,
            comments=self.comments,
        )


@dataclass
class BoundingBoxData:
    """Data model for a photo bounding box."""

    corners: QuadArray
    box_id: str
    attributes: PhotoAttributes

    @classmethod
    def new(cls, corners: BoundingBoxAny, attributes: PhotoAttributes | None = None):
        return BoundingBoxData(
            corners=bounding_box_as_array(corners),
            box_id=str(uuid.uuid4()),
            attributes=attributes if attributes else PhotoAttributes(),
        )

    def is_rectangle(self, tolerance_degrees: float = 1e-1) -> bool:
        """Check if this bounding box is a rectangle."""
        return is_rectangle(self.corners, tolerance_degrees=tolerance_degrees)

    def validate(self) -> list[ValidationIssue]:
        """Validate the bounding box and return any issues found."""
        issues = []

        # Check for unparseable date hint (ERROR)
        if self.attributes.date_hint.strip():
            parsed_date = date_utils.parse_flexible_date_as_datetime(
                self.attributes.date_hint
            )
            if parsed_date is None:
                issues.append(
                    ValidationIssue(
                        type="unparseable_date",
                        severity=Severity.ERROR,
                        message="Date hint cannot be parsed",
                    )
                )

        # Check for date inconsistency (WARNING)
        if self.attributes.date_inconsistent:
            issues.append(
                ValidationIssue(
                    type="date_inconsistent",
                    severity=Severity.WARNING,
                    message="Dates out of order: hinted date is earlier than preceding photos.",
                )
            )

        # Check if bounding box is not rectangular (WARNING)
        if not self.is_rectangle():
            issues.append(
                ValidationIssue(
                    type="non_rectangular",
                    severity=Severity.WARNING,
                    message="Bounding box is not rectangular",
                )
            )

        return issues
