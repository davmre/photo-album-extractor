from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np

from core.geometry import is_rectangle
from core.photo_types import (
    BoundingBoxAny,
    PhotoOrientation,
    QuadArray,
    bounding_box_as_array,
)

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
    orientation: PhotoOrientation = PhotoOrientation.NORMAL

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> PhotoAttributes:
        # Handle orientation with backwards compatibility
        orientation_str = data.get("orientation", PhotoOrientation.NORMAL.value)
        try:
            orientation = PhotoOrientation(orientation_str)
        except ValueError:
            orientation = PhotoOrientation.NORMAL

        return cls(
            date_hint=data.get(
                "date_hint",
                # Try old names for backwards compatibility
                data.get("date_string", data.get("date_time", "")),
            ),
            exif_date=data.get("exif_date", ""),
            comments=data.get("comments", ""),
            date_inconsistent=data.get("date_inconsistent", "false").lower() == "true",
            orientation=orientation,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "date_hint": self.date_hint,
            "exif_date": self.exif_date,
            "comments": self.comments,
            "date_inconsistent": str(self.date_inconsistent),
            "orientation": self.orientation.value,
        }

    def __bool__(self) -> bool:
        return bool(self.date_hint or self.comments)

    def __hash__(self):
        return hash(
            (
                self.date_hint,
                self.exif_date,
                self.comments,
                self.date_inconsistent,
                self.orientation,
            )
        )

    def copy(self) -> PhotoAttributes:
        return PhotoAttributes(
            date_hint=self.date_hint,
            exif_date=self.exif_date,
            date_inconsistent=self.date_inconsistent,
            comments=self.comments,
            orientation=self.orientation,
        )


@dataclass
class BoundingBox:
    """Data model for a photo bounding box."""

    corners: QuadArray
    box_id: str
    attributes: PhotoAttributes
    marked_as_good: bool = False

    @classmethod
    def new(cls, corners: BoundingBoxAny, attributes: PhotoAttributes | None = None):
        return BoundingBox(
            corners=bounding_box_as_array(corners),
            box_id=str(uuid.uuid4()),
            attributes=attributes if attributes else PhotoAttributes(),
        )

    def __hash__(self):
        corners = ((c[0], c[1]) for c in self.corners)  # convert from ndarray
        return hash((corners, self.box_id, self.attributes, self.marked_as_good))

    def __eq__(self, other):
        if (self.box_id, self.attributes, self.marked_as_good) != (
            other.box_id,
            other.attributes,
            other.marked_as_good,
        ):
            return False
        try:
            return np.all(self.corners == other.corners)
        except Exception:
            return False

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": "quad",
            "corners": [[corner[0], corner[1]] for corner in self.corners],
            "attributes": self.attributes.to_dict(),
            "id": self.box_id,
            "marked_as_good": self.marked_as_good,
        }

    @classmethod
    def from_dict(cls, data: dict) -> BoundingBox:
        """Create from dictionary loaded from JSON."""

        # Extract and convert corners
        corners = data.get("corners", [])
        corners_array = np.array(corners, dtype=np.float64)

        # Extract attributes
        attributes_dict = data.get("attributes", {})
        attributes = PhotoAttributes.from_dict(attributes_dict)

        # Create BoundingBoxData
        return cls(
            corners=corners_array,
            box_id=data["id"],
            attributes=attributes,
            marked_as_good=data.get("marked_as_good", False),
        )

    def is_rectangle(self, tolerance_degrees: float = 1e-1) -> bool:
        """Check if this bounding box is a rectangle."""
        return is_rectangle(self.corners, tolerance_degrees=tolerance_degrees)
