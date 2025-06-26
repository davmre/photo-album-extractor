"""
Core type definitions for Photo Album Extractor.

This module defines semantic type aliases and protocols used throughout the codebase
to provide clear interfaces and better type safety.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NewType, Union

import numpy as np
import numpy.typing as npt
from PyQt6.QtCore import QPointF

# =============================================================================
# Semantic Types for Stronger Type Safety
# =============================================================================

ImageCoordinate = NewType("ImageCoordinate", tuple[float, float])

# =============================================================================
# Array Types with Shape Constraints
# =============================================================================

# General-purpose array type aliases for cleaner code
FloatArray = npt.NDArray[np.floating[Any]]  # General floating-point arrays
IntArray = npt.NDArray[np.int_]  # General integer arrays
UInt8Array = npt.NDArray[np.uint8]  # General uint8 arrays
AnyArray = npt.NDArray[Any]  # Generic arrays when dtype is mixed/unknown

# Shape-constrained arrays for better type safety
# More specific array types for documentation and IDE support
QuadArray = npt.NDArray[np.float64]  # Shape: (4, 2) - four corner points
TransformMatrix = npt.NDArray[np.float64]  # Shape: (3, 3) - perspective transform
BGRImage = npt.NDArray[np.uint8]  # Shape: (H, W, 3) - BGR color image

# =============================================================================
# Bounding box quadrilaterals
# =============================================================================

BoundingBoxFloatPoints = Sequence[tuple[float, float]]
BoundingBoxQPointF = list[QPointF]

BoundingBoxAny = Union[BoundingBoxFloatPoints, BoundingBoxQPointF, QuadArray]


# =============================================================================
# Photo Attributes
# =============================================================================


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


def bounding_box_as_array(corner_points: BoundingBoxAny) -> QuadArray:
    try:
        # Handle arrays and lists-of-tuple-coordinates
        result = np.asarray(corner_points, dtype=float)
    except Exception:  # QpointF
        try:
            result = np.array(
                [
                    (p.x(), p.y())  # type: ignore
                    for p in corner_points
                ],
                dtype=float,
            )
        except Exception:
            raise ValueError(
                f"Object {corner_points} not interpretable as a bounding box."
            ) from None
    return result


def bounding_box_as_list_of_qpointfs(
    corner_points: BoundingBoxAny,
) -> BoundingBoxQPointF:
    if isinstance(corner_points[0], QPointF):
        return corner_points  # type: ignore
    return [QPointF(float(p[0]), float(p[1])) for p in corner_points]  # type: ignore
