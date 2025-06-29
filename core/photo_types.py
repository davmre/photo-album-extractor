"""
Core type definitions for Photo Album Extractor.

This module defines semantic type aliases and protocols used throughout the codebase
to provide clear interfaces and better type safety.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Any, NewType, Union

import numpy as np
import numpy.typing as npt
from PyQt6.QtCore import QPointF

# =============================================================================
# Semantic Types for Stronger Type Safety
# =============================================================================

ImageCoordinate = NewType("ImageCoordinate", tuple[float, float])


# =============================================================================
# Photo Orientation
# =============================================================================


class PhotoOrientation(Enum):
    """Photo orientation relative to its normal upright position."""

    NORMAL = "normal"                    # 0 degrees (right-side up)
    ROTATED_90_CW = "rotated_90_cw"     # 90 degrees clockwise
    UPSIDE_DOWN = "upside_down"         # 180 degrees
    ROTATED_90_CCW = "rotated_90_ccw"   # 90 degrees counterclockwise

    @property
    def rotation_degrees(self) -> int:
        """Return the rotation in degrees needed to make the photo upright."""
        rotation_map = {
            PhotoOrientation.NORMAL: 0,
            PhotoOrientation.ROTATED_90_CW: -90,     # Need to rotate 90° CCW to fix
            PhotoOrientation.UPSIDE_DOWN: 180,
            PhotoOrientation.ROTATED_90_CCW: 90,     # Need to rotate 90° CW to fix
        }
        return rotation_map[self]

    @property
    def display_name(self) -> str:
        """Return a user-friendly display name."""
        display_map = {
            PhotoOrientation.NORMAL: "Normal (0°)",
            PhotoOrientation.ROTATED_90_CW: "Rotated 90° CW",
            PhotoOrientation.UPSIDE_DOWN: "Upside Down (180°)",
            PhotoOrientation.ROTATED_90_CCW: "Rotated 90° CCW",
        }
        return display_map[self]

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
