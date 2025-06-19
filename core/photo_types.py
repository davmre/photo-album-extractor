"""
Core type definitions for Photo Album Extractor.

This module defines semantic type aliases and protocols used throughout the codebase
to provide clear interfaces and better type safety.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NewType, Protocol, Union

import numpy as np
import numpy.typing as npt
from PyQt6.QtCore import QPointF

# =============================================================================
# Semantic Types for Stronger Type Safety
# =============================================================================

ImageCoordinate = NewType("ImageCoordinate", tuple[float, float])


# File paths with semantic meaning
DirectoryPath = NewType("DirectoryPath", str)

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
class PhotoAttributes:
    """Attributes associated with a photo bounding box."""

    date_time: str = ""
    comments: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> PhotoAttributes:
        return cls(
            date_time=data.get("date_time", ""), comments=data.get("comments", "")
        )

    def to_dict(self) -> dict[str, str]:
        return {"date_time": self.date_time, "comments": self.comments}

    def __bool__(self) -> bool:
        return bool(self.date_time or self.comments)


def bounding_box_as_array(corner_points: BoundingBoxAny) -> QuadArray:
    try:
        # Handle arrays and lists-of-tuple-coordinates
        result = np.asarray(corner_points, dtype=float)
    except:  # QpointF
        try:
            result = np.array(
                [
                    (p.x(), p.y())  # type: ignore
                    for p in corner_points
                ],
                dtype=float,
            )
        except:
            raise Exception(
                f"Object {corner_points} not interpretable as a bounding box."
            )
    return result


def bounding_box_as_list_of_qpointfs(
    corner_points: BoundingBoxAny,
) -> BoundingBoxQPointF:
    if isinstance(corner_points[0], QPointF):
        return corner_points  # type: ignore
    return [QPointF(float(p[0]), float(p[1])) for p in corner_points]  # type: ignore


# =============================================================================
# Protocol Definitions
# =============================================================================


class BoundaryRefinementStrategy(Protocol):
    """Protocol for boundary refinement algorithms with semantic types."""

    def refine_boundary(
        self,
        image: BGRImage,
        initial_quad: QuadArray,
        debug_dir: DirectoryPath | None = None,
        **kwargs: Any,
    ) -> QuadArray: ...

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...
