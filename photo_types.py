"""
Core type definitions for Photo Album Extractor.

This module defines semantic type aliases and protocols used throughout the codebase
to provide clear interfaces and better type safety.
"""

from __future__ import annotations

from typing import Any, NewType, Optional, Protocol, Tuple

import numpy as np
import numpy.typing as npt
import PIL.Image

# =============================================================================
# Semantic Types for Stronger Type Safety
# =============================================================================

ImageCoordinate = NewType('ImageCoordinate', Tuple[float, float])

# File paths with semantic meaning
DirectoryPath = NewType('DirectoryPath', str)

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
# Protocol Definitions
# =============================================================================

class BoundaryRefinementStrategy(Protocol):
    """Protocol for boundary refinement algorithms with semantic types."""
    def refine_boundary(
        self, 
        image: BGRImage, 
        initial_quad: QuadArray,
        debug_dir: Optional[DirectoryPath] = None,
        **kwargs: Any
    ) -> QuadArray: ...
    
    @property
    def name(self) -> str: ...
    
    @property  
    def description(self) -> str: ...

