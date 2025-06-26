from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from PIL import Image

from core import geometry
from core.photo_types import QuadArray, UInt8Array
from core.settings import AppSettings
from photo_detection import refine_bounds, refine_strips


class RefinementStrategy(ABC):
    """Base class for photo detection strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the strategy."""
        pass

    @abstractmethod
    def refine(
        self,
        image: Image.Image | UInt8Array,
        corners: QuadArray,
        debug_dir: str | None = None,
    ) -> QuadArray:
        """
        Refine the corners of a photo's bounding box within a scanned image.
        """
        pass


class RefinementStrategyOriginalMultiscale(RefinementStrategy):
    @property
    def name(self):
        return "Original (multiscale)"

    def refine(
        self,
        image: Image.Image | UInt8Array,
        corner_points: QuadArray,
        debug_dir: str | None,
    ):
        return refine_bounds.refine_bounding_box_multiscale(
            image, corner_points, enforce_parallel_sides=True, debug_dir=debug_dir
        )


class RefinementStrategyStrips(RefinementStrategy):
    @property
    def name(self):
        return "Strips (native res)"

    def refine(
        self,
        image: Image.Image | UInt8Array,
        corner_points: QuadArray,
        debug_dir: str | None,
    ):
        return refine_strips.refine_bounding_box_strips(
            image,
            corner_points,
            enforce_parallel_sides=True,
            debug_dir=debug_dir,
            reltol=0.05,
        )


class RefinementStrategyStripsIterated(RefinementStrategy):
    def __init__(self, max_iterations=4, atol=2.0):
        self.max_iterations = max_iterations
        self.atol = atol

    @property
    def name(self):
        return "Strips (iterated)"

    def refine(
        self,
        image: Image.Image | UInt8Array,
        corner_points: QuadArray,
        debug_dir: str | None,
    ):
        for _ in range(self.max_iterations):
            new_corner_points = refine_strips.refine_bounding_box_strips(
                image, corner_points, enforce_parallel_sides=True, debug_dir=debug_dir
            )
            deviations = geometry.get_corner_deviations(
                corner_points, new_corner_points
            )
            if np.max(deviations) <= self.atol:
                break
            corner_points = new_corner_points
        return new_corner_points


_REFINEMENT_STRATEGIES: list[RefinementStrategy] = [
    RefinementStrategyStrips(),
    RefinementStrategyStripsIterated(),
    RefinementStrategyOriginalMultiscale(),
]

REFINEMENT_STRATEGIES: dict[str, RefinementStrategy] = {
    s.name: s for s in _REFINEMENT_STRATEGIES
}


def configure_refinement_strategy(settings: AppSettings) -> RefinementStrategy:
    strategy_name = settings.refinement_strategy
    return REFINEMENT_STRATEGIES.get(strategy_name, _REFINEMENT_STRATEGIES[0])
