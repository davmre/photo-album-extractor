from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from PIL import Image

from core import geometry, refine_strips, refine_strips_hough
from core.photo_types import QuadArray, UInt8Array
from core.settings import AppSettings, app_settings


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
        corner_points: QuadArray,
        reltol: float = 0.05,
        debug_dir: str | None = None,
    ) -> QuadArray:
        """
        Refine the corners of a photo's bounding box within a scanned image.
        """
        pass


class RefinementStrategyStrips(RefinementStrategy):
    @property
    def name(self):
        return "Strips (native res)"

    def refine(
        self,
        image: Image.Image | UInt8Array,
        corner_points: QuadArray,
        reltol: float = 0.05,
        debug_dir: str | None = None,
    ):
        return refine_strips.refine_bounding_box_strips(
            image,
            corner_points,
            enforce_parallel_sides=True,
            debug_dir=debug_dir,
            reltol=reltol,
        )


class RefinementStrategyStripsIndependent(RefinementStrategy):
    @property
    def name(self):
        return "Strips independent edges"

    def refine(
        self,
        image: Image.Image | UInt8Array,
        corner_points: QuadArray,
        reltol: float = 0.05,
        debug_dir: str | None = None,
    ):
        return refine_strips.refine_bounding_box_strips(
            image,
            corner_points,
            enforce_parallel_sides=False,
            debug_dir=debug_dir,
            reltol=reltol,
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
        reltol: float = 0.05,
        debug_dir: str | None = None,
    ):
        for _ in range(self.max_iterations):
            new_corner_points = refine_strips.refine_bounding_box_strips(
                image,
                corner_points,
                enforce_parallel_sides=True,
                debug_dir=debug_dir,
                reltol=reltol,
            )
            deviations = geometry.get_corner_deviations(
                corner_points, new_corner_points
            )
            if np.max(deviations) <= self.atol:
                break
            corner_points = new_corner_points
        return new_corner_points


class RefinementStrategyHoughReranked(RefinementStrategy):
    @property
    def name(self):
        return "Hough transform (reranked)"

    def refine(
        self,
        image: Image.Image | UInt8Array,
        corner_points: QuadArray,
        reltol: float = 0.05,
        debug_dir: str | None = None,
    ):
        print("refining with Hough and debug dir", debug_dir)
        return refine_strips_hough.refine_strips_hough(
            image,
            corner_points,
            debug_dir=debug_dir,
            reltol=reltol,
            max_candidate_angles=2,
            max_candidate_intercepts_per_angle=2,
            aspect_preference_strength=0.1,
            candidate_aspect_ratios=app_settings.standard_aspect_ratios,
        )


class RefinementStrategyHoughSoft(RefinementStrategy):
    @property
    def name(self):
        return "Hough transform (soft reranked)"

    def refine(
        self,
        image: Image.Image | UInt8Array,
        corner_points: QuadArray,
        reltol: float = 0.1,
        debug_dir: str | None = None,
    ):
        print("refining with Hough and debug dir", debug_dir)
        return refine_strips_hough.refine_strips_hough(
            image,
            corner_points,
            debug_dir=debug_dir,
            reltol=reltol,
            soft_boundaries=True,
            max_candidate_angles=2,
            max_candidate_intercepts_per_angle=2,
            include_single_strip_angle_hypotheses=True,
            aspect_preference_strength=0.01,
            candidate_aspect_ratios=app_settings.standard_aspect_ratios,
        )


class RefinementStrategyHoughStripsRect(RefinementStrategy):
    @property
    def name(self):
        return "Hough transform (with indep rectanglification)"

    def refine(
        self,
        image: Image.Image | UInt8Array,
        corner_points: QuadArray,
        reltol: float = 0.1,
        debug_dir: str | None = None,
    ):
        print("refining with Hough and debug dir", debug_dir)
        refined_points = refine_strips_hough.refine_strips_hough(
            image,
            corner_points,
            debug_dir=debug_dir,
            reltol=reltol,
            soft_boundaries=True,
            max_candidate_angles=2,
            max_candidate_intercepts_per_angle=2,
            include_single_strip_angle_hypotheses=True,
            aspect_preference_strength=0.01,
            candidate_aspect_ratios=app_settings.standard_aspect_ratios,
        )
        refined_points = refine_strips.refine_bounding_box_strips(
            image,
            refined_points,
            minimum_strip_size_pixels=10,
            reltol=0.005,
            enforce_parallel_sides=False,
        )
        return refined_points
        from core import inscribed_rectangle

        return inscribed_rectangle.largest_inscribed_rectangle(refined_points)


class RefinementStrategyHoughMultiscale(RefinementStrategy):
    @property
    def name(self):
        return "Hough transform (multiscale)"

    def refine(
        self,
        image: Image.Image | UInt8Array,
        corner_points: QuadArray,
        reltol: float = 0.1,
        debug_dir: str | None = None,
    ):
        result1 = refine_strips_hough.refine_strips_hough(
            image,
            corner_points,
            debug_dir=debug_dir,
            reltol=reltol,
            soft_boundaries=True,
            max_candidate_angles=2,
            max_candidate_intercepts_per_angle=2,
            include_single_strip_angle_hypotheses=True,
            aspect_preference_strength=0.04,
            candidate_aspect_ratios=app_settings.standard_aspect_ratios,
        )
        return refine_strips_hough.refine_strips_hough(
            image,
            result1,
            debug_dir=debug_dir,
            reltol=reltol / 3.0,
            soft_boundaries=True,
            max_candidate_angles=2,
            max_candidate_intercepts_per_angle=1,
            include_single_strip_angle_hypotheses=True,
            aspect_preference_strength=0.0,
            candidate_aspect_ratios=app_settings.standard_aspect_ratios,
        )


class RefinementStrategyHoughGreedy(RefinementStrategy):
    @property
    def name(self):
        return "Hough transform (greedy)"

    def refine(
        self,
        image: Image.Image | UInt8Array,
        corner_points: QuadArray,
        reltol: float = 0.05,
        debug_dir: str | None = None,
    ):
        print("refining with Hough greedy and debug dir", debug_dir)
        return refine_strips_hough.refine_strips_hough(
            image,
            corner_points,
            debug_dir=debug_dir,
            reltol=reltol,
            max_candidate_angles=1,
            max_candidate_intercepts_per_angle=1,
            aspect_preference_strength=0.05,
            candidate_aspect_ratios=app_settings.standard_aspect_ratios,
        )


_REFINEMENT_STRATEGIES: list[RefinementStrategy] = [
    RefinementStrategyHoughSoft(),
    RefinementStrategyHoughMultiscale(),
    RefinementStrategyStrips(),
    RefinementStrategyStripsIndependent(),
]

REFINEMENT_STRATEGIES: dict[str, RefinementStrategy] = {
    s.name: s for s in _REFINEMENT_STRATEGIES
}


def configure_refinement_strategy(settings: AppSettings) -> RefinementStrategy:
    strategy_name = settings.refinement_strategy
    return REFINEMENT_STRATEGIES.get(strategy_name, _REFINEMENT_STRATEGIES[0])
