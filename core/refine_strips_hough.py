# Suppress lint errors from uppercase variable names
# ruff: noqa N806, N803

from __future__ import annotations

import functools
import logging
import os
import pathlib
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Generic, Sequence, TypeVar

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

from core import debug_plots, geometry, images, refine_strips
from core.photo_types import (
    FloatArray,
    IntArray,
    QuadArray,
    UInt8Array,
    bounding_box_as_array,
)

LOGGER = logging.getLogger("logger")

# Algorithm constants
DEFAULT_RESOLUTION_SCALE_FACTOR = 1.0
MIN_IMAGE_PIXELS_DEFAULT = 8
IMAGE_BOUNDS_SLACK = 0.01
EDGE_WEIGHT_BLUR_KERNEL_SIZE = 5
EDGE_PIXELS_TO_ZERO = 2  # Zero out edge pixels where gradients aren't well-defined
FFT_MAX_RADIUS_FRACTION = 0.8  # Don't sample all the way to FFT edges
FFT_RADIAL_SAMPLES = 50
FFT_ANGLE_RANGE_FRACTION = 1.0 / 8.0  # +/- pi/8 around central angle
PEAK_PROMINENCE_FRACTION = 1.0 / 10.0  # Peaks must be 1/10 of max height
IMAGE_EDGE_BOOST_FRACTION = 0.15  # Boost for edges at image boundary
SHRINK_INWARDS_PIXELS = 5.0  # Pixels to shrink edge inwards
MIN_SLOPE_RESOLUTION = 0.0025
UNIQUE_ANGLE_TOLERANCE = np.pi / (180.0 * 40.0)  # When to combine angle hypotheses.
ASPECT_TOLERANCE = 0.025  # Relative tolerance for aspect ratio matching
INTERPOLATION_ORDER = 0  # Order for ndimage.map_coordinates
EPS_VERTICAL_LINE = 1e-10  # Threshold for detecting vertical lines
EPS_ANGLE_CORRECTION = 1e-4  # Threshold for angle correction


class StripPosition(Enum):
    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"

    @property
    def is_horizontal(self):
        return self.value in ("top", "bottom")


@dataclass(frozen=True)
class Strip:
    """Immutable basic strip information."""

    position: StripPosition
    pixels: UInt8Array
    image_to_strip_transform: Callable[[FloatArray], FloatArray]
    strip_to_image_transform: Callable[[FloatArray], FloatArray]
    edge_weights: FloatArray
    mask: UInt8Array
    image_corners_in_strip_coords: QuadArray


@dataclass
class RectangleBounds:
    top_y: float
    bottom_y: float
    left_x: float
    right_x: float

    def to_corners_array(self) -> QuadArray:
        """Returns rectangle corners, ordered clockwise from top right."""
        return np.array(
            [
                [self.left_x, self.top_y],
                [self.right_x, self.top_y],
                [self.right_x, self.bottom_y],
                [self.left_x, self.bottom_y],
            ]
        )

    def __iter__(self):
        yield from (self.top_y, self.bottom_y, self.left_x, self.right_x)


@dataclass(frozen=True)
class AngleScores:
    """Immutable angle scoring results."""

    angles: FloatArray
    scores: FloatArray

    def __iter__(self):
        yield from (self.angles, self.scores)


@dataclass(frozen=True)
class CandidateEdges:
    """Immutable candidate edges for all angles."""

    angle_to_intercepts: dict[float, Sequence[np.floating[Any]]]
    angle_to_intercept_scores: dict[float, Sequence[np.floating[Any]]]
    angle_to_edges: dict[float, Sequence[FloatArray]]


T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


@dataclass(frozen=True)
class StripSet(Generic[T]):
    """Generic container for top/bottom/left/right elements."""

    top: T
    bottom: T
    left: T
    right: T

    def __iter__(self):
        yield from (self.top, self.bottom, self.left, self.right)

    def to_dict(self) -> dict[StripPosition, T]:
        """Convert to dictionary for compatibility with existing code."""
        return {
            StripPosition.TOP: self.top,
            StripPosition.BOTTOM: self.bottom,
            StripPosition.LEFT: self.left,
            StripPosition.RIGHT: self.right,
        }


STRIP_POSITIONS = StripSet(
    top=StripPosition.TOP,
    bottom=StripPosition.BOTTOM,
    left=StripPosition.LEFT,
    right=StripPosition.RIGHT,
)


def map_strip_set(f: Callable[[T], U], x: StripSet[T]) -> StripSet[U]:
    return StripSet(*map(f, x))


def map2_strip_set(
    f: Callable[[T, V], U], x: StripSet[T], y: StripSet[V]
) -> StripSet[U]:
    return StripSet(*map(f, x, y))


def snap_to(x, candidates):
    """Snap a value to the nearest candidate from a list.

    Args:
        x: Value to snap
        candidates: List of candidate values

    Returns:
        The candidate value closest to x
    """
    return min(candidates, key=lambda c: abs(x - c))


def get_default_normalized_bounds_for_strip(
    position: StripPosition, reltol_x: float, reltol_y: float
) -> RectangleBounds:
    if position == StripPosition.TOP:
        return RectangleBounds(
            top_y=-reltol_y, bottom_y=reltol_y, left_x=-reltol_x, right_x=1 + reltol_x
        )
    elif position == StripPosition.BOTTOM:
        return RectangleBounds(
            top_y=1 - reltol_y,
            bottom_y=1 + reltol_y,
            left_x=-reltol_x,
            right_x=1 + reltol_x,
        )
    elif position == StripPosition.LEFT:
        return RectangleBounds(
            top_y=-reltol_y, bottom_y=1 + reltol_y, left_x=-reltol_x, right_x=reltol_x
        )
    elif position == StripPosition.RIGHT:
        return RectangleBounds(
            top_y=-reltol_y,
            bottom_y=1 + reltol_y,
            left_x=1 - reltol_x,
            right_x=1 + reltol_x,
        )


def shrink_normalized_strip_bounds_to_image(
    bounds: RectangleBounds,
    image_rect_in_normalized_coords: FloatArray,
    slack=IMAGE_BOUNDS_SLACK,
) -> RectangleBounds:
    top_y, bottom_y, left_x, right_x = bounds
    (img_top_left, img_top_right, img_bottom_right, img_bottom_left) = (
        image_rect_in_normalized_coords
    )
    xs = np.array([left_x, right_x])
    ys = np.array([top_y, bottom_y])

    # For each of the four bounds, find where the relevant border of the image
    # intersects the strip. For example, for the top bound, we find where the
    # image top border crosses the strip at its left and right bounds.
    # If *both* of these points are inside the strip (both intercepts are positive),
    # then shrink the bound inwards until it contacts the first one (optionally
    # minus some slack).
    top_edge_slope, top_edge_icept, top_edge_is_vertical = line_from_points(
        y1=img_top_left[1], x1=img_top_left[0], y2=img_top_right[1], x2=img_top_right[0]
    )
    if not top_edge_is_vertical:
        y_intercepts_at_bounds = top_edge_icept + xs * top_edge_slope
        top_y = max(top_y, np.min(y_intercepts_at_bounds) - slack)

    # Bottom bound
    bottom_edge_slope, bottom_edge_icept, bottom_edge_is_vertical = line_from_points(
        y1=img_bottom_left[1],
        x1=img_bottom_left[0],
        y2=img_bottom_right[1],
        x2=img_bottom_right[0],
    )
    if not bottom_edge_is_vertical:
        y_intercepts_at_bounds = bottom_edge_icept + xs * bottom_edge_slope
        bottom_y = min(bottom_y, np.max(y_intercepts_at_bounds) + slack)

    # Left bound
    (
        left_edge_transpose_slope,
        left_edge_transpose_icept,
        left_edge_transpose_is_vertical,
    ) = line_from_points(
        y1=img_top_left[0],
        x1=img_top_left[1],
        y2=img_bottom_left[0],
        x2=img_bottom_left[1],
    )
    if not left_edge_transpose_is_vertical:
        x_intercepts_at_bounds = (
            left_edge_transpose_icept + ys * left_edge_transpose_slope
        )
        left_x = max(left_x, np.min(x_intercepts_at_bounds) - slack)

    # Right bound
    (
        right_edge_transpose_slope,
        right_edge_transpose_icept,
        right_edge_transpose_is_vertical,
    ) = line_from_points(
        y1=img_top_right[0],
        x1=img_top_right[1],
        y2=img_bottom_right[0],
        x2=img_bottom_right[1],
    )
    if not right_edge_transpose_is_vertical:
        x_intercepts_at_bounds = (
            right_edge_transpose_icept + ys * right_edge_transpose_slope
        )
        right_x = min(right_x, np.max(x_intercepts_at_bounds) + slack)

    if left_x >= right_x or top_y >= bottom_y:
        raise ValueError("Strip does not contact the image!")
    return RectangleBounds(
        top_y=float(top_y),
        bottom_y=float(bottom_y),
        left_x=float(left_x),
        right_x=float(right_x),
    )


def calculate_strip_bounds_unit_square(
    position: StripPosition,
    image_rect_in_normalized_coords: QuadArray,
    reltol_x: float,
    reltol_y: float,
    candidate_aspect_ratios: Sequence[float] | None,
) -> RectangleBounds:
    del candidate_aspect_ratios  # Currently unused.

    initial_bounds = get_default_normalized_bounds_for_strip(
        position, reltol_x, reltol_y
    )
    bounds = shrink_normalized_strip_bounds_to_image(
        initial_bounds, image_rect_in_normalized_coords=image_rect_in_normalized_coords
    )
    return bounds


def create_strip_coordinate_transforms(
    strip_corners_image: QuadArray,
    strip_width: int,
    strip_height: int,
) -> tuple[Callable[[FloatArray], FloatArray], Callable[[FloatArray], FloatArray]]:
    """Create coordinate transformation functions for a strip.

    This function creates a pair of transformation functions that convert
    between image coordinates and strip-local pixel coordinates. The strip
    coordinates range from (0,0) to (strip_width, strip_height).

    Args:
        strip_corners_image: Corner coordinates of the strip in image space
        strip_width: Width of the strip in pixels
        strip_height: Height of the strip in pixels

    Returns:
        Tuple of (image_to_strip_transform, strip_to_image_transform) where:
        - image_to_strip_transform: Function to convert image coords to strip coords
        - strip_to_image_transform: Function to convert strip coords to image coords
    """
    strip_converter = geometry.PatchCoordinatesConverter(strip_corners_image)

    def image_to_strip_transform(pts: FloatArray) -> FloatArray:
        return strip_converter.image_to_unit_square(pts) * np.array(
            [strip_width, strip_height]
        )

    def strip_to_image_transform(pts: FloatArray) -> FloatArray:
        return strip_converter.unit_square_to_image(
            pts / np.array([strip_width, strip_height])
        )

    return image_to_strip_transform, strip_to_image_transform


def compute_fft_spectrum(image):
    """Compute 2D FFT magnitude spectrum for angle detection.

    This function applies a Hanning window to reduce edge effects,
    then computes the 2D FFT and returns the magnitude spectrum.
    The FFT is used to detect dominant angles in the edge image.

    Args:
        image: 2D array of edge weights

    Returns:
        Tuple of (fft_shifted, magnitude) where:
        - fft_shifted: Complex FFT coefficients, shifted so DC is at center
        - magnitude: Magnitude spectrum of the FFT
    """
    """Compute 2D FFT and return magnitude spectrum"""

    # Shrinking the image gives a big speed improvement but small accuracy drop.
    # TODO: correct for the occasional slight shift in aspect ratio from shrinking.
    # height, width = image.shape
    # new_height = int(height / 2)
    # new_width = int(width / 2)
    # image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Apply window to reduce edge effects
    window = np.outer(np.hanning(image.shape[0]), np.hanning(image.shape[1]))
    windowed = image * window

    # Compute 2D FFT
    fft = np.fft.fft2(windowed)
    fft_shifted = np.fft.fftshift(fft)

    # Get magnitude spectrum (log scale for better visualization)
    magnitude = np.abs(fft_shifted)

    return fft_shifted, magnitude


def calculate_angle_range_for_radial_profile(
    central_angle: float, height: int, width: int, transpose_coordinates: bool = False
) -> tuple[FloatArray, FloatArray]:
    """Calculate the angle range for radial profile sampling.

    This function determines the range of angles to sample around a central
    angle, with aspect ratio corrections applied.

    Args:
        central_angle: The aspect-corrected central angle
        height: Image height
        width: Image width
        transpose_coordinates: whether to pretend we are working with the
          transposed (height/width swapped) aspect ratio.
    Returns:
        Tuple of (angles, aspect_corrected_angles) where:
        - angles: Raw angles for sampling
        - aspect_corrected_angles: Angles corrected for aspect ratio
    """
    # Convert central angle to perpendicular for sampling range calculation
    central_angle_perpendicular = (central_angle + np.pi / 2) % np.pi

    if transpose_coordinates:
        width, height = height, width

    # Generate a core set of slopes that are sufficient to resolve changes
    # of one pixel at the far edges of the image. (Note that the actual edge
    # would range from -height/2 to height/2; we double this to cover some lines that
    # don't quite reach the edge).
    slopes = np.linspace(-height, height, height * 2 + 1) / (width / 2)

    # Next expand the slopes to cover the full range [-max_slope, max_slope] (if
    # they don't already). These are sampled at a fixed resolution
    # (MIN_SLOPE_RESOLUTION) which is generally coarser than the per-pixel resolution
    # of the core slopes.
    max_slope = 0.4
    slope_gap = max_slope - np.max(slopes)
    if slope_gap > 0:
        num_additional_slopes = int(slope_gap // MIN_SLOPE_RESOLUTION)
        additional_slopes_neg = np.linspace(
            -max_slope, np.min(slopes), num_additional_slopes, endpoint=False
        )
        slopes = np.concatenate(
            [additional_slopes_neg, slopes, -additional_slopes_neg[::-1]],
            axis=0,
        )

    # Finally convert the slopes into angles in the image and FFT, correcting
    # for the non-square aspect ratio.
    image_angles = np.arctan(slopes) + central_angle_perpendicular
    fft_angles = np.arctan((width / height) * slopes) + central_angle
    return fft_angles, image_angles


def sample_radial_direction(
    magnitude: FloatArray,
    angle: float,
    center_x: int,
    center_y: int,
    height: int,
    width: int,
    max_radius: float = FFT_MAX_RADIUS_FRACTION,
    num_samples: int = FFT_RADIAL_SAMPLES,
) -> float:
    """Sample magnitude along a radial direction.

    This function samples the FFT magnitude along a radial line from the
    center outward at a given angle. It handles boundary checking to ensure
    samples stay within the image bounds.

    Args:
        magnitude: FFT magnitude array
        angle: Angle to sample along
        center_x: Center x coordinate
        center_y: Center y coordinate
        height: Image height
        width: Image width
        max_radius: Maximum radius to sample (as fraction of image size)
        num_samples: Number of samples along the radius

    Returns:
        Mean magnitude along the radial direction
    """
    # In normalized coordinates (where both dimensions go from -1 to 1)
    # we need to account for the aspect ratio
    rs = np.linspace(0.1, max_radius, num_samples)

    # Standard polar coordinates
    dx = np.cos(angle)
    dy = np.sin(angle)

    # Scale to actual image coordinates
    xs = center_x + rs * dx * min(center_x, center_y)
    ys = center_y + rs * dy * min(center_x, center_y)

    # Ensure we stay within bounds
    mask = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    xs, ys = xs[mask], ys[mask]

    if len(xs) > 0:
        samples = ndimage.map_coordinates(
            magnitude, [ys, xs], order=INTERPOLATION_ORDER
        )
        return float(np.mean(samples))
    else:
        return 0.0


def radial_profile_corrected(magnitude, central_angle, transpose_coordinates=False):
    """Compute radial profile with aspect ratio correction.

    This function samples the FFT magnitude along radial directions,
    with aspect ratio corrections applied to account for non-square images.
    The sampling is focused around a central angle with a limited range.

    Args:
        magnitude: 2D FFT magnitude array
        central_angle: The aspect-corrected angle around which to focus sampling
        transpose_coordinates: whether to treat `magnitude` as `magnitude.T` in
          defining angles.

    Returns:
        Tuple of (aspect_corrected_angles, radial_profile) where:
        - aspect_corrected_angles: Angles corrected for aspect ratio
        - radial_profile: Magnitude values sampled along each angle
    """
    height, width = magnitude.shape
    center_y, center_x = height // 2, width // 2

    # Calculate angle range and aspect corrections
    angles, aspect_corrected_angle = calculate_angle_range_for_radial_profile(
        central_angle, height, width, transpose_coordinates=transpose_coordinates
    )

    # Sample radial profile for each angle
    profile = np.zeros_like(angles)
    for i, angle in enumerate(angles):
        profile[i] = sample_radial_direction(
            magnitude, angle, center_x, center_y, height, width
        )

    return aspect_corrected_angle, profile


def add_sampled_functions(xs, ys, kind="cubic"):
    """Add multiple functions sampled at different x-coordinates.

    This function is used to combine angle scores from different strips,
    which may have been sampled at different angle values. It interpolates
    all functions to a common set of x-coordinates and sums them.

    Args:
        xs: List of x-coordinate arrays, one per function
        ys: List of y-coordinate arrays, one per function
        kind: Interpolation method ('linear', 'cubic', 'quadratic', etc.)

    Returns:
        Tuple of (x_union, y_sum) where:
        - x_union: Combined x-coordinates where all functions are evaluated
        - y_sum: Sum of all functions at the union points
    """
    """
    Add two functions sampled at different x-coordinates.

    Parameters:
    -----------
    x1, y1 : array_like
        First function's sample points
    x2, y2 : array_like
        Second function's sample points
    kind : str
        Interpolation method ('linear', 'cubic', 'quadratic', etc.)

    Returns:
    --------
    x_union : ndarray
        Union of all x-coordinates where sum is evaluated
    y_sum : ndarray
        Sum of the two functions at x_union points
    """

    # Create union of x-coordinates and sort
    x_union = np.union1d(xs[0], xs[1])
    for i in range(2, len(xs)):
        x_union = np.union1d(x_union, xs[i])

    # Determine interpolation bounds and
    # only evaluate sum where both functions can be interpolated
    x_valid = x_union
    for i in range(len(xs)):
        x_bounds_min, x_bounds_max = np.min(xs[i]), np.max(xs[i])
        x_valid = x_valid[(x_valid >= x_bounds_min) & (x_valid <= x_bounds_max)]

    y_interps = []
    # Interpolate
    for i in range(len(xs)):
        f_interp = interp1d(xs[i], ys[i], kind=kind, bounds_error=False, fill_value=0)
        y_interps.append(f_interp(x_valid))

    # Sum the functions
    y_sum = np.sum(y_interps, axis=0)

    return x_valid, y_sum


def get_sorted_peak_indices(scores: FloatArray, max_num_peaks=2) -> IntArray:
    """Find the most prominent peaks in a score array.

    This function uses scipy.signal.find_peaks to identify local maxima
    with sufficient prominence, then sorts them by prominence and returns
    the top peaks.

    Args:
        scores: Array of scores to find peaks in
        max_num_peaks: Maximum number of peaks to return

    Returns:
        Array of indices of the most prominent peaks, sorted by prominence
        (highest prominence first)
    """
    peaks, extra = find_peaks(
        scores, prominence=np.max(scores) * PEAK_PROMINENCE_FRACTION
    )
    if len(peaks) == 0:
        return np.array([np.argmax(scores)])
    else:
        perm = np.argsort(extra["prominences"])
        # Return peaks sorted by score in descending order.
        return np.array(peaks)[perm][: -max_num_peaks - 1 : -1]


def score_angles_in_strip(strip: Strip, debug_dir: str | None = None) -> AngleScores:
    """Score angles in a strip using FFT-based analysis.

    This function uses the 2D FFT of the edge weights to detect dominant
    angles in the strip. The FFT magnitude is sampled radially at different
    angles, with aspect ratio corrections applied.

    The central angle for sampling depends on strip orientation:
    - Horizontal strips: π/2 (vertical sampling)
    - Vertical strips: 0 (horizontal sampling)

    Args:
        strip: The strip with edge detection results
        debug_dir: Optional directory for saving debug plots

    Returns:
        Tuple of (angles, angle_scores) where both are sorted by angle.
        Angles are in image coordinates (not strip coordinates).
    """
    _, magnitude = compute_fft_spectrum(strip.edge_weights)
    central_angle = np.pi / 2 if strip.position.is_horizontal else 0.0
    strip_angles, profile = radial_profile_corrected(
        magnitude,
        central_angle=central_angle,
        transpose_coordinates=not strip.position.is_horizontal,
    )
    image_angles = (strip_angles - central_angle) % np.pi - np.pi / 2

    perm = np.argsort(image_angles)
    angles = image_angles[perm]
    angle_scores = profile[perm]

    if debug_dir is not None:
        log_magnitude = np.log(1 + magnitude)
        normed_log_magnitude = log_magnitude - np.min(log_magnitude)
        normed_log_magnitude = (
            normed_log_magnitude * 255.0 / np.max(normed_log_magnitude)
        )
        debug_plots.save_image(
            os.path.join(
                debug_dir,
                f"log_fft_magnitude_{strip.position.value}.png",
            ),
            normed_log_magnitude,
        )
        debug_plots.save_plot(
            os.path.join(debug_dir, f"angles_fft_{strip.position.value}.png"),
            np.degrees(angles),
            angle_scores,
            f"{strip.position} image angles max deg {np.degrees(angles[np.argmax(profile)])}",
        )
    return AngleScores(angles, angle_scores)


def line_from_points(y1: float, x1: float, y2: float, x2: float):
    """Convert two points to line equation y = mx + b.

    Args:
        y1, x1: First point coordinates
        y2, x2: Second point coordinates

    Returns:
        Tuple of (slope, intercept, is_vertical) where:
        - slope: The slope m, or inf for vertical lines
        - intercept: The y-intercept b, or x-coordinate for vertical lines
        - is_vertical: True if the line is vertical
    """
    """
    Convert two points to line equation y = mx + b.

    Returns:
        (slope, intercept, is_vertical)
    """
    if abs(x2 - x1) < EPS_VERTICAL_LINE:  # Vertical line
        return float("inf"), x1, True

    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept, False


def intercept_for_line_bounded_by_edge(
    edge_pt1,
    edge_pt2,
    border_slope,
    strip_x_min,
    strip_x_max,
    bounded_above=True,
    shrink_inwards_by=SHRINK_INWARDS_PIXELS,
):
    """Calculate intercept for a line bounded by an image edge.

    This function determines where a line with the given slope would intersect
    a strip if it were positioned at the image boundary. It accounts for the
    direction of the boundary and shrinks the line inwards slightly.

    Args:
        edge_pt1, edge_pt2: Two points defining the image boundary
        border_slope: Slope of the line to position at the boundary
        x_min, x_max: The x-coordinate range in strip coordinates
        bounded_above: Whether the line is bounded from above or below
        shrink_inwards_by: Pixels to shrink the line inwards from the boundary

    Returns:
        The intercept value where the line should be positioned,
        or None if the edge is vertical.
    """
    x1, y1 = edge_pt1
    x2, y2 = edge_pt2
    m1, b1, edge_is_vertical = line_from_points(y1, x1, y2, x2)

    if edge_is_vertical:
        return None

    relative_slope = m1 - border_slope
    cond_sign = 1 if bounded_above else -1
    if cond_sign * relative_slope <= 0:
        return relative_slope * strip_x_min + b1 + cond_sign * shrink_inwards_by
    else:
        return relative_slope * strip_x_max + b1 + cond_sign * shrink_inwards_by


def bincount_histogram(xs, bin_min, bin_max, weights):
    """Create a histogram using numpy.bincount for better performance.

    This function is optimized for the common case where bin centers are
    integers, avoiding the overhead of numpy.histogram.

    Args:
        xs: Values to histogram
        bin_min: Minimum bin value
        bin_max: Maximum bin value
        weights: Weights for each value in xs

    Returns:
        Tuple of (histogram, bin_centers) where:
        - histogram: Count/weight in each bin
        - bin_centers: Integer bin center values
    """
    bins = np.arange(bin_min, bin_max + 1)
    # Bin xs to integer indices.
    x_indices = np.floor(xs - bin_min).astype(int)
    # Create a zeroth bin for outliers.
    x_indices = np.where((x_indices < 0) | (x_indices >= bin_max), 0, x_indices + 1)
    hist: FloatArray = np.bincount(x_indices, weights=weights, minlength=len(bins))[1:]  # type: ignore
    return hist, bins


def score_intercepts_for_strip_functional(
    strip: Strip, slope: float
) -> tuple[FloatArray, FloatArray]:
    """Score all possible intercepts for a given slope using edge voting.

    This implements a Hough-like voting scheme where each pixel with high
    edge weight votes for the intercept of the line passing through it
    at the given slope. The result is a histogram of intercept scores.

    Args:
        strip_with_edges: The strip with edge detection results
        slope: The slope of lines to test

    Returns:
        Tuple of (intercept_scores, intercept_bins) where:
        - intercept_scores: Histogram of edge vote scores for each intercept
        - intercept_bins: Bin center values for the histogram
    """
    weights = strip.edge_weights
    height, width = strip.pixels.shape[:2]
    if strip.position.is_horizontal:
        x_coords = np.arange(width) - (width // 2) + 0.5
        y_coords = np.arange(height)[:, None] + 0.5
        intercepts = y_coords - slope * x_coords
        min_intercept = -np.abs(slope) * width // 2
        max_intercept = (height - 1) + np.abs(slope) * (width // 2)
    else:
        y_coords = np.arange(height)[:, None] - (height // 2) + 0.5
        x_coords = np.arange(width) + 0.5
        intercepts = x_coords + slope * y_coords
        min_intercept = -np.abs(slope) * height // 2
        max_intercept = (width - 1) + np.abs(slope) * height // 2

    intercept_scores, intercept_bins = bincount_histogram(
        intercepts.flatten(),
        int(np.floor(min_intercept)),
        int(np.ceil(max_intercept)),
        weights=weights.flatten(),
    )
    print(
        f"strip {strip.position.value} shape {strip.pixels.shape[:2]} min intercept {intercept_bins[0]} max intercept {intercept_bins[-1]} bins {len(intercept_bins)}"
    )
    return intercept_scores, intercept_bins


def intercept_to_image_points_functional(
    strip: Strip, slope: float, intercept: float
) -> FloatArray:
    """Convert a line (slope, intercept) to two points in image coordinates.

    Args:
        strip: The strip data containing coordinate transformations
        slope: The slope of the line in strip coordinates
        intercept: The intercept of the line in strip coordinates

    Returns:
        Array of shape (2, 2) containing two points that define the line
        in image coordinates: [[x1, y1], [x2, y2]]
    """
    height, width = strip.pixels.shape[:2]
    if strip.position.is_horizontal:
        pt1 = np.array([0, intercept - slope * width / 2])
        pt2 = np.array([width - 1, intercept + slope * width / 2])
    else:
        pt1 = np.array([intercept + slope * height / 2, 0])
        pt2 = np.array([intercept - slope * height / 2, height - 1])
    pt1_image = strip.strip_to_image_transform(pt1.reshape((1, -1)))
    pt2_image = strip.strip_to_image_transform(pt2.reshape((1, -1)))
    return np.concatenate([pt1_image, pt2_image], axis=0)


def score_aspect_ratio(
    corners: QuadArray,
    candidate_aspect_ratios,
    aspect_preference_strength=1.0,
    aspect_rtol=ASPECT_TOLERANCE,
):
    """Score a set of corners based on how well they match expected aspect ratios.

    This function computes the aspect ratio of the given corners and provides
    a bonus score if it closely matches any of the candidate aspect ratios.
    The scoring uses a quadratic penalty function that peaks at exact matches
    and drops to zero at the tolerance boundary.

    Args:
        corners: The four corner points of the detected rectangle
        candidate_aspect_ratios: List of expected aspect ratios (e.g., [1.4, 1.5])
        aspect_preference_strength: Multiplier for the aspect ratio bonus
        aspect_rtol: Relative tolerance for aspect ratio matching

    Returns:
        Bonus score to add to the edge detection score. Higher values indicate
        better matches to the expected aspect ratios.
    """
    width, height = geometry.dimension_bounds(corners)
    aspect = max(width, height) / min(width, height)
    print(f"scoring aspect {aspect: .3f}")
    score = 0.0
    for candidate_aspect in candidate_aspect_ratios:
        relative_aspect = aspect / candidate_aspect
        aspect_error = abs(relative_aspect - 1.0)
        if aspect_error > aspect_rtol:
            continue
        score += aspect_preference_strength * (1 - (aspect_error / aspect_rtol) ** 2)
        print(f"   matches candidate {candidate_aspect: .3f}! score boost {score}")
    return score


def rescore_rects_with_edge_integral(
    strips: StripSet[Strip], corner_points: FloatArray
):
    """Evaluates a list of candidate rectangles by total edge weight under their edges.

    This should be similar to the sum of their intercept scores (which also sum edge
    weights), but may be slightly more informative because
    a) while the intercept scores integrated across the entire line, here we have
       corner points, so we can integrate only over the line *segment* up to the
       relevant corner, which is ultimately what matters for the rectangle.
    b) ....

    Args:
      strips: set of edge strips with edge weights to integrate over
      corner_points: array of size [num_candidates, 4, 2] containing candidate rectangle
      corners [x, y] in image coordinates in clockwise
      `[top_left, top_right, bottom_right, bottom_left]` order.

    Returns:
      integrals: array of size `[num_candidates, 4]` containing line integrals
        over the relevant edge weights for the top, bottom, left, right edges
        respectively.
    """
    top_edge_scores = geometry.line_integral_chunked(
        image=strips.top.edge_weights,
        start_points=strips.top.image_to_strip_transform(corner_points[:, 0, :]),
        end_points=strips.top.image_to_strip_transform(corner_points[:, 1, :]),
    )
    right_edge_scores = geometry.line_integral_chunked(
        image=strips.right.edge_weights,
        start_points=strips.right.image_to_strip_transform(corner_points[:, 1, :]),
        end_points=strips.right.image_to_strip_transform(corner_points[:, 2, :]),
    )
    bottom_edge_scores = geometry.line_integral_chunked(
        image=strips.bottom.edge_weights,
        start_points=strips.bottom.image_to_strip_transform(corner_points[:, 2, :]),
        end_points=strips.bottom.image_to_strip_transform(corner_points[:, 3, :]),
    )
    left_edge_scores = geometry.line_integral_chunked(
        image=strips.left.edge_weights,
        start_points=strips.left.image_to_strip_transform(corner_points[:, 3, :]),
        end_points=strips.left.image_to_strip_transform(corner_points[:, 0, :]),
    )
    return np.stack(
        [top_edge_scores, bottom_edge_scores, left_edge_scores, right_edge_scores],
        axis=-1,
    )


def get_soft_edge_prior(
    initial_rect_in_patch_coords: QuadArray, position: StripPosition, patch_shape
):
    height, width = patch_shape
    left_x, top_y = initial_rect_in_patch_coords[0, :]  # top left corner
    right_x, bottom_y = initial_rect_in_patch_coords[2, :]  # bottom right corner
    if position == StripPosition.TOP:
        abs_offsets = np.abs(np.arange(height) - top_y)[..., np.newaxis]
    elif position == StripPosition.BOTTOM:
        abs_offsets = np.abs(np.arange(height) - bottom_y)[..., np.newaxis]
    elif position == StripPosition.LEFT:
        abs_offsets = np.abs(np.arange(width) - left_x)[np.newaxis, :]
    elif position == StripPosition.RIGHT:
        abs_offsets = np.abs(np.arange(width) - right_x)[np.newaxis, :]
    # Weight the edges by a Gaussian prior.
    stddev = np.max(abs_offsets) / 2.0
    weights = np.exp(-0.5 * (abs_offsets / stddev) ** 2)
    return weights


def maybe_boost_image_edge(
    strip: Strip,
    candidate_angle: float,
    intercept_bins: FloatArray,
    intercept_scores: FloatArray,
):
    (top_left, top_right, bottom_right, bottom_left) = (
        strip.image_corners_in_strip_coords
    )
    slope = np.tan(candidate_angle)
    height, width = strip.pixels.shape[:2]
    if not strip.position.is_horizontal:
        width, height = height, width
    x_min = -width // 2
    x_max = x_min + width - 1

    if strip.position == StripPosition.TOP:
        intercept = intercept_for_line_bounded_by_edge(
            top_left,
            top_right,
            border_slope=slope,
            strip_x_min=0,
            strip_x_max=width,
            bounded_above=True,
            shrink_inwards_by=SHRINK_INWARDS_PIXELS,
        )
    elif strip.position == StripPosition.BOTTOM:
        intercept = intercept_for_line_bounded_by_edge(
            bottom_left,
            bottom_right,
            border_slope=slope,
            strip_x_min=0,
            strip_x_max=width,
            bounded_above=False,
            shrink_inwards_by=SHRINK_INWARDS_PIXELS,
        )
    elif strip.position == StripPosition.LEFT:
        intercept = intercept_for_line_bounded_by_edge(
            top_left[::-1],
            bottom_left[::-1],
            border_slope=slope,
            strip_x_min=0,
            strip_x_max=height,
            bounded_above=False,
            shrink_inwards_by=-SHRINK_INWARDS_PIXELS,
        )
    elif strip.position == StripPosition.RIGHT:
        intercept = intercept_for_line_bounded_by_edge(
            top_right[::-1],
            bottom_right[::-1],
            border_slope=slope,
            strip_x_min=0,
            strip_x_max=height,
            bounded_above=True,
            shrink_inwards_by=-SHRINK_INWARDS_PIXELS,
        )
    if intercept is None:
        return intercept_scores
    if strip.position.is_horizontal:
        centered_intercept = intercept - slope * x_min
    else:
        centered_intercept = intercept + slope * x_min
    print(
        f"strip {strip.position}: edge intercept orig {intercept} centered {centered_intercept}"
    )
    if centered_intercept >= np.min(intercept_bins) and centered_intercept <= np.max(
        intercept_bins
    ):
        bin = np.searchsorted(intercept_bins, centered_intercept) - 1
        print(
            f"previous scores max {np.max(intercept_scores)} mean {np.mean(intercept_scores)} binval {intercept_scores[bin]}"
        )
        intercept_scores[bin] += IMAGE_EDGE_BOOST_FRACTION * (x_max - x_min)

        print(
            f"boosting bin {bin} icept {intercept_bins[bin]} by {IMAGE_EDGE_BOOST_FRACTION * (x_max - x_min)}, new val {intercept_scores[bin]}"
        )
    return intercept_scores


# ===================================================================
# CORE FUNCTIONS
# ===================================================================


def extract_border_strips(
    image: Image.Image | UInt8Array,
    rect: QuadArray,
    reltol: float,
    soft_boundaries: bool = False,
    resolution_scale_factor: float = DEFAULT_RESOLUTION_SCALE_FACTOR,
    min_image_pixels: int = MIN_IMAGE_PIXELS_DEFAULT,
    candidate_aspect_ratios: list[float] | None = None,
    debug_dir: str | None = None,
) -> StripSet[Strip]:
    """Extract four border strips from around a rectangle in the image.

    This is a functional version that returns a StripSet instead of modifying
    a mutable dictionary.

    Args:
        image: The source image
        rect: Four corners of the rectangle in image coordinates
        reltol: Relative tolerance for strip width (fraction of image size)
        resolution_scale_factor: Scale factor for strip resolution
        min_image_pixels: Minimum strip width in pixels
        candidate_aspect_ratios: Expected aspect ratios for expansion
        debug_dir: Optional directory for saving debug images

    Returns:
        StripSet containing the four extracted strips
    """
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
        del image
    else:
        pil_image = image

    image_corners = np.array(
        [
            [0.0, 0.0],
            [pil_image.width - 1.0, 0.0],
            [pil_image.width - 1.0, pil_image.height - 1.0],
            [0.0, pil_image.height - 1.0],
        ]
    )

    # Ensure rect is sorted clockwise
    rect = geometry.sort_clockwise(rect)
    width, height = geometry.dimension_bounds(rect)

    # if soft_boundaries:
    #    reltol *= 2

    reltol_x = max(reltol, min_image_pixels / width)
    reltol_y = max(reltol, min_image_pixels / height)
    converter = geometry.PatchCoordinatesConverter(rect)

    strip_boundaries_unit_square = map_strip_set(
        functools.partial(
            calculate_strip_bounds_unit_square,
            image_rect_in_normalized_coords=converter.image_to_unit_square(
                image_corners
            ),
            reltol_x=reltol_x,
            reltol_y=reltol_y,
            candidate_aspect_ratios=candidate_aspect_ratios,
        ),
        STRIP_POSITIONS,
    )

    # Convert to image coordinates and extract each strip

    def create_strip(position: StripPosition, unit_square_bounds: RectangleBounds):
        # Convert normalized coords to image coords
        strip_corners_image = converter.unit_square_to_image(
            unit_square_bounds.to_corners_array()
        )
        strip_corners_image = np.round(strip_corners_image)
        width, height = geometry.dimension_bounds(strip_corners_image)
        strip_width = int(np.ceil(width * resolution_scale_factor))
        strip_height = int(np.ceil(height * resolution_scale_factor))

        # Extract the strip
        LOGGER.debug(
            f"extracting strip {position} with corners {strip_corners_image} image width {width} height {height} strip width {strip_width} height {strip_height}"
        )
        strip_pixels = images.extract_perspective_image(
            pil_image,
            strip_corners_image,
            output_width=strip_width,
            output_height=strip_height,
            mode=Image.Resampling.BILINEAR,
        )
        pixels_array = np.array(strip_pixels)

        # Create coordinate converters for this strip
        image_to_strip_transform, strip_to_image_transform = (
            create_strip_coordinate_transforms(
                strip_corners_image, strip_width, strip_height
            )
        )

        image_corners_in_patch_coords = image_to_strip_transform(image_corners)

        strip_mask = geometry.image_boundary_mask(
            patch_shape=pixels_array.shape,
            mask_corners_in_patch_coords=image_corners_in_patch_coords,
            offset=-2,
        ).astype(pixels_array.dtype)
        edge_weights = detect_edges(position, pixels_array, strip_mask)
        if soft_boundaries:
            edge_scaling = get_soft_edge_prior(
                initial_rect_in_patch_coords=image_to_strip_transform(rect),
                position=position,
                patch_shape=edge_weights.shape,
            )
            edge_weights *= edge_scaling

        return Strip(
            position=position,
            pixels=pixels_array,
            image_to_strip_transform=image_to_strip_transform,
            strip_to_image_transform=strip_to_image_transform,
            edge_weights=edge_weights,
            mask=strip_mask,
            image_corners_in_strip_coords=image_corners_in_patch_coords,
        )

    return map2_strip_set(create_strip, STRIP_POSITIONS, strip_boundaries_unit_square)


def get_half_blur_kernel(position: StripPosition, half_field_size=2) -> FloatArray:
    """Build a Gaussian blur kernel that blurs "inwards" within the bounding rect.

    For the top strip, each blurred pixel only depends on pixels above it;
    for the left strip, each bluured pixel depends only pixels to its left, and so on.

    This is because we want to err on the side of finding edges within the image
    bounds. Cropping off a pixel or two of a scanned image is better than having an
    unsightly strip along the edge.
    """
    blur_kernel = cv2.getGaussianKernel(2 * half_field_size + 1, 0)
    blur_kernel = blur_kernel * blur_kernel.T
    if position == StripPosition.TOP:
        blur_kernel[-half_field_size:, :] = 0
    elif position == StripPosition.BOTTOM:
        blur_kernel[:half_field_size, 0] = 0
    elif position == StripPosition.LEFT:
        blur_kernel[:, -half_field_size:] = 0
    elif position == StripPosition.RIGHT:
        blur_kernel[:, :half_field_size] = 0
    blur_kernel /= np.sum(blur_kernel)
    return blur_kernel


def detect_edges(
    position: StripPosition, pixels: FloatArray, mask: UInt8Array
) -> FloatArray:
    """Detect edge weights in a strip using gradient filters.

    This function applies directional gradient filters to detect edges
    perpendicular to the strip orientation. For horizontal strips, it uses
    a vertical gradient filter, and for vertical strips, a horizontal filter.

    Args:
        strip: The strip to process
        image_shape: Shape of the original image for boundary masking

    Returns:
        StripWithEdges containing the strip and edge detection results
    """

    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY).astype(np.float32)

    filter = np.array(
        [[-1, -2, -1], [-1, -2, -1], [1, 2, 1], [1, 2, 1]], dtype=np.float32
    )
    # Apply gradient filter perpendicular to strip orientation
    # Horizontal strips need vertical gradients to detect horizontal edges
    # Vertical strips need horizontal gradients to detect vertical edges
    if position.is_horizontal:
        edge_weights = cv2.filter2D(gray, -1, filter)  # vertical gradient (sobel_y)
    else:
        edge_weights = cv2.filter2D(
            gray, -1, np.array(filter.T)
        )  # horizontal gradient (sobel_x)

    # Apply image boundary mask and post-process edge weights
    edge_weights *= mask  # Zero out pixels outside image boundary
    # Square root: favor long coherent edges over smaller areas with high response.
    edge_weights = np.sqrt(np.abs(edge_weights))
    # Blur pixels "inwards" to favor edges within image bounds.
    half_blur_kernel = get_half_blur_kernel(position, 2)
    edge_weights = cv2.filter2D(edge_weights, -1, half_blur_kernel)
    # edge_weights = cv2.GaussianBlur(
    #    edge_weights, (EDGE_WEIGHT_BLUR_KERNEL_SIZE, EDGE_WEIGHT_BLUR_KERNEL_SIZE), 0
    # )
    edge_weights *= mask  # Re-apply mask after blurring
    # don't use votes from the edge pixels where sobel directions aren't
    # fully defined
    edge_weights[:EDGE_PIXELS_TO_ZERO, :] = 0.0
    edge_weights[-EDGE_PIXELS_TO_ZERO:, :] = 0
    edge_weights[:, :EDGE_PIXELS_TO_ZERO] = 0.0
    edge_weights[:, -EDGE_PIXELS_TO_ZERO:] = 0
    max_weight = np.max(edge_weights)
    if max_weight > 0:
        edge_weights /= max_weight
    else:
        # If no edges detected, use uniform small weights
        edge_weights = np.ones_like(edge_weights) * 0.01

    return edge_weights


def find_best_overall_angles(
    strips: StripSet,
    max_num_peaks: int = 3,
    include_single_strip_hypotheses: bool = False,
    debug_dir: str | None = None,
) -> list[float]:
    """Find the best overall angles by combining evidence from all strips.

    This function implements a voting mechanism where each strip contributes
    its angle preferences from FFT analysis. The scores are normalized so each
    strip has equal maximum influence, then combined using interpolation to
    find the strongest overall angles.

    Args:
        strips_with_edges: StripSetWithEdges containing all strips with edge data
        max_num_peaks: Maximum number of angle peaks to return
        debug_dir: Optional directory for saving debug plots

    Returns:
        List of best angles in radians, sorted by confidence
    """
    # Compute angle scores from each edge strip.
    strip_scored_angles = map_strip_set(
        lambda strip: score_angles_in_strip(strip, debug_dir=debug_dir), strips
    )

    # Sum scores across strips, letting each strip 'vote' for the overall angle. We
    # normalize the scores, so each strip has max score 1, to improve robustness.
    combined_angles, overall_angle_scores = add_sampled_functions(
        [s.angles for s in strip_scored_angles],
        [s.scores / np.max(s.scores) for s in strip_scored_angles],
    )

    best_idxs = get_sorted_peak_indices(
        overall_angle_scores, max_num_peaks=max_num_peaks
    )
    best_angles = [combined_angles[int(idx)] for idx in best_idxs]
    print("best angle", [np.degrees(a) for a in best_angles])
    best_idx = int(best_idxs[0])
    if best_idx > 0:
        best_angles.append(combined_angles[best_idx - 1])
    if best_idx < len(combined_angles) - 1:
        best_angles.append(combined_angles[best_idx + 1])

    print("INcluding single strip?", include_single_strip_hypotheses)
    if include_single_strip_hypotheses:
        for strip_angles, strip_scores in strip_scored_angles:
            strip_best_angle = strip_angles[np.argmax(strip_scores)]
            diffs = [strip_best_angle - a for a in best_angles]
            print(
                f"Considering single-strip angle {strip_best_angle:.4f} to best angles {[best_angles]}"
            )
            if np.min(np.abs(diffs)) > UNIQUE_ANGLE_TOLERANCE:
                print(
                    f"Adding single-strip angle {strip_best_angle:.4f} to best angles {[best_angles]}"
                )
                best_angles.append(strip_best_angle)

    if debug_dir is not None:
        debug_plots.save_plot(
            os.path.join(debug_dir, f"angles_fft_overall.png"),
            np.degrees(combined_angles),
            overall_angle_scores,
            f"overall angle peaks deg {[np.degrees(a) for a in best_angles]}",
        )

    return best_angles


def get_candidate_edges(
    strip: Strip,
    best_angles: list[float],
    max_num_peaks: int = 1,
    debug_dir: str | None = None,
) -> CandidateEdges:
    """Generate candidate edges for a strip at given angles.

    This function orchestrates the edge detection process by:
    1. Scoring all possible intercepts using a Hough-like voting scheme
    2. Finding the most prominent peaks in the intercept scores
    3. Converting intercepts to actual edge lines in image coordinates

    Args:
        strip_with_edges: StripWithEdges containing strip and edge data
        best_angles: List of angles to test for edges
        max_num_peaks: Maximum number of intercept peaks to keep per angle
        debug_dir: Optional directory for saving debug histograms

    Returns:
        StripWithCandidates containing the strip, edge data, and candidate edges
    """
    angle_to_intercepts = {}
    angle_to_intercept_scores = {}
    angle_to_edges = {}

    for candidate_angle in best_angles:
        slope = np.tan(candidate_angle)
        print("candidate angle", candidate_angle, "slope", slope)

        # Score intercepts for this angle
        intercept_scores, intercept_bins = score_intercepts_for_strip_functional(
            strip, slope
        )

        intercept_scores = maybe_boost_image_edge(
            strip=strip,
            candidate_angle=candidate_angle,
            intercept_bins=intercept_bins,
            intercept_scores=intercept_scores,
        )

        # Find peaks in intercept scores
        peak_indices = get_sorted_peak_indices(
            intercept_scores, max_num_peaks=max_num_peaks
        )
        peak_indices = [int(i) for i in peak_indices]

        # Store candidate intercepts and scores
        angle_to_intercepts[candidate_angle] = [intercept_bins[i] for i in peak_indices]
        angle_to_intercept_scores[candidate_angle] = [
            intercept_scores[i] for i in peak_indices
        ]

        # Convert intercepts to edge lines in image coordinates
        angle_to_edges[candidate_angle] = [
            intercept_to_image_points_functional(strip, slope, float(intercept))
            for intercept in angle_to_intercepts[candidate_angle]
        ]

        if debug_dir:
            debug_plots.save_histogram(
                os.path.join(
                    debug_dir,
                    f"intercepts_at_{np.degrees(candidate_angle): .2f}_{strip.position.value}.png",
                ),
                intercept_bins,
                intercept_scores,
                f"{strip.position.value} bins {np.min(intercept_bins)} {np.max(intercept_bins)} intercepts at {np.degrees(candidate_angle): .2f} {angle_to_intercepts[candidate_angle]}",
            )

    return CandidateEdges(
        angle_to_intercepts=angle_to_intercepts,
        angle_to_intercept_scores=angle_to_intercept_scores,
        angle_to_edges=angle_to_edges,
    )


def iterate_over_edge_combinations(candidates: StripSet[CandidateEdges], angle: float):
    for top_idx in range(len(candidates.top.angle_to_intercepts[angle])):
        for bottom_idx in range(len(candidates.bottom.angle_to_intercepts[angle])):
            for left_idx in range(len(candidates.left.angle_to_intercepts[angle])):
                for right_idx in range(
                    len(candidates.right.angle_to_intercepts[angle])
                ):
                    yield StripSet(
                        top=top_idx, bottom=bottom_idx, left=left_idx, right=right_idx
                    )


def find_best_hypothesis(
    strips: StripSet[Strip],
    candidates: StripSet[CandidateEdges],
    candidate_aspect_ratios: list[float] | None = None,
    aspect_preference_strength: float = 0.1,
    debug_dir: str | None = None,
) -> QuadArray:
    """Find the best hypothesis by scoring all combinations of candidate edges.

    This function creates the Cartesian product of all candidate edges,
    where each hypothesis consists of one edge from each of the four strips.
    For each combination, it evaluates both edge strength and aspect ratio preference.

    Args:
        strips_with_candidates: StripSetWithCandidates containing all strips with candidates
        candidate_aspect_ratios: List of expected aspect ratios for scoring
        aspect_preference_strength: Weight for aspect ratio preference

    Returns:
        QuadArray containing the corners of the best hypothesis
    """
    best_score = 0.0
    best_corners = None

    total_border_pixels = sum(
        [
            strip.pixels.shape[1 if strip.position.is_horizontal else 0]
            for strip in strips
        ]
    )

    # Generate all combinations of candidate edges
    angles = candidates.top.angle_to_intercepts.keys()
    for angle in angles:
        for edge_indices in iterate_over_edge_combinations(candidates, angle):
            # Get edges for this combination
            edges = map2_strip_set(
                lambda c, i: c.angle_to_edges[angle][i], candidates, edge_indices
            )

            # Get scores for this combination
            edge_scores = map2_strip_set(
                lambda c, i: c.angle_to_intercept_scores[angle][i],
                candidates,
                edge_indices,
            )

            print(
                f"proposing hypothesis w angle {angle} idx {edge_indices} edge score {sum(edge_scores)}"
            )

            # Find corner intersections
            corners = refine_strips.find_corner_intersections(
                {p.value: e for (p, e) in edges.to_dict().items()}
            )
            edge_score = sum(edge_scores)

            # Add aspect ratio preference if specified
            if candidate_aspect_ratios is not None:
                aspect_score = score_aspect_ratio(
                    corners,
                    candidate_aspect_ratios,
                    aspect_preference_strength=aspect_preference_strength
                    * total_border_pixels,
                )
            else:
                aspect_score = 0.0

            score = edge_score + aspect_score
            if score > best_score:
                best_score = score
                best_corners = corners

                if debug_dir is not None:
                    hypothesis_dir = os.path.join(
                        debug_dir,
                        f"hypothesis_score_{score:.2f}_angle_deg_{np.degrees(angle):.4f}_{edge_indices}",
                    )
                    for strip, edge in zip(strips, edges):
                        strip_edge = strip.image_to_strip_transform(edge)
                        debug_plots.save_image(
                            os.path.join(
                                hypothesis_dir, f"edge_{strip.position.value}.png"
                            ),
                            debug_plots.annotate_image(
                                strip.edge_weights * 255,
                                edges=[np.round(strip_edge).astype(int)],
                            ),
                        )
                        debug_plots.save_image(
                            os.path.join(
                                hypothesis_dir, f"pixels_{strip.position.value}.png"
                            ),
                            debug_plots.annotate_image(
                                strip.pixels,
                                edges=[np.round(strip_edge).astype(int)],
                            ),
                        )

    if best_corners is None:
        raise ValueError("no hypotheses had positive score!!")

    return best_corners


def refine_strips_hough(
    image: Image.Image | np.ndarray,
    corner_points: QuadArray,
    reltol: float = 0.05,
    soft_boundaries: bool = False,
    debug_dir: str | None = None,
    max_candidate_angles: int = 2,
    max_candidate_intercepts_per_angle: int = 2,
    include_single_strip_angle_hypotheses: bool = False,
    aspect_preference_strength: float = 0.1,
    candidate_aspect_ratios: list[float] | None = None,
) -> QuadArray:
    """Refine bounding box corners using functional Hough transform approach.

    This is the main entry point for the functional version of the Hough-based
    refinement algorithm. It implements a clean 5-stage pipeline with immutable
    data structures and pure functions.

    The pipeline stages are:
    1. Extract strips: image → StripSet
    2. Detect edges: StripSet → StripSetWithEdges (map operation)
    3. Find best angles: StripSetWithEdges → list[float] (reduction)
    4. Get candidates: StripSetWithEdges → StripSetWithCandidates (map)
    5. Find best hypothesis: StripSetWithCandidates → QuadArray (reduction)

    Args:
        image: The source image
        corner_points: Initial corners of the bounding box
        reltol: Relative tolerance for strip extraction
        debug_dir: Optional directory for saving debug outputs
        max_candidate_angles: Maximum number of angles to consider
        max_candidate_intercepts_per_angle: Maximum intercepts per angle per strip
        aspect_preference_strength: Weight for aspect ratio preference
        candidate_aspect_ratios: List of expected aspect ratios

    Returns:
        Refined corner points as a QuadArray
    """
    if debug_dir is not None:
        p = pathlib.Path(debug_dir).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        debug_dir = str(p)
        print("made debug dir", debug_dir)
        LOGGER.info(f"logging to debug dir {debug_dir}")

    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
        del image
    else:
        pil_image = image

    corner_points = bounding_box_as_array(corner_points)

    # Get minimum bounding rectangle
    rect, _ = geometry.minimum_bounding_rectangle(corner_points)
    rect = geometry.sort_clockwise(rect)

    # Stage 1: Extract strips (image → StripSet)
    strips: StripSet[Strip] = extract_border_strips(
        pil_image,
        rect,
        reltol=reltol,
        soft_boundaries=soft_boundaries,
        resolution_scale_factor=DEFAULT_RESOLUTION_SCALE_FACTOR,
        candidate_aspect_ratios=candidate_aspect_ratios,
        debug_dir=debug_dir,
    )

    # Optional debug output for edge detection
    if debug_dir is not None:
        for strip in strips:
            debug_plots.save_image(
                os.path.join(debug_dir, f"strip_{strip.position.value}.png"),
                strip.pixels[:, :, ::-1],
            )
            debug_plots.save_image(
                os.path.join(debug_dir, f"edge_weights_{strip.position.value}.png"),
                strip.edge_weights * 255.0,
            )

    # Stage 3: Find best angles (reduction: all strips → best angles)
    best_angles: list[float] = find_best_overall_angles(
        strips,
        max_num_peaks=max_candidate_angles,
        include_single_strip_hypotheses=include_single_strip_angle_hypotheses,
        debug_dir=debug_dir,
    )

    # Stage 4: Get candidates (map: (strip, angles) → strip + candidates)
    candidates = map_strip_set(
        functools.partial(
            get_candidate_edges,
            best_angles=best_angles,
            max_num_peaks=max_candidate_intercepts_per_angle,
            debug_dir=debug_dir,
        ),
        strips,
    )

    # Stage 5: Find best hypothesis (reduction: all candidates → best corners)
    best_corners: QuadArray = find_best_hypothesis(
        strips,
        candidates,
        candidate_aspect_ratios=candidate_aspect_ratios,
        aspect_preference_strength=aspect_preference_strength,
        debug_dir=debug_dir,
    )

    # Optional debug output for final result
    if debug_dir is not None:
        debug_plots.save_image(
            os.path.join(debug_dir, "result_functional.png"),
            debug_plots.annotate_image(pil_image, [best_corners]),
        )

    return best_corners
