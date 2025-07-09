# Suppress lint errors from uppercase variable names
# ruff: noqa N806, N803

from __future__ import annotations

import logging
import os
import pathlib
from enum import Enum
from typing import Any, Callable, Sequence

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


class StripPosition(Enum):
    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"

    @property
    def is_horizontal(self):
        return self.value in ("top", "bottom")


class StripData:
    """Container for a border strip and its coordinate transformations."""

    position: StripPosition
    pixels: UInt8Array
    image_to_strip_transform: Callable[[FloatArray], FloatArray]
    strip_to_image_transform: Callable[[FloatArray], FloatArray]
    image_height: int
    image_width: int

    edge_weights: FloatArray
    mask: UInt8Array

    angle_scores: FloatArray
    angles: FloatArray

    intercept_scores: FloatArray
    intercept_bins: FloatArray

    candidate_intercepts: dict[float, Sequence[np.floating[Any]]]
    candidate_intercept_scores: dict[float, Sequence[np.floating[Any]]]
    candidate_edges: dict[float, Sequence[FloatArray]]

    def __init__(
        self,
        position: StripPosition,
        pixels: UInt8Array,
        image_to_strip_transform: Callable[[FloatArray], FloatArray],
        strip_to_image_transform: Callable[[FloatArray], FloatArray],
        image_height: int,
        image_width: int,
    ) -> None:
        self.position = position
        self.pixels = pixels
        self.image_to_strip = image_to_strip_transform
        self.strip_to_image = strip_to_image_transform
        self.image_height = image_height
        self.image_width = image_width

        self.candidate_edges = {}
        self.candidate_intercepts = {}
        self.candidate_intercept_scores = {}


def snap_to(x, candidates):
    return min(candidates, key=lambda c: abs(x - c))


def extract_border_strips(
    image: Image.Image | UInt8Array,
    rect: QuadArray,
    reltol: float,
    resolution_scale_factor: float = 1.0,
    min_image_pixels: int = 8,
    candidate_aspect_ratios: list[float] | None = None,
    debug_dir: str | None = None,
) -> dict[StripPosition, StripData]:
    """Extract four border strips from the image."""
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
        del image
    else:
        pil_image = image

    # Ensure rect is sorted clockwise
    rect = geometry.sort_clockwise(rect)
    width, height = geometry.dimension_bounds(rect)

    reltol_x = max(reltol, min_image_pixels / width)
    reltol_y = max(reltol, min_image_pixels / height)

    # say given rect has aspect ratio init_ratio = `width/height`. we suspect
    # the refined rect will have aspect ratio `target_ratio`. we want to
    # ensure we extract strips that will contain the edges at the target ratio.
    # the model here is that three of the four edges are within a tolerance
    # of the initial edges. and given that, we want to expand the search space
    # for the final edge. and of course we wat to do this for all edges.
    # so say we're concerned with the right edge. we're working in unit-square
    # coords, meaning that everything we do is going to be scaled up by
    # `init_ratio`. so we say that the left edge is at position `[-rx, rx]`
    # and our candidate right strip is at position `[1 - rx, 1 + rx]`.
    # now if our left strip were *actually* at -rx, and we had the target aspect
    # ratio, the right strip would be at `-rx + target_ratio / init_ratio`
    # similarly if the left edge were at +rx, the right edge would be at `rx + target_ratio/init_ratio`
    # now if the top edge were at -ry, the bottom edge would be at
    # `-ry + target_height / init_height` = `-ry + init_ratio / target_ratio`
    # since the aspect ratio width/height varies reciprocally with height.

    long = max(width, height)
    short = min(width, height)
    portrait = width < height
    init_aspect_ratio = long / short
    if candidate_aspect_ratios:
        target_aspect_ratio = snap_to(init_aspect_ratio, candidate_aspect_ratios)
        print(
            f"expanding to include canddiate aspect ratio {target_aspect_ratio} (from {init_aspect_ratio})"
        )
    else:
        target_aspect_ratio = init_aspect_ratio

    aspect_expansion_x = target_aspect_ratio / init_aspect_ratio - 1.0
    aspect_expansion_y = init_aspect_ratio / target_aspect_ratio - 1.0
    if portrait:
        aspect_expansion_x, aspect_expansion_y = aspect_expansion_y, aspect_expansion_x

    # TODO: aspect ratio expansion is not just translated across top/bottom
    # and left/right! we want to push up the top boundary of the top strip,
    # and down the bottom boundary of the bottom strip.
    # and generally it matters what sign aspect_expansion_x is
    # 05-24-0008 top image is a test case
    strip_left_x = min(-reltol_x, -reltol_x + aspect_expansion_x)
    strip_right_x = max(reltol_x, reltol_x + aspect_expansion_x)
    strip_top_y = min(-reltol_y, -reltol_y + aspect_expansion_y)
    strip_bottom_y = max(reltol_y, reltol_y + aspect_expansion_y)

    strips = {}

    # Define normalized coordinates for each strip in the unit square
    # Horizontal strips
    top_strip_normalized = np.array(
        [
            [0 - reltol_x, strip_top_y],
            [1 + reltol_x, strip_top_y],
            [1 + reltol_x, strip_bottom_y],
            [0 - reltol_x, strip_bottom_y],
        ]
    )
    bottom_strip_normalized = top_strip_normalized + np.array([0.0, 1.0])

    # Vertical strips
    left_strip_normalized = np.array(
        [
            [strip_left_x, 0 - reltol_y],
            [strip_right_x, 0 - reltol_y],
            [strip_right_x, 1 + reltol_y],
            [strip_left_x, 1 + reltol_y],
        ]
    )
    right_strip_normalized = left_strip_normalized + np.array([1.0, 0.0])

    # Convert to image coordinates and extract each strip
    converter = geometry.PatchCoordinatesConverter(rect)

    for position, strip_normalized in [
        (StripPosition.TOP, top_strip_normalized),
        (StripPosition.BOTTOM, bottom_strip_normalized),
        (StripPosition.LEFT, left_strip_normalized),
        (StripPosition.RIGHT, right_strip_normalized),
    ]:
        # Convert normalized coords to image coords
        strip_corners_image = converter.unit_square_to_image(strip_normalized)
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
        strip_converter = geometry.PatchCoordinatesConverter(strip_corners_image)
        image_to_strip_transform = (
            lambda pts,
            sc=strip_converter,
            sw=strip_width,
            sh=strip_height: sc.image_to_unit_square(pts) * np.array([sw, sh])
        )
        strip_to_image_transform = (
            lambda pts,
            sc=strip_converter,
            sw=strip_width,
            sh=strip_height: sc.unit_square_to_image(pts / np.array([sw, sh]))
        )

        # Store strip data
        strips[position] = StripData(
            position=position,
            pixels=pixels_array,
            image_to_strip_transform=image_to_strip_transform,
            strip_to_image_transform=strip_to_image_transform,
            image_height=pil_image.height,
            image_width=pil_image.width,
        )

    return strips


def detect_edges(strip: StripData, image_shape):
    strip_mask = geometry.image_boundary_mask(
        image_shape=image_shape,
        patch_shape=strip.pixels.shape,
        image_to_patch_coords=strip.image_to_strip,
    ).astype(strip.pixels.dtype)

    # edge_response_horizontal = refine_strips.detect_edges_color(
    #    pixels_array,
    #    horizontal=True,
    #    mask=strip_mask,
    # )
    # edge_response_vertical = refine_strips.detect_edges_color(
    #    pixels_array,
    #    horizontal=False,
    #    mask=strip_mask,
    # )
    gray = cv2.cvtColor(strip.pixels, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # sobel_x = cv2.Sobel(gray, -1, 1, 0, ksize=5)
    # sobel_y = cv2.Sobel(gray, -1, 0, 1, ksize=5)

    filter = np.array(
        [[-1, -2, -1], [-1, -2, -1], [1, 2, 1], [1, 2, 1]], dtype=np.float32
    )
    # sobel_y = cv2.filter2D(gray, -1, filter)  # type: ignore
    # sobel_x = cv2.filter2D(gray, -1, np.array(filter.T))
    # magnitude = np.sqrt(sobel_x**2 + sobel_y**2) * strip_mask
    # normed_magnitude = magnitude / np.max(magnitude)
    # angle = np.arctan2(sobel_y, sobel_x)
    if strip.position.is_horizontal:
        edge_weights = cv2.filter2D(gray, -1, filter)  # sobel_y
    else:
        edge_weights = cv2.filter2D(gray, -1, np.array(filter.T))  # sobel_x

    edge_weights *= strip_mask
    edge_weights = np.sqrt(np.abs(edge_weights))
    edge_weights = cv2.GaussianBlur(edge_weights, (5, 5), 0)
    edge_weights *= strip_mask
    # don't use votes from the edge pixels where sobel directions aren't
    # fully defined
    edge_weights[:2, :] = 0.0
    edge_weights[-2:, :] = 0
    edge_weights[:, :2] = 0.0
    edge_weights[:, -2:] = 0
    edge_weights /= np.max(edge_weights)

    strip.edge_weights = edge_weights
    strip.mask = strip_mask


def compute_fft_spectrum(image):
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


def correct_angle_aspect(angle, height, width, eps=1e-4):
    aspect_corrected_angle = np.arctan((height / width) * np.tan(angle))
    aspect_corrected_angle = np.where(
        abs(angle - np.pi / 2) < eps,
        angle,
        aspect_corrected_angle,
    )
    return aspect_corrected_angle


def radial_profile_corrected(magnitude, central_angle, num_angles=180):
    """Compute radial profile with aspect ratio correction

    central_angle: the *aspect-corrected* angle around which the profile should be focused
    """
    height, width = magnitude.shape
    center_y, center_x = height // 2, width // 2

    # radius_pixels = max(center_y, center_x)
    # circumference_pixels = 2 * np.pi * radius_pixels
    # pixel_subtended = 2 * np.pi / circumference_pixels
    # num_angles = int(np.ceil(np.pi / pixel_subtended))
    # num_angles += num_angles % 2  # ensure even
    # print("USING ANGLES", num_angles)
    # want angles from -22.5 to 22.5 (-pi/8 to pi/8)
    central_angle_perp = (central_angle + np.pi / 2) % np.pi
    min_angle = correct_angle_aspect(
        central_angle_perp - 1 / 8 * np.pi, width=height, height=width
    )
    min_angle = (min_angle + central_angle) % np.pi - central_angle_perp
    max_angle = correct_angle_aspect(
        central_angle_perp + 1 / 8 * np.pi, width=height, height=width
    )
    max_angle = (max_angle + central_angle) % np.pi - central_angle_perp
    # min_angle, max_angle = min(min_angle, max_angle), max(min_angle, max_angle)
    # angles = np.linspace(0, np.pi, num_angles, endpoint=False)
    angles = np.linspace(min_angle, max_angle, num_angles, endpoint=False)

    profile = np.zeros(num_angles)

    # Maximum radius in normalized coordinates
    max_radius = 0.8  # Don't go all the way to edges

    # Correct for aspect ratio (inverse transformation)
    perpendicular_angle = (angles + np.pi / 2) % np.pi
    aspect_corrected_angle = (
        correct_angle_aspect(perpendicular_angle, height=height, width=width) % np.pi
    )
    print("min_angle", min_angle, "max", max_angle)
    print(
        "corrected perp min",
        np.degrees(aspect_corrected_angle[0]),
        "max",
        np.degrees(aspect_corrected_angle[-1]),
    )
    if min_angle > max_angle:
        import pdb

        pdb.set_trace()
    for i, angle in enumerate(angles):
        # Sample points along this radial direction
        # Correct for aspect ratio by scaling the x-component
        num_samples = 50

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
            # TODO: what order interpolation for sampling?
            samples = ndimage.map_coordinates(magnitude, [ys, xs], order=0)
            profile[i] = np.mean(samples)
            # print(
            #    f"angle {np.degrees(angle): .3f} (cor {np.degrees(aspect_corrected_angle[i]): .2f}) endpoint {xs[-1]: .4f}, {ys[-1]: .4f} prof {profile[i]: .2f}"
            # )

    return aspect_corrected_angle, profile


def add_sampled_functions(xs, ys, kind="cubic"):
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
        x_min, x_max = np.min(xs[i]), np.max(xs[i])
        x_valid = x_valid[(x_valid >= x_min) & (x_valid <= x_max)]

    y_interps = []
    # Interpolate
    for i in range(len(xs)):
        f_interp = interp1d(xs[i], ys[i], kind=kind, bounds_error=False, fill_value=0)
        y_interps.append(f_interp(x_valid))

    # Sum the functions
    y_sum = np.sum(y_interps, axis=0)

    return x_valid, y_sum


def get_sorted_peak_indices(scores: FloatArray, max_num_peaks=2) -> IntArray:
    peaks, extra = find_peaks(scores, prominence=np.max(scores) / 10.0)
    if len(peaks) == 0:
        return np.array([np.argmax(scores)])
    else:
        perm = np.argsort(extra["prominences"])
        # Return peaks sorted by score in descending order.
        return np.array(peaks)[perm][: -max_num_peaks - 1 : -1]


def score_angles_in_strip(
    strip: StripData, debug_dir: str | None = None
) -> tuple[FloatArray, FloatArray]:
    _, magnitude = compute_fft_spectrum(strip.edge_weights)
    image_angles, profile = radial_profile_corrected(
        magnitude, central_angle=np.pi / 2 if strip.position.is_horizontal else 0.0
    )
    if strip.position.is_horizontal:
        image_angles = (image_angles - np.pi / 2) % np.pi

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
            os.path.join(debug_dir, f"log_fft_magnitude_{strip.position.value}.png"),
            normed_log_magnitude,
        )
        debug_plots.save_plot(
            os.path.join(debug_dir, f"angles_fft_{strip.position.value}.png"),
            np.degrees(angles),
            angle_scores,
            f"{strip.position} image angles max {angles[np.argmax(profile)]}",
        )
    return angles, angle_scores


def find_best_overall_angles(
    strips: dict[StripPosition, StripData],
    debug_dir: str | None = None,
    max_num_peaks=3,
) -> list[float]:
    # Compute angle scores from each edge strip.
    strip_angles, strip_angle_scores = zip(
        *[
            score_angles_in_strip(strip, debug_dir=debug_dir)
            for strip in strips.values()
        ]
    )

    # Sum scores across strips, letting each strip 'vote' for the overall angle. We
    # normalize the scores, so each strip has max score 1, to improve robustness.
    # This bounds the impact of any weird stuff going on in one of the four strips and
    # prefers angles that are present in multiple strips (e.g., any two strips voting
    # together for a coherent angle can get a combined score of 2, outvoting any lone
    # strip, which will have a max vote of 1 no matter how weird it is).
    normalized_angle_scores = [s / np.max(s) for s in strip_angle_scores]
    combined_angles, overall_angle_scores = add_sampled_functions(
        strip_angles, normalized_angle_scores
    )

    if debug_dir is not None:
        peaks, _ = find_peaks(
            overall_angle_scores, prominence=np.max(overall_angle_scores) / 10.0
        )
        angle_peaks = np.array([combined_angles[idx] for idx in peaks])
        print(f"overall peaks {angle_peaks}")
        debug_plots.save_plot(
            os.path.join(debug_dir, f"angles_fft_overall.png"),
            np.degrees(combined_angles),
            overall_angle_scores,
            f"overall angle peaks {[np.degrees(a) for a in angle_peaks]}",
        )

    best_idxs = get_sorted_peak_indices(
        overall_angle_scores, max_num_peaks=max_num_peaks
    )
    # best_idx = np.argmax(overall_angle_scores)
    best_angles = [combined_angles[int(idx)] for idx in best_idxs]
    print("best angle", [np.degrees(a) for a in best_angles])
    return [a - (np.pi / 2) for a in best_angles]


def line_from_points(y1: float, x1: float, y2: float, x2: float):
    """
    Convert two points to line equation y = mx + b.

    Returns:
        (slope, intercept, is_vertical)
    """
    if abs(x2 - x1) < 1e-10:  # Vertical line
        return float("inf"), x1, True

    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept, False


def intercept_for_line_bounded_by_edge(
    edge_pt1,
    edge_pt2,
    border_slope,
    x_min,
    x_max,
    bounded_above=True,
    shrink_inwards_by=2.0,
):
    x1, y1 = edge_pt1
    x2, y2 = edge_pt2
    m1, b1, edge_is_vertical = line_from_points(y1, x1, y2, x2)

    if edge_is_vertical:
        return None

    relative_slope = m1 - border_slope
    cond_sign = 1 if bounded_above else -1
    if cond_sign * relative_slope <= 0:
        return relative_slope * x_min + b1 + cond_sign * shrink_inwards_by
    else:
        return relative_slope * x_max + b1 + cond_sign * shrink_inwards_by


def bincount_histogram(xs, bin_min, bin_max, weights):
    bins = np.arange(bin_min, bin_max + 1)
    # Bin xs to integer indices.
    x_indices = np.floor(xs - bin_min).astype(int)
    # Create a zeroth bin for outliers.
    x_indices = np.where((x_indices < 0) | (x_indices >= bin_max), 0, x_indices + 1)
    hist: FloatArray = np.bincount(x_indices, weights=weights, minlength=len(bins))[1:]  # type: ignore
    return hist, bins


def score_intercepts_for_strip(strip: StripData, slope: float):
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
        f"strip {strip.position.value} shape {strip.pixels.shape[:2]} min intercept {intercept_bins[0]} max intercept {intercept_bins[1]} bins {len(intercept_bins)}"
    )
    strip.intercept_bins = intercept_bins
    strip.intercept_scores = intercept_scores


def get_bin_idx(bins: np.ndarray, x: float):
    if x is not None and (x >= bins[0] and x < bins[-1]):
        # Find the bin containing x (the first bin with right boundary greater than it).
        bin_idx = np.argmax(bins[1:] > x)
        return bin_idx
    return None


def intercept_of_line_touching_image_edge(
    strip: StripData,
    slope: float,
) -> float | None:
    image_height, image_width = strip.image_height, strip.image_width
    strip_height, strip_width = strip.pixels.shape[:2]
    if strip.position == StripPosition.TOP:
        x_min = -strip_width / 2
        x_max = strip_width / 2
        border_pt1 = strip.image_to_strip(np.array([(0, 0.0)])).flatten()
        border_pt2 = strip.image_to_strip(np.array([(image_width - 1, 0.0)])).flatten()
        bounded_above = True
    elif strip.position == StripPosition.BOTTOM:
        x_min = -strip_width / 2
        x_max = strip_width / 2
        border_pt1 = strip.image_to_strip(np.array([(image_width - 1, 0.0)])).flatten()
        border_pt2 = strip.image_to_strip(
            np.array([(image_width - 1, image_height - 1)])
        ).flatten()
        bounded_above = False
    elif strip.position == StripPosition.LEFT:
        x_min = -strip_height / 2
        x_max = strip_height / 2
        border_pt1 = strip.image_to_strip(np.array([(0.0, 0.0)])).flatten()
        border_pt2 = strip.image_to_strip(np.array([(0.0, image_height - 1)])).flatten()
        # Transpose coordinates, treat strip as horizontal.
        border_pt1 = border_pt1[::-1]
        border_pt2 = border_pt2[::-1]
        bounded_above = True
    elif strip.position == StripPosition.RIGHT:
        x_min = -strip_height / 2
        x_max = strip_height / 2
        border_pt1 = strip.image_to_strip(np.array([(image_width - 1, 0.0)])).flatten()
        border_pt2 = strip.image_to_strip(
            np.array([(image_width - 1, image_height - 1)])
        ).flatten()
        # Transpose coordinates, treat strip as horizontal.
        border_pt1 = border_pt1[::-1]
        border_pt2 = border_pt2[::-1]
        bounded_above = False

    return intercept_for_line_bounded_by_edge(
        border_pt1, border_pt2, slope, x_min, x_max, bounded_above=bounded_above
    )


def intercept_to_image_points(
    strip: StripData, slope: float, intercept: float
) -> FloatArray:
    height, width = strip.pixels.shape[:2]
    if strip.position.is_horizontal:
        pt1 = np.array([0, intercept - slope * width / 2])
        pt2 = np.array([width - 1, intercept + slope * width / 2])
    else:
        pt1 = np.array([intercept + slope * height / 2, 0])
        pt2 = np.array([intercept - slope * height / 2, height - 1])
    pt1_image = strip.strip_to_image(pt1.reshape((1, -1)))
    pt2_image = strip.strip_to_image(pt2.reshape((1, -1)))
    return np.concatenate([pt1_image, pt2_image], axis=0)


def get_candidate_edges(
    strips: dict[StripPosition, StripData],
    candidate_angle,
    max_num_peaks: int = 1,
    debug_dir=None,
):
    slope = np.tan(candidate_angle)
    print("candidate angle", candidate_angle, "slope", slope)

    # For each strip, compute the set of candidate edge intercepts and their scores.
    # This sets strip.intercept_bins and strip.intercept_scores.
    for strip in strips.values():
        score_intercepts_for_strip(strip, slope=slope)

    # For strips that touch the border of the image, insert a "bonus" candidate
    # edge at that border, in case the photo itself runs past the border and
    # its real edge isn't visible.
    median_peak_score = np.median(  # TODO: can/should this just be a constant?
        [np.max(strip.intercept_scores) for strip in strips.values()]
    )
    for position, strip in strips.items():
        image_edge_intercept = intercept_of_line_touching_image_edge(strip, slope)
        if image_edge_intercept is not None:
            image_edge_bin_idx = get_bin_idx(strip.intercept_bins, image_edge_intercept)
            if image_edge_bin_idx is not None:
                # TODO tune the strength of edge preference here.
                print(
                    f"strip touched image border, boosting intercept {strip.intercept_bins[image_edge_bin_idx]}"
                )
                strip.intercept_scores[image_edge_bin_idx] += median_peak_score / 4.0

    # For each strip, identify the most prominent candidate edges.
    for position, strip in strips.items():
        peak_indices = get_sorted_peak_indices(
            strip.intercept_scores, max_num_peaks=max_num_peaks
        )
        peak_indices = [int(i) for i in peak_indices]  # Make the type checker happy.
        strip.candidate_intercepts[candidate_angle] = [
            strip.intercept_bins[i] for i in peak_indices
        ]
        strip.candidate_intercept_scores[candidate_angle] = [
            strip.intercept_scores[i] for i in peak_indices
        ]
        print(
            f"angle {np.degrees(candidate_angle): .2f} {position}: candidate intercepts {strip.candidate_intercepts} with scores {strip.candidate_intercept_scores}"
        )

        if debug_dir:
            debug_plots.save_histogram(
                os.path.join(
                    debug_dir,
                    f"intercepts_at_{np.degrees(candidate_angle): .2f}_{position.value}.png",
                ),
                strip.intercept_bins,
                strip.intercept_scores,
                f"{position.value} intercepts at {np.degrees(candidate_angle): .2f} {strip.candidate_intercepts[candidate_angle]}",
            )

        strip.candidate_edges[candidate_angle] = [
            intercept_to_image_points(strip, slope=slope, intercept=float(intercept))
            for intercept in strip.candidate_intercepts[candidate_angle]
        ]


def enumerate_hypotheses(strips: dict[StripPosition, StripData]):
    angles = strips[StripPosition.TOP].candidate_intercepts.keys()
    for angle in angles:
        for top_idx in range(
            len(strips[StripPosition.TOP].candidate_intercepts[angle])
        ):
            for bottom_idx in range(
                len(strips[StripPosition.BOTTOM].candidate_intercepts[angle])
            ):
                for left_idx in range(
                    len(strips[StripPosition.LEFT].candidate_intercepts[angle])
                ):
                    for right_idx in range(
                        len(strips[StripPosition.RIGHT].candidate_intercepts[angle])
                    ):
                        edges = {
                            StripPosition.TOP: strips[
                                StripPosition.TOP
                            ].candidate_edges[angle][top_idx],
                            StripPosition.BOTTOM: strips[
                                StripPosition.BOTTOM
                            ].candidate_edges[angle][bottom_idx],
                            StripPosition.LEFT: strips[
                                StripPosition.LEFT
                            ].candidate_edges[angle][left_idx],
                            StripPosition.RIGHT: strips[
                                StripPosition.RIGHT
                            ].candidate_edges[angle][right_idx],
                        }
                        scores = [
                            strips[StripPosition.TOP].candidate_intercept_scores[angle][
                                top_idx
                            ],
                            strips[StripPosition.BOTTOM].candidate_intercept_scores[
                                angle
                            ][bottom_idx],
                            strips[StripPosition.LEFT].candidate_intercept_scores[
                                angle
                            ][left_idx],
                            strips[StripPosition.RIGHT].candidate_intercept_scores[
                                angle
                            ][right_idx],
                        ]
                        print(
                            f"proposing hypothesis w angle {angle} idx {(top_idx, bottom_idx, left_idx, right_idx)} edge score {sum(scores)}"
                        )
                        yield edges, scores


def score_aspect_ratio(
    corners: QuadArray,
    candidate_aspect_ratios,
    aspect_preference_strength=1.0,
    aspect_rtol=0.025,
):
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


def refine_strips_hough(
    image: Image.Image | np.ndarray,
    corner_points,
    reltol=0.05,
    debug_dir=None,
    max_candidate_angles=2,
    max_candidate_intercepts_per_angle=2,
    aspect_preference_strength: float = 0.1,
    candidate_aspect_ratios: list[float] | None = None,
):
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
    image_shape = (pil_image.height, pil_image.width)

    corner_points = bounding_box_as_array(corner_points)

    # Get minimum bounding rectangle
    # rect is in (x, y) image coords.
    rect, _ = geometry.minimum_bounding_rectangle(corner_points)
    rect = geometry.sort_clockwise(rect)

    strips = extract_border_strips(
        pil_image,
        rect,
        reltol=reltol,
        resolution_scale_factor=1.0,
        candidate_aspect_ratios=candidate_aspect_ratios,
        debug_dir=debug_dir,
    )
    for strip in strips.values():
        detect_edges(strip, image_shape=image_shape)
    if debug_dir is not None:
        for position, strip in strips.items():
            debug_plots.save_image(
                os.path.join(debug_dir, f"strip_{position.value}.png"),
                strip.pixels[:, :, ::-1],
            )
            debug_plots.save_image(
                os.path.join(debug_dir, f"edge_weights_{position.value}.png"),
                strip.edge_weights * 255.0,
            )

    best_angles = find_best_overall_angles(
        strips, debug_dir=debug_dir, max_num_peaks=max_candidate_angles
    )
    for candidate_angle in best_angles:
        get_candidate_edges(
            strips,
            candidate_angle=candidate_angle,
            debug_dir=debug_dir,
            max_num_peaks=max_candidate_intercepts_per_angle,
        )

    best_score = 0.0
    best_edges = None
    best_corners = None
    total_border_pixels = sum(
        [
            strip.pixels.shape[1 if strip.position.is_horizontal else 0]
            for strip in strips.values()
        ]
    )
    for edges, edge_scores in enumerate_hypotheses(strips):
        corners = refine_strips.find_corner_intersections(
            {p.value: e for (p, e) in edges.items()}
        )
        edge_score = sum(edge_scores)
        aspect_score = score_aspect_ratio(
            corners,
            candidate_aspect_ratios,
            # TODO calibrate preference strength vs edge scores / pixel resolution.
            aspect_preference_strength=aspect_preference_strength * total_border_pixels,
        )
        score = edge_score + aspect_score
        if score > best_score:
            best_score = score
            best_edges = edges
            best_corners = corners

    if best_edges is None or best_corners is None:
        raise ValueError("no hypotheses had positive score!!")  # should never happen.

    if debug_dir:
        for position, strip in strips.items():
            strip_edge = strip.image_to_strip(best_edges[position])
            debug_plots.save_image(
                os.path.join(debug_dir, f"edge_{position.value}.png"),
                debug_plots.annotate_image(
                    strip.pixels, edges=[np.round(strip_edge).astype(int)]
                ),
            )

    if debug_dir is not None:
        debug_plots.save_image(
            os.path.join(debug_dir, "result.png"),
            debug_plots.annotate_image(pil_image, [best_corners]),
        )

    # if np.abs(np.degrees(best_angle)) > 10.0:
    #    # Automatically re-refine if the original bounding box was waay skewed.
    #    return refine_strips_hough(
    #        pil_image,
    #        corner_points=best_corners,
    #        reltol=reltol,
    #        debug_dir=debug_dir,
    #        candidate_aspect_ratios=candidate_aspect_ratios,
    #    )

    return best_corners
