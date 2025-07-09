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

# Algorithm constants
DEFAULT_RESOLUTION_SCALE_FACTOR = 1.0
MIN_IMAGE_PIXELS_DEFAULT = 8
EDGE_WEIGHT_BLUR_KERNEL_SIZE = (5, 5)
EDGE_WEIGHT_BLUR_SIGMA = 0
EDGE_PIXELS_TO_ZERO = 2  # Zero out edge pixels where gradients aren't well-defined
FFT_MAX_RADIUS_FRACTION = 0.8  # Don't sample all the way to FFT edges
FFT_RADIAL_SAMPLES = 50
FFT_ANGLE_RANGE_FRACTION = 1.0 / 8.0  # +/- pi/8 around central angle
PEAK_PROMINENCE_FRACTION = 1.0 / 10.0  # Peaks must be 1/10 of max height
IMAGE_EDGE_BOOST_FRACTION = 1.0 / 4.0  # Boost for edges at image boundary
SHRINK_INWARDS_PIXELS = 2.0  # Pixels to shrink edge inwards
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


class StripData:
    """Container for a border strip and its coordinate transformations.

    This class holds all the data associated with one of the four border strips
    extracted around a photo. It includes the strip image data, coordinate
    transformations, and results from various stages of the edge detection process.

    The coordinate transformations allow converting between:
    - Image coordinates: (x, y) in the original image
    - Strip coordinates: (x, y) in the extracted strip's pixel space

    Fields are populated progressively as the algorithm proceeds:
    - Initial: position, pixels, transforms, image dimensions
    - After edge detection: edge_weights, mask
    - After angle scoring: angles, angle_scores
    - After intercept scoring: intercept_bins, intercept_scores
    - After candidate generation: candidate_* dictionaries
    """

    position: StripPosition
    pixels: UInt8Array
    image_to_strip_transform: Callable[[FloatArray], FloatArray]
    strip_to_image_transform: Callable[[FloatArray], FloatArray]
    image_height: int
    image_width: int

    edge_weights: FloatArray | None
    mask: UInt8Array | None

    angle_scores: FloatArray | None
    angles: FloatArray | None

    intercept_scores: FloatArray | None
    intercept_bins: FloatArray | None

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
        # Initial strip data populated at construction
        self.position = position
        self.pixels = pixels
        self.image_to_strip = image_to_strip_transform
        self.strip_to_image = strip_to_image_transform
        self.image_height = image_height
        self.image_width = image_width

        # Fields populated by detect_edges() - initialized to None
        self.edge_weights: FloatArray | None = None
        self.mask: UInt8Array | None = None

        # Fields populated by find_best_overall_angles() - initialized to None
        self.angles: FloatArray | None = None
        self.angle_scores: FloatArray | None = None

        # Fields populated by score_intercepts_for_strip() - initialized to None
        self.intercept_bins: FloatArray | None = None
        self.intercept_scores: FloatArray | None = None

        # Fields populated by get_candidate_edges() - initialized to empty dicts
        self.candidate_edges: dict[float, Sequence[FloatArray]] = {}
        self.candidate_intercepts: dict[float, Sequence[np.floating[Any]]] = {}
        self.candidate_intercept_scores: dict[float, Sequence[np.floating[Any]]] = {}


def snap_to(x, candidates):
    """Snap a value to the nearest candidate from a list.

    Args:
        x: Value to snap
        candidates: List of candidate values

    Returns:
        The candidate value closest to x
    """
    return min(candidates, key=lambda c: abs(x - c))


def calculate_strip_boundaries(
    width: float,
    height: float,
    reltol_x: float,
    reltol_y: float,
    candidate_aspect_ratios: list[float] | None = None,
) -> tuple[float, float, float, float]:
    """Calculate the boundaries for border strips, including aspect ratio expansion.

    This function determines how far to extend the strips beyond the initial
    rectangle boundaries. If candidate aspect ratios are provided, it expands
    the strips to ensure they would capture edges at the expected ratios.

    The expansion logic accounts for portrait vs landscape orientation and
    ensures the strips are wide enough to detect edges that might be positioned
    differently if the photo has a different aspect ratio than initially estimated.

    Args:
        width: Rectangle width
        height: Rectangle height
        reltol_x: Relative tolerance in x direction
        reltol_y: Relative tolerance in y direction
        candidate_aspect_ratios: Optional list of expected aspect ratios

    Returns:
        Tuple of (strip_left_x, strip_right_x, strip_top_y, strip_bottom_y)
        representing the strip boundaries in normalized coordinates
    """
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

    # Calculate how much to expand strips for target aspect ratio
    # If target is wider than current, expand horizontally (aspect_expansion_x > 0)
    # If target is taller than current, expand vertically (aspect_expansion_y > 0)
    aspect_expansion_x = target_aspect_ratio / init_aspect_ratio - 1.0
    aspect_expansion_y = init_aspect_ratio / target_aspect_ratio - 1.0

    # For portrait images, swap x and y expansions since long/short are flipped
    if portrait:
        aspect_expansion_x, aspect_expansion_y = aspect_expansion_y, aspect_expansion_x

    # TODO: aspect ratio expansion is not just translated across top/bottom
    # and left/right! we want to push up the top boundary of the top strip,
    # and down the bottom boundary of the bottom strip.
    # and generally it matters what sign aspect_expansion_x is
    # 05-24-0008 top image is a test case
    strip_boundary_left = min(-reltol_x, -reltol_x + aspect_expansion_x)
    strip_boundary_right = max(reltol_x, reltol_x + aspect_expansion_x)
    strip_boundary_top = min(-reltol_y, -reltol_y + aspect_expansion_y)
    strip_boundary_bottom = max(reltol_y, reltol_y + aspect_expansion_y)

    return (
        strip_boundary_left,
        strip_boundary_right,
        strip_boundary_top,
        strip_boundary_bottom,
    )


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


def extract_border_strips(
    image: Image.Image | UInt8Array,
    rect: QuadArray,
    reltol: float,
    resolution_scale_factor: float = DEFAULT_RESOLUTION_SCALE_FACTOR,
    min_image_pixels: int = MIN_IMAGE_PIXELS_DEFAULT,
    candidate_aspect_ratios: list[float] | None = None,
    debug_dir: str | None = None,
) -> dict[StripPosition, StripData]:
    """Extract four border strips from around a rectangle in the image.

    This function creates strips around the perimeter of a rectangle,
    with optional expansion to account for expected aspect ratios.
    Each strip contains the image data and coordinate transformations
    needed for edge detection.

    Args:
        image: The source image
        rect: Four corners of the rectangle in image coordinates
        reltol: Relative tolerance for strip width (fraction of image size)
        resolution_scale_factor: Scale factor for strip resolution
        min_image_pixels: Minimum strip width in pixels
        candidate_aspect_ratios: Expected aspect ratios for expansion
        debug_dir: Optional directory for saving debug images

    Returns:
        Dictionary mapping StripPosition to StripData for each of the four strips
    """
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

    # Calculate strip boundaries with aspect ratio expansion
    (
        strip_boundary_left,
        strip_boundary_right,
        strip_boundary_top,
        strip_boundary_bottom,
    ) = calculate_strip_boundaries(
        width, height, reltol_x, reltol_y, candidate_aspect_ratios
    )

    strips = {}

    # Define normalized coordinates for each strip in the unit square
    # Horizontal strips (top and bottom)
    top_strip_normalized = np.array(
        [
            [0 - reltol_x, strip_boundary_top],
            [1 + reltol_x, strip_boundary_top],
            [1 + reltol_x, strip_boundary_bottom],
            [0 - reltol_x, strip_boundary_bottom],
        ]
    )
    bottom_strip_normalized = top_strip_normalized + np.array([0.0, 1.0])

    # Vertical strips (left and right)
    left_strip_normalized = np.array(
        [
            [strip_boundary_left, 0 - reltol_y],
            [strip_boundary_right, 0 - reltol_y],
            [strip_boundary_right, 1 + reltol_y],
            [strip_boundary_left, 1 + reltol_y],
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
        image_to_strip_transform, strip_to_image_transform = (
            create_strip_coordinate_transforms(
                strip_corners_image, strip_width, strip_height
            )
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
    """Detect edge weights in a strip using gradient filters.

    This function applies directional gradient filters to detect edges
    perpendicular to the strip orientation. For horizontal strips, it uses
    a vertical gradient filter, and for vertical strips, a horizontal filter.

    The process includes:
    1. Convert to grayscale
    2. Apply gradient filter appropriate for strip orientation
    3. Apply masking to exclude pixels outside the image boundary
    4. Smooth with Gaussian blur
    5. Normalize weights to [0, 1] range

    Args:
        strip: The strip data to process
        image_shape: Shape of the original image for boundary masking

    Side Effects:
        Sets strip.edge_weights and strip.mask
    """
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
    # Apply gradient filter perpendicular to strip orientation
    # Horizontal strips need vertical gradients to detect horizontal edges
    # Vertical strips need horizontal gradients to detect vertical edges
    if strip.position.is_horizontal:
        edge_weights = cv2.filter2D(gray, -1, filter)  # vertical gradient (sobel_y)
    else:
        edge_weights = cv2.filter2D(
            gray, -1, np.array(filter.T)
        )  # horizontal gradient (sobel_x)

    # Apply image boundary mask and post-process edge weights
    edge_weights *= strip_mask  # Zero out pixels outside image boundary
    # Square root: favor long coherent edges over smaller areas with high response.
    edge_weights = np.sqrt(np.abs(edge_weights))
    edge_weights = cv2.GaussianBlur(
        edge_weights, EDGE_WEIGHT_BLUR_KERNEL_SIZE, EDGE_WEIGHT_BLUR_SIGMA
    )
    edge_weights *= strip_mask  # Re-apply mask after blurring
    # don't use votes from the edge pixels where sobel directions aren't
    # fully defined
    edge_weights[:EDGE_PIXELS_TO_ZERO, :] = 0.0
    edge_weights[-EDGE_PIXELS_TO_ZERO:, :] = 0
    edge_weights[:, :EDGE_PIXELS_TO_ZERO] = 0.0
    edge_weights[:, -EDGE_PIXELS_TO_ZERO:] = 0
    edge_weights /= np.max(edge_weights)

    strip.edge_weights = edge_weights
    strip.mask = strip_mask


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


def correct_angle_aspect(angle, height, width, eps=EPS_ANGLE_CORRECTION):
    """Correct angles for aspect ratio effects in FFT analysis.

    When analyzing FFT patterns in non-square images, angles appear distorted
    due to the aspect ratio. This function corrects for that distortion.

    Args:
        angle: Original angle in radians
        height: Image height
        width: Image width
        eps: Epsilon for handling vertical angles (near π/2)

    Returns:
        Aspect-ratio corrected angle
    """
    # For non-square images, the FFT scaling distorts angles
    # The correction factor (height/width) accounts for this distortion
    aspect_corrected_angle = np.arctan((height / width) * np.tan(angle))

    # Handle vertical angles (π/2) separately since tan(π/2) is undefined
    # Near vertical angles should remain unchanged
    aspect_corrected_angle = np.where(
        abs(angle - np.pi / 2) < eps,
        angle,  # Keep vertical angles unchanged
        aspect_corrected_angle,
    )
    return aspect_corrected_angle


def calculate_angle_range_for_radial_profile(
    central_angle: float,
    height: int,
    width: int,
    num_angles: int = 180,
) -> tuple[FloatArray, FloatArray]:
    """Calculate the angle range for radial profile sampling.

    This function determines the range of angles to sample around a central
    angle, with aspect ratio corrections applied. The sampling is focused
    within ±π/8 of the central angle.

    Args:
        central_angle: The aspect-corrected central angle
        height: Image height
        width: Image width
        num_angles: Number of angles to sample

    Returns:
        Tuple of (angles, aspect_corrected_angles) where:
        - angles: Raw angles for sampling
        - aspect_corrected_angles: Angles corrected for aspect ratio
    """
    # Convert central angle to perpendicular for sampling range calculation
    central_angle_perpendicular = (central_angle + np.pi / 2) % np.pi

    # Calculate angle range boundaries with aspect ratio correction
    min_angle_corrected = correct_angle_aspect(
        central_angle_perpendicular - FFT_ANGLE_RANGE_FRACTION * np.pi,
        width=height,
        height=width,
    )
    min_angle_sampling = (
        min_angle_corrected + central_angle
    ) % np.pi - central_angle_perpendicular

    max_angle_corrected = correct_angle_aspect(
        central_angle_perpendicular + FFT_ANGLE_RANGE_FRACTION * np.pi,
        width=height,
        height=width,
    )
    max_angle_sampling = (
        max_angle_corrected + central_angle
    ) % np.pi - central_angle_perpendicular

    angles = np.linspace(
        min_angle_sampling, max_angle_sampling, num_angles, endpoint=False
    )

    # Correct for aspect ratio (inverse transformation)
    perpendicular_angle = (angles + np.pi / 2) % np.pi
    aspect_corrected_angle = (
        correct_angle_aspect(perpendicular_angle, height=height, width=width) % np.pi
    )

    print("min_angle_sampling", min_angle_sampling, "max", max_angle_sampling)
    print(
        "corrected perp min",
        np.degrees(aspect_corrected_angle[0]),
        "max",
        np.degrees(aspect_corrected_angle[-1]),
    )
    if min_angle_sampling > max_angle_sampling:
        import pdb

        pdb.set_trace()

    return angles, aspect_corrected_angle


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


def radial_profile_corrected(magnitude, central_angle, num_angles=180):
    """Compute radial profile with aspect ratio correction.

    This function samples the FFT magnitude along radial directions,
    with aspect ratio corrections applied to account for non-square images.
    The sampling is focused around a central angle with a limited range.

    Args:
        magnitude: 2D FFT magnitude array
        central_angle: The aspect-corrected angle around which to focus sampling
        num_angles: Number of angles to sample

    Returns:
        Tuple of (aspect_corrected_angles, radial_profile) where:
        - aspect_corrected_angles: Angles corrected for aspect ratio
        - radial_profile: Magnitude values sampled along each angle
    """
    height, width = magnitude.shape
    center_y, center_x = height // 2, width // 2

    # Calculate angle range and aspect corrections
    angles, aspect_corrected_angle = calculate_angle_range_for_radial_profile(
        central_angle, height, width, num_angles
    )

    # Sample radial profile for each angle
    profile = np.zeros(num_angles)
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


def score_angles_in_strip(
    strip: StripData, debug_dir: str | None = None
) -> tuple[FloatArray, FloatArray]:
    """Score angles in a strip using FFT-based analysis.

    This function uses the 2D FFT of the edge weights to detect dominant
    angles in the strip. The FFT magnitude is sampled radially at different
    angles, with aspect ratio corrections applied.

    The central angle for sampling depends on strip orientation:
    - Horizontal strips: π/2 (vertical sampling)
    - Vertical strips: 0 (horizontal sampling)

    Args:
        strip: The strip data containing edge weights
        debug_dir: Optional directory for saving debug plots

    Returns:
        Tuple of (angles, angle_scores) where both are sorted by angle.
        Angles are in image coordinates (not strip coordinates).
    """
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
    """Find the best overall angles by combining evidence from all strips.

    This function implements a voting mechanism where each strip contributes
    its angle preferences from FFT analysis. The scores are normalized so each
    strip has equal maximum influence (preventing any single strip from dominating),
    then combined using interpolation to find the strongest overall angles.

    The approach allows multiple strips to vote for the same angle, building
    confidence, while preventing any single strip from overwhelming the others.

    Args:
        strips: Dictionary mapping strip positions to their data
        debug_dir: Optional directory for saving debug plots
        max_num_peaks: Maximum number of angle peaks to return

    Returns:
        List of best angles in radians, sorted by confidence.
        Note: These angles are relative to the horizontal and need to be
        converted to standard image coordinates (subtract π/2).
    """
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
            overall_angle_scores,
            prominence=np.max(overall_angle_scores) * PEAK_PROMINENCE_FRACTION,
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
    best_angles = [combined_angles[int(idx)] for idx in best_idxs]
    print("best angle", [np.degrees(a) for a in best_angles])
    return [a - (np.pi / 2) for a in best_angles]


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


def score_intercepts_for_strip(
    strip: StripData, slope: float
) -> tuple[FloatArray, FloatArray]:
    """Score all possible intercepts for a given slope using edge voting.

    This implements a Hough-like voting scheme where each pixel with high
    edge weight votes for the intercept of the line passing through it
    at the given slope. The result is a histogram of intercept scores.

    The coordinate system depends on the strip orientation:
    - Horizontal strips: x from -width/2 to +width/2, y from 0 to height
    - Vertical strips: y from -height/2 to +height/2, x from 0 to width

    Args:
        strip: The strip data containing edge weights
        slope: The slope of lines to test

    Returns:
        Tuple of (intercept_scores, intercept_bins) where:
        - intercept_scores: Histogram of edge vote scores for each intercept
        - intercept_bins: Bin center values for the histogram
    """
    weights = strip.edge_weights
    assert weights is not None, "edge_weights should be set before calling score_intercepts_for_strip"
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
    return intercept_scores, intercept_bins


def get_bin_idx(bins: np.ndarray, x: float):
    """Find the bin index containing a given value.

    Args:
        bins: Array of bin boundaries
        x: Value to find the bin for

    Returns:
        Index of the bin containing x, or None if x is outside the range
    """
    if x is not None and (x >= bins[0] and x < bins[-1]):
        # Find the bin containing x (the first bin with right boundary greater than it).
        bin_idx = np.argmax(bins[1:] > x)
        return bin_idx
    return None


def intercept_of_line_touching_image_edge(
    strip: StripData,
    slope: float,
) -> float | None:
    """Calculate the intercept of a line that touches the image boundary.

    This function determines where a line with the given slope would intersect
    the strip if it were positioned at the boundary of the original image.
    This is used to boost edge detection for photos that extend to the image edge.

    The coordinate system used is strip-local coordinates, where:
    - For horizontal strips: x ranges from -strip_width/2 to +strip_width/2
    - For vertical strips: y ranges from -strip_height/2 to +strip_height/2

    Args:
        strip: The strip data containing coordinate transformations
        slope: The slope of the line in strip coordinates

    Returns:
        The intercept value where the line touches the image boundary,
        or None if the strip doesn't touch the image boundary
    """
    image_height, image_width = strip.image_height, strip.image_width
    strip_height, strip_width = strip.pixels.shape[:2]
    if strip.position == StripPosition.TOP:
        strip_x_min = -strip_width / 2
        strip_x_max = strip_width / 2
        border_pt1 = strip.image_to_strip(np.array([(0, 0.0)])).flatten()
        border_pt2 = strip.image_to_strip(np.array([(image_width - 1, 0.0)])).flatten()
        bounded_above = True
    elif strip.position == StripPosition.BOTTOM:
        strip_x_min = -strip_width / 2
        strip_x_max = strip_width / 2
        border_pt1 = strip.image_to_strip(np.array([(image_width - 1, 0.0)])).flatten()
        border_pt2 = strip.image_to_strip(
            np.array([(image_width - 1, image_height - 1)])
        ).flatten()
        bounded_above = False
    elif strip.position == StripPosition.LEFT:
        strip_x_min = -strip_height / 2
        strip_x_max = strip_height / 2
        border_pt1 = strip.image_to_strip(np.array([(0.0, 0.0)])).flatten()
        border_pt2 = strip.image_to_strip(np.array([(0.0, image_height - 1)])).flatten()
        # Transpose coordinates, treat strip as horizontal.
        border_pt1 = border_pt1[::-1]
        border_pt2 = border_pt2[::-1]
        bounded_above = True
    elif strip.position == StripPosition.RIGHT:
        strip_x_min = -strip_height / 2
        strip_x_max = strip_height / 2
        border_pt1 = strip.image_to_strip(np.array([(image_width - 1, 0.0)])).flatten()
        border_pt2 = strip.image_to_strip(
            np.array([(image_width - 1, image_height - 1)])
        ).flatten()
        # Transpose coordinates, treat strip as horizontal.
        border_pt1 = border_pt1[::-1]
        border_pt2 = border_pt2[::-1]
        bounded_above = False

    return intercept_for_line_bounded_by_edge(
        border_pt1,
        border_pt2,
        slope,
        strip_x_min,
        strip_x_max,
        bounded_above=bounded_above,
    )


def intercept_to_image_points(
    strip: StripData, slope: float, intercept: float
) -> FloatArray:
    """Convert a line (slope, intercept) to two points in image coordinates.

    This function takes a line defined by slope and intercept in strip
    coordinates and converts it to two endpoint coordinates in the
    original image coordinate system.

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
    pt1_image = strip.strip_to_image(pt1.reshape((1, -1)))
    pt2_image = strip.strip_to_image(pt2.reshape((1, -1)))
    return np.concatenate([pt1_image, pt2_image], axis=0)


def boost_boundary_edge_scores(strips: dict[StripPosition, StripData], slope: float):
    """Boost intercept scores for edges that align with image boundaries.

    For strips that touch the border of the image, this function adds a "bonus"
    score to intercepts that align with the image boundary. This handles cases
    where the photo extends to the edge and its real edge isn't visible.

    Args:
        strips: Dictionary mapping strip positions to their data
        slope: The slope of the line being tested

    Side Effects:
        Modifies strip.intercept_scores arrays by adding bonus scores
        to intercepts that align with image boundaries.
    """
    median_peak_score = np.median(  # TODO: can/should this just be a constant?
        [np.max(strip.intercept_scores) for strip in strips.values() if strip.intercept_scores is not None]
    )
    for strip in strips.values():
        assert strip.intercept_scores is not None, "intercept_scores should be set before calling boost_boundary_edge_scores"
        assert strip.intercept_bins is not None, "intercept_bins should be set before calling boost_boundary_edge_scores"
        
        image_edge_intercept = intercept_of_line_touching_image_edge(strip, slope)
        if image_edge_intercept is not None:
            image_edge_bin_idx = get_bin_idx(strip.intercept_bins, image_edge_intercept)
            if image_edge_bin_idx is not None:
                # TODO tune the strength of edge preference here.
                print(
                    f"strip touched image border, boosting intercept {strip.intercept_bins[image_edge_bin_idx]}"
                )
                strip.intercept_scores[image_edge_bin_idx] += (
                    median_peak_score * IMAGE_EDGE_BOOST_FRACTION
                )


def extract_candidate_peaks(
    strips: dict[StripPosition, StripData],
    candidate_angle: float,
    max_num_peaks: int,
    debug_dir: str | None = None,
):
    """Extract the most prominent peaks from intercept scores for each strip.

    This function identifies the highest-scoring intercepts for each strip
    and stores them in the strip's candidate dictionaries. It also handles
    debug output if requested.

    Args:
        strips: Dictionary mapping strip positions to their data
        candidate_angle: The angle (in radians) being processed
        max_num_peaks: Maximum number of peaks to extract per strip
        debug_dir: Optional directory for saving debug histograms

    Side Effects:
        Modifies each strip's candidate_intercepts and candidate_intercept_scores
        dictionaries for the given angle.
    """
    for position, strip in strips.items():
        assert strip.intercept_scores is not None, "intercept_scores should be set before calling extract_candidate_peaks"
        assert strip.intercept_bins is not None, "intercept_bins should be set before calling extract_candidate_peaks"
        
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


def convert_intercepts_to_edges(
    strips: dict[StripPosition, StripData],
    candidate_angle: float,
    slope: float,
):
    """Convert intercept values to actual edge lines in image coordinates.

    This function takes the candidate intercepts found for each strip and
    converts them to edge line coordinates in the original image space.

    Args:
        strips: Dictionary mapping strip positions to their data
        candidate_angle: The angle (in radians) being processed
        slope: The slope of the line (tangent of the angle)

    Side Effects:
        Modifies each strip's candidate_edges dictionary for the given angle.
    """
    for strip in strips.values():
        strip.candidate_edges[candidate_angle] = [
            intercept_to_image_points(strip, slope=slope, intercept=float(intercept))
            for intercept in strip.candidate_intercepts[candidate_angle]
        ]


def get_candidate_edges(
    strips: dict[StripPosition, StripData],
    candidate_angle,
    max_num_peaks: int = 1,
    debug_dir=None,
):
    """Generate candidate edges for all strips at a given angle.

    This function orchestrates the edge detection process by:
    1. Scoring all possible intercepts using a Hough-like voting scheme
    2. Boosting intercepts that align with the image boundary
    3. Finding the most prominent peaks in the intercept scores
    4. Converting intercepts to actual edge lines in image coordinates

    Args:
        strips: Dictionary mapping strip positions to their data
        candidate_angle: The angle (in radians) to test for edges
        max_num_peaks: Maximum number of intercept peaks to keep per strip
        debug_dir: Optional directory for saving debug histograms

    Side Effects:
        Modifies each strip's candidate_intercepts, candidate_intercept_scores,
        and candidate_edges dictionaries for the given angle.
    """
    slope = np.tan(candidate_angle)
    print("candidate angle", candidate_angle, "slope", slope)

    # For each strip, compute the set of candidate edge intercepts and their scores.
    for strip in strips.values():
        strip.intercept_scores, strip.intercept_bins = score_intercepts_for_strip(
            strip, slope=slope
        )

    # Boost intercepts that align with image boundaries
    boost_boundary_edge_scores(strips, slope)

    # Extract the most prominent peaks from intercept scores
    extract_candidate_peaks(strips, candidate_angle, max_num_peaks, debug_dir)

    # Convert intercept values to actual edge lines in image coordinates
    convert_intercepts_to_edges(strips, candidate_angle, slope)


def generate_candidate_index_combinations(
    strips: dict[StripPosition, StripData], angle: float
):
    """Generate all possible combinations of candidate indices for a given angle.

    This helper function creates the Cartesian product of all candidate indices
    for each strip at a specific angle. It separates the nested loop logic from
    the main hypothesis enumeration.

    Args:
        strips: Dictionary mapping strip positions to their data
        angle: The angle for which to generate index combinations

    Yields:
        Tuple of (top_idx, bottom_idx, left_idx, right_idx) representing
        one combination of candidate indices across all four strips
    """
    for top_idx in range(len(strips[StripPosition.TOP].candidate_intercepts[angle])):
        for bottom_idx in range(
            len(strips[StripPosition.BOTTOM].candidate_intercepts[angle])
        ):
            for left_idx in range(
                len(strips[StripPosition.LEFT].candidate_intercepts[angle])
            ):
                for right_idx in range(
                    len(strips[StripPosition.RIGHT].candidate_intercepts[angle])
                ):
                    yield top_idx, bottom_idx, left_idx, right_idx


def enumerate_hypotheses(strips: dict[StripPosition, StripData]):
    """Generate all possible combinations of edges from the candidate sets.

    This function creates the Cartesian product of all candidate edges,
    where each hypothesis consists of one edge from each of the four strips.
    For each combination, it yields both the edges and their associated scores.

    The number of hypotheses grows as the product of the number of candidates
    in each strip, so with max_num_peaks=2, this generates up to 2^4 = 16 hypotheses.

    Args:
        strips: Dictionary mapping strip positions to their data

    Yields:
        Tuple of (edges_dict, scores_list) where:
        - edges_dict: Maps StripPosition to edge line coordinates
        - scores_list: List of scores for each edge in the combination
    """
    angles = strips[StripPosition.TOP].candidate_intercepts.keys()
    for angle in angles:
        for (
            top_idx,
            bottom_idx,
            left_idx,
            right_idx,
        ) in generate_candidate_index_combinations(strips, angle):
            edges = {
                StripPosition.TOP: strips[StripPosition.TOP].candidate_edges[angle][
                    top_idx
                ],
                StripPosition.BOTTOM: strips[StripPosition.BOTTOM].candidate_edges[
                    angle
                ][bottom_idx],
                StripPosition.LEFT: strips[StripPosition.LEFT].candidate_edges[angle][
                    left_idx
                ],
                StripPosition.RIGHT: strips[StripPosition.RIGHT].candidate_edges[angle][
                    right_idx
                ],
            }
            scores = [
                strips[StripPosition.TOP].candidate_intercept_scores[angle][top_idx],
                strips[StripPosition.BOTTOM].candidate_intercept_scores[angle][
                    bottom_idx
                ],
                strips[StripPosition.LEFT].candidate_intercept_scores[angle][left_idx],
                strips[StripPosition.RIGHT].candidate_intercept_scores[angle][
                    right_idx
                ],
            ]
            print(
                f"proposing hypothesis w angle {angle} idx {(top_idx, bottom_idx, left_idx, right_idx)} edge score {sum(scores)}"
            )
            yield edges, scores


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
    """Refine bounding box corners using Hough transform-based edge detection.

    This is the main entry point for the Hough-based refinement algorithm.
    It implements a multi-stage process:

    1. Extract border strips around the initial rectangle
    2. Detect edges in each strip using gradient filters
    3. Find dominant angles using FFT analysis of edge patterns
    4. For each angle, score intercepts using Hough-like voting
    5. Generate all combinations of candidate edges
    6. Score each combination using edge strength + aspect ratio preference
    7. Return the corners of the highest-scoring rectangle

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
        resolution_scale_factor=DEFAULT_RESOLUTION_SCALE_FACTOR,
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
            assert strip.edge_weights is not None, "edge_weights should be set after detect_edges"
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
