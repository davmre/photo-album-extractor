# Suppress lint errors from uppercase variable names
# ruff: noqa N806, N803

from __future__ import annotations

import logging
import os
import pathlib
from typing import Callable

import core.geometry as geometry
import core.images as images
import cv2
import numpy as np
from core.photo_types import (
    BoundingBoxAny,
    FloatArray,
    QuadArray,
    UInt8Array,
    bounding_box_as_array,
)
from PIL import Image

from photo_detection.refine_bounds import annotate_image, save_image

LOGGER = logging.getLogger("logger")


# Strip-based edge detection implementation


class StripData:
    """Container for a border strip and its coordinate transformations."""

    def __init__(
        self,
        pixels: UInt8Array,
        edge_response: FloatArray,
        image_to_strip_transform: Callable[[FloatArray], FloatArray],
        strip_to_image_transform: Callable[[FloatArray], FloatArray],
    ) -> None:
        self.pixels = pixels
        self.edge_response = edge_response
        self.image_to_strip = image_to_strip_transform
        self.strip_to_image = strip_to_image_transform


def detect_edges_sobel(pixels: UInt8Array, horizontal: bool = True) -> FloatArray:
    # Convert to numpy array and compute edge response
    gray = cv2.cvtColor(np.array(pixels), cv2.COLOR_RGB2GRAY)

    if horizontal:
        # Horizontal edges - use vertical Sobel
        edge_response = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5))
    else:  # left, right
        # Vertical edges - use horizontal Sobel
        edge_response = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5))

    # Apply same preprocessing as original algorithm
    edge_response = np.sqrt(edge_response)
    edge_response = cv2.GaussianBlur(edge_response, (5, 5), 0)
    return edge_response  # type: ignore[return-value]


def detect_edges_sharp(
    pixels: UInt8Array, horizontal: bool = True, debug_dir: str | None = None
) -> FloatArray:
    # Convert to numpy array and compute edge response
    gray = np.asarray(cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY), dtype=np.float32)

    filter = np.array(
        [[-1, -2, -1], [-1, -2, -1], [1, 2, 1], [1, 2, 1]], dtype=np.float32
    )

    if not horizontal:
        filter = np.array(filter.T)

    convolved = cv2.filter2D(gray, -1, filter)  # type: ignore
    edge_response = np.sqrt(np.abs(convolved))
    # edge_response = cv2.GaussianBlur(edge_response, (5, 5), 0)
    return edge_response


def detect_edges_color(
    pixels: UInt8Array, horizontal: bool = True, mask: FloatArray | None = None
) -> FloatArray:
    # Convert to numpy array and compute edge response
    if pixels.shape[-1] != 3:
        raise ValueError(f"Expected color image, got array of shape {pixels.shape}")
    r, g, b = [np.astype(pixels[..., i], np.float32) for i in range(3)]

    filter = np.array(
        [[-1, -2, -1], [-1, -2, -1], [1, 2, 1], [1, 2, 1]], dtype=np.float32
    )

    if not horizontal:
        filter = np.array(filter.T)

    cr, cg, cb = [cv2.filter2D(channel, -1, filter) for channel in [r, g, b]]
    edge_response = np.sqrt(np.linalg.norm([cr, cg, cb], axis=0))
    if mask is not None:
        edge_response *= mask
    edge_response = cv2.GaussianBlur(edge_response, (5, 5), 0)
    return edge_response


def extract_border_strips(
    image: Image.Image | UInt8Array,
    rect: QuadArray,
    reltol: float,
    resolution_scale_factor: float = 1.0,
    min_image_pixels: int = 8,
    debug_dir: str | None = None,
) -> dict[str, StripData]:
    """Extract four border strips from the image."""
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
        del image
    else:
        pil_image = image

    image_shape = (pil_image.height, pil_image.width)

    # Ensure rect is sorted clockwise
    rect = geometry.sort_clockwise(rect)
    width, height = geometry.dimension_bounds(rect)

    reltol_x = max(reltol, min_image_pixels / width)
    reltol_y = max(reltol, min_image_pixels / height)

    strips = {}

    # Define normalized coordinates for each strip in the unit square
    # Top strip: full width, top portion
    top_strip_normalized = np.array(
        [
            [0 - reltol_x, -reltol_y],
            [1 + reltol_x, -reltol_y],
            [1 + reltol_x, reltol_y],
            [0 - reltol_x, reltol_y],
        ]
    )

    # Bottom strip: full width, bottom portion
    bottom_strip_normalized = np.array(
        [
            [0 - reltol_x, 1 - reltol_y],
            [1 + reltol_x, 1 - reltol_y],
            [1 + reltol_x, 1 + reltol_y],
            [0 - reltol_x, 1 + reltol_y],
        ]
    )

    # Left strip: left portion, full height
    left_strip_normalized = np.array(
        [
            [-reltol_x, 0 - reltol_y],
            [reltol_x, 0 - reltol_y],
            [reltol_x, 1 + reltol_y],
            [-reltol_x, 1 + reltol_y],
        ]
    )

    # Right strip: right portion, full height
    right_strip_normalized = np.array(
        [
            [1 - reltol_x, 0 - reltol_y],
            [1 + reltol_x, 0 - reltol_y],
            [1 + reltol_x, 1 + reltol_y],
            [1 - reltol_x, 1 + reltol_y],
        ]
    )

    # Convert to image coordinates and extract each strip
    converter = geometry.PatchCoordinatesConverter(rect)

    for name, strip_normalized in [
        ("top", top_strip_normalized),
        ("bottom", bottom_strip_normalized),
        ("left", left_strip_normalized),
        ("right", right_strip_normalized),
    ]:
        # Convert normalized coords to image coords
        strip_corners_image = converter.unit_square_to_image(strip_normalized)
        strip_corners_image = np.round(strip_corners_image)
        width, height = geometry.dimension_bounds(strip_corners_image)
        strip_width = int(np.ceil(width * resolution_scale_factor))
        strip_height = int(np.ceil(height * resolution_scale_factor))

        # Extract the strip
        LOGGER.debug(
            f"extracting strip {name} with corners {strip_corners_image} image width {width} height {height} strip width {strip_width} height {strip_height}"
        )
        strip_pixels = images.extract_perspective_image(
            pil_image,
            strip_corners_image,
            output_width=strip_width,
            output_height=strip_height,
            mode=Image.Resampling.BICUBIC,
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

        strip_mask = geometry.image_boundary_mask(
            image_shape=image_shape,
            patch_shape=pixels_array.shape,
            image_to_patch_coords=image_to_strip_transform,
        )

        edge_response = detect_edges_color(
            pixels_array, horizontal=(name in ["top", "bottom"]), mask=strip_mask
        )

        # Store strip data
        strips[name] = StripData(
            pixels=pixels_array,
            edge_response=edge_response,
            image_to_strip_transform=image_to_strip_transform,
            strip_to_image_transform=strip_to_image_transform,
        )

    return strips


def evaluate_edges_at_angle(strip, angle, strip_type, debug_dir=None):
    """Evaluate all possible edge positions at given angle in a strip."""
    if debug_dir is not None:
        debug_dir = os.path.join(debug_dir, "candidates")
        pathlib.Path(debug_dir).mkdir(parents=True, exist_ok=True)

    h, w = strip.edge_response.shape

    if strip_type in ["top", "bottom"]:
        # Horizontal strips - edges go left to right with slight angle
        ys = np.arange(h)
        edge_pts1 = np.column_stack((np.zeros_like(ys), ys - w / 2 * np.tan(angle)))
        edge_pts2 = np.column_stack(
            (np.full_like(ys, w - 1), ys + w / 2 * np.tan(angle))
        )
    else:  # left or right
        # Vertical strips - edges go top to bottom with slight angle
        xs = np.arange(w)
        edge_pts1 = np.column_stack((xs + h / 2 * np.tan(angle), np.zeros_like(xs)))
        edge_pts2 = np.column_stack(
            (xs - h / 2 * np.tan(angle), np.full_like(xs, h - 1))
        )

    scores = geometry.line_integral_vectorized(
        strip.edge_response, edge_pts1, edge_pts2
    )
    return scores


def search_rectangle_factored(strips, initial_rect, debug_dir=None):
    """Search for best rectangle using factored approach."""
    # Compute initial angle
    edge_vector = initial_rect[1] - initial_rect[0]
    base_angle = np.arctan2(edge_vector[1], edge_vector[0])

    # Use enough angles that we can hit any pixel on the edge of the strip,
    # and ensure that the number is odd so that we always consider angle==0.
    num_angles = strips["top"].edge_response.shape[0]
    num_angles = num_angles + 1 if (num_angles % 2 == 0) else num_angles

    # Limit angle search to ensure edges span strips
    # Maximum angle where edges still span the strip
    h, w = strips["top"].edge_response.shape
    max_angle_deviation = np.arctan(h / w)

    strip_angles = np.linspace(-max_angle_deviation, max_angle_deviation, num_angles)

    best_score = -np.inf
    best_edges = None
    best_angle = None

    for angle in strip_angles:
        # Evaluate all edges at this angle
        edge_scores = {}

        # Top and bottom edges are at angle θ
        edge_scores["top"] = evaluate_edges_at_angle(
            strips["top"], angle, "top", debug_dir=debug_dir
        )
        edge_scores["bottom"] = evaluate_edges_at_angle(
            strips["bottom"], angle, "bottom"
        )

        # Left and right edges are perpendicular (angle θ + π/2)
        edge_scores["left"] = evaluate_edges_at_angle(strips["left"], angle, "left")
        edge_scores["right"] = evaluate_edges_at_angle(strips["right"], angle, "right")

        score = (
            np.max(edge_scores["top"])
            + np.max(edge_scores["bottom"])
            + np.max(edge_scores["left"])
            + np.max(edge_scores["right"])
        )

        edges = {
            "top": np.argmax(edge_scores["top"]),
            "bottom": np.argmax(edge_scores["bottom"]),
            "left": np.argmax(edge_scores["left"]),
            "right": np.argmax(edge_scores["right"]),
        }

        if score > best_score:
            best_score = score
            best_edges = edges
            best_angle = base_angle + angle

        LOGGER.debug(f"angle {angle:.3f}: best score {score:.1f}")

    return best_edges, best_angle, best_score


def find_corner_intersections(edges):
    # Find corner intersections
    upper_left = geometry.line_intersection(
        edges["top"][0], edges["top"][1], edges["left"][0], edges["left"][1]
    )
    upper_right = geometry.line_intersection(
        edges["top"][0], edges["top"][1], edges["right"][0], edges["right"][1]
    )
    lower_right = geometry.line_intersection(
        edges["bottom"][0], edges["bottom"][1], edges["right"][0], edges["right"][1]
    )
    lower_left = geometry.line_intersection(
        edges["bottom"][0], edges["bottom"][1], edges["left"][0], edges["left"][1]
    )
    return np.array([upper_left, upper_right, lower_right, lower_left])


def get_edges_as_image_coordinates(edge_indices, strips, angle_offset, debug_dir=None):
    """Convert edge indices to image coordinates."""
    # Get the edge lines in strip coordinates
    edges_strip = {}

    # Top edge
    top_y = edge_indices["top"]
    _, top_w = strips["top"].edge_response.shape
    top_y0 = top_y - top_w / 2 * np.tan(angle_offset)
    top_y1 = top_y + top_w / 2 * np.tan(angle_offset)
    edges_strip["top"] = (np.array([0, top_y0]), np.array([top_w - 1, top_y1]))

    # Bottom edge
    bottom_y = edge_indices["bottom"]
    bottom_y0 = bottom_y - top_w / 2 * np.tan(angle_offset)
    bottom_y1 = bottom_y + top_w / 2 * np.tan(angle_offset)
    edges_strip["bottom"] = (np.array([0, bottom_y0]), np.array([top_w - 1, bottom_y1]))

    # Left edge
    left_x = edge_indices["left"]
    left_h, _ = strips["left"].edge_response.shape
    left_x0 = left_x + left_h / 2 * np.tan(angle_offset)
    left_x1 = left_x - left_h / 2 * np.tan(angle_offset)
    edges_strip["left"] = (np.array([left_x0, 0]), np.array([left_x1, left_h - 1]))

    # Right edge
    right_x = edge_indices["right"]
    right_x0 = right_x + left_h / 2 * np.tan(angle_offset)
    right_x1 = right_x - left_h / 2 * np.tan(angle_offset)
    edges_strip["right"] = (np.array([right_x0, 0]), np.array([right_x1, left_h - 1]))

    if debug_dir:
        for name, strip in strips.items():
            edge = edges_strip[name]
            save_image(
                os.path.join(debug_dir, f"edge_{name}.png"),
                annotate_image(strip.edge_response, edges=[np.array(edge, dtype=int)]),
            )

    # Convert to image coordinates
    edges_image = {}
    for name, (pt1, pt2) in edges_strip.items():
        strip = strips[name]
        edges_image[name] = (
            strip.strip_to_image(pt1.reshape(1, -1))[0],
            strip.strip_to_image(pt2.reshape(1, -1))[0],
        )

    return edges_image


def search_best_edge(strip, edge_is_horizontal=True):
    h, w = strip.edge_response.shape

    # Generate candidate edge endpoints in strip coordinates
    if edge_is_horizontal:
        # Horizontal edge - spans width
        y_values = np.arange(h)
        best_score = -np.inf
        best_edge = None

        for y in y_values:
            strip_pt1 = np.array([0, y])
            strip_pt2 = np.array([w - 1, y])
            score = geometry.line_integral_simple(
                strip.edge_response, strip_pt1, strip_pt2
            )
            if score > best_score:
                best_score = score
                # Convert to image coordinates
                best_edge = (
                    strip.strip_to_image(strip_pt1.reshape(1, -1))[0],
                    strip.strip_to_image(strip_pt2.reshape(1, -1))[0],
                )
    else:  # left, right
        # Vertical edge - spans height
        x_values = np.arange(w)
        best_score = -np.inf
        best_edge = None

        for x in x_values:
            strip_pt1 = np.array([x, 0])
            strip_pt2 = np.array([x, h - 1])
            score = geometry.line_integral_simple(
                strip.edge_response, strip_pt1, strip_pt2
            )
            if score > best_score:
                best_score = score
                # Convert to image coordinates
                best_edge = (
                    strip.strip_to_image(strip_pt1.reshape(1, -1))[0],
                    strip.strip_to_image(strip_pt2.reshape(1, -1))[0],
                )

    return best_edge


def refine_bounding_box_strips(
    image: Image.Image | UInt8Array,
    corner_points: BoundingBoxAny,
    reltol: float = 0.05,
    resolution_scale_factor: float = 1.0,
    enforce_parallel_sides: bool = False,
    debug_dir: str | None = None,
) -> QuadArray:
    """Refine bounding box using strip-based edge detection."""
    corner_points = bounding_box_as_array(corner_points)
    if debug_dir is not None:
        pathlib.Path(debug_dir).mkdir(parents=True, exist_ok=True)
        LOGGER.info(f"logging to debug dir {debug_dir}")
        save_image(
            os.path.join(debug_dir, "init.png"), annotate_image(image, [corner_points])
        )

    # Get minimum bounding rectangle
    LOGGER.debug(f"refining initial corner points {corner_points}")
    rect, _ = geometry.minimum_bounding_rectangle(corner_points)
    rect = geometry.sort_clockwise(rect)
    LOGGER.debug(f"bounding rect {rect}")

    # Extract border strips

    """
    
    how to extract border strips?
    
    first general question about resolution
    I think we want a mode where there's no downscaling and we work with
    image pixels themselves, as much as we can
    this means we get different sizes for different strips. but that's
    okay because they all have their own coords.
    
    but we also want coarse modes. when coarsening, probably we don't want
    even downscaling. for a 1000 x 10 rectangle, downscaling by 100 still lets
    you resolve the long edge, but it turns the short edge into a single pixel!
    we want more resolution on the shorter edge. both in order to resolve it,
    and probably it's good to give it voting power? so I think it's fine to
    have a single resolution for both edges. otoh, maybe this doesn't matter
    at all bc we'll do coarse-to-fine so we can just do a fixed downscaling
    and we'll resolve the short edge better at the finer resolutions
    
    this argues for a fixed downscaling factor? 
    
    okay the other question is generally about reltol
    do we want a fixed reltol? 
    
    positionally maybe we want tolerance relative to the image as a whole
    if we're fitting a `1000 x 1` rectangle, our initial bounding box might even
    be a few pixels off and not touch the original rectangle at all.
    
    suppose we have an axis-aligned bounding box of width [w] and height [w]
    and we *think* the photo is at some angle between (`[-tol, tol]`)
    how much do we need to expand the width and height to be sure to capture
    the edges?
    relative to center of rectangle
    
    
    
    extreme cases:
    1. we are given an axis-aligned initial square with `resolution`
    
    
    """

    strips = extract_border_strips(
        image,
        rect,
        reltol=reltol,
        resolution_scale_factor=resolution_scale_factor,
        debug_dir=debug_dir,
    )

    # Apply boundary masking to each strip
    # TODO: Implement boundary masking for strips

    # Save debug images if requested
    if debug_dir:
        for name, strip in strips.items():
            save_image(
                os.path.join(debug_dir, f"strip_{name}.png"), strip.pixels[:, :, ::-1]
            )
            save_image(
                os.path.join(debug_dir, f"edge_response_{name}.png"),
                strip.edge_response,
            )

    if enforce_parallel_sides:
        # Use factored search for efficiency
        best_edges_strip, best_angle, best_score = search_rectangle_factored(
            strips, rect, debug_dir=debug_dir
        )

        LOGGER.debug(f"best edges: {best_edges_strip}")
        LOGGER.debug(f"best rectangle score: {best_score}")
        LOGGER.debug(f"best angle: {best_angle}")

        # Convert edge indices to actual corner points
        best_edges = get_edges_as_image_coordinates(
            best_edges_strip,
            strips,
            best_angle - np.arctan2(rect[1][1] - rect[0][1], rect[1][0] - rect[0][0]),
            debug_dir=debug_dir,
        )
    else:
        # Find best edge in each strip independently
        best_edges = {}
        for edge_name in ["top", "bottom", "left", "right"]:
            best_edges[edge_name] = search_best_edge(
                strips[edge_name], edge_is_horizontal=(edge_name in ["top", "bottom"])
            )
            LOGGER.debug(f"best {edge_name} edge: {best_edges[edge_name]}")

    LOGGER.debug(f"got best edges {best_edges}")
    corners = find_corner_intersections(best_edges)
    LOGGER.debug(f"returning refined corners {corners}")
    if debug_dir is not None:
        save_image(
            os.path.join(debug_dir, "result.png"), annotate_image(image, [corners])
        )
    return corners


def refine_bounding_box_strips_multiscale(
    image: UInt8Array,
    corner_points: BoundingBoxAny,
    reltol: float = 0.05,
    base_resolution: int = 200,
    scale_step: int = 4,
    enforce_parallel_sides: bool = False,
    debug_dir: str | None = None,
) -> QuadArray:
    """Coarse-to-fine refinement using strip-based edge detection."""
    corner_points = bounding_box_as_array(corner_points)

    outer_resolution = int(max(geometry.dimension_bounds(corner_points)))

    scale_factors = [base_resolution / outer_resolution]
    reltols = [reltol]

    new_scale_factor = scale_factors[-1]

    while new_scale_factor < 1.0:
        new_scale_factor = min(1.0, new_scale_factor * scale_step)
        scale_factors.append(new_scale_factor)
        reltols.append(
            min(reltol / 2.0, scale_step / (new_scale_factor * base_resolution))
        )

    scale_factors[-1] = 1.0

    LOGGER.debug("SCALE FACTORS", scale_factors)
    LOGGER.debug("RELTOLS", reltols)

    for reltol, scale_factor in zip(reltols, scale_factors):
        debug_subdir = None
        if debug_dir:
            debug_subdir = os.path.join(
                debug_dir, f"strips_{scale_factor}_{reltol:.5f}"
            )

        corner_points = refine_bounding_box_strips(
            image,
            corner_points,
            reltol=reltol,
            resolution_scale_factor=scale_factor,
            enforce_parallel_sides=enforce_parallel_sides,
            debug_dir=debug_subdir,
        )
        LOGGER.debug(
            f"strips: resolution {scale_factor} reltol {reltol} corner_points {corner_points}"
        )

    return corner_points
