# Suppress lint errors from uppercase variable names
# ruff: noqa N806, N803

from __future__ import annotations

import logging
import os
import pathlib

import core.geometry as geometry
import core.images as images
import cv2
import numpy as np
import PIL.Image
from core.photo_types import (
    AnyArray,
    BoundingBoxAny,
    FloatArray,
    IntArray,
    QuadArray,
    UInt8Array,
    bounding_box_as_array,
)
from PIL import Image

LOGGER = logging.getLogger("logger")


def _candidate_edge_points(
    patch_n: int, border_n: int
) -> tuple[IntArray, IntArray, IntArray, IntArray]:
    """Enumerate potential endpoints for each of the four image borders.

    For each border, we consider N potential start points and N potential end
    points, where N is the number of pixels within a range `border_n` of the
    original bounding box. Each pair of points defines a candidate border line.

    Each returned array `pts` contains potential start points as `pts[0, :, :]`
    and end points as `pts[1, :, :]`. The start and end points are always on the
    border of the extracted image patch. For example, to search for the top
    border of the image, the start points will be on the (top side of the) left
    edge, and the end points will be on the (top side of the) right edge.
    """
    N = 2 * border_n
    initial_range = np.arange(0, N)
    final_range = np.arange(patch_n - N, patch_n)

    # Top border
    top_pts = np.zeros([2, N, 2], dtype=int)
    top_pts[0, :, 0] = 0
    top_pts[1, :, 0] = patch_n - 1
    top_pts[:, :, 1] = initial_range

    # Bottom border
    bottom_pts = np.zeros([2, N, 2], dtype=int)
    bottom_pts[0, :, 0] = 0
    bottom_pts[1, :, 0] = patch_n - 1
    bottom_pts[:, :, 1] = final_range

    # Left border
    left_pts = np.zeros([2, N, 2], dtype=int)
    left_pts[:, :, 0] = initial_range
    left_pts[0, :, 1] = 0
    left_pts[1, :, 1] = patch_n - 1

    # Right border
    right_pts = np.zeros([2, N, 2], dtype=int)
    right_pts[:, :, 0] = final_range
    right_pts[0, :, 1] = 0
    right_pts[1, :, 1] = patch_n - 1

    return top_pts, bottom_pts, left_pts, right_pts


def get_best_edge(edge_weights: FloatArray, pts: IntArray) -> tuple[IntArray, float]:
    # Get the edge point pairs that maximize the Sobel response.
    left_idx, right_idx = np.unravel_index(edge_weights.argmax(), edge_weights.shape)
    pt0 = pts[0, left_idx]
    pt1 = pts[1, right_idx]
    edge = np.asarray([pt0, pt1])
    LOGGER.debug(f"best edge {edge}")
    return edge, edge_weights[left_idx, right_idx]


def search_best_rhombus(
    top_pts,
    bottom_pts,
    left_pts,
    right_pts,
    top_edge_weights,
    bottom_edge_weights,
    left_edge_weights,
    right_edge_weights,
):
    N = top_edge_weights.shape[0]
    offsets = np.arange(-N, N)
    offset_scores = np.zeros_like(offsets)
    idxs2, idxs1 = np.meshgrid(np.arange(N), np.arange(N))
    for i, offset in enumerate(offsets):
        horizontal_offset_mask = (idxs2 - idxs1) == offset
        vertical_offset_mask = (idxs1 - idxs2) == offset

        top_score = np.max(top_edge_weights * horizontal_offset_mask)
        bottom_score = np.max(bottom_edge_weights * horizontal_offset_mask)
        left_score = np.max(left_edge_weights * vertical_offset_mask)
        right_score = np.max(right_edge_weights * vertical_offset_mask)
        offset_scores[i] = top_score + bottom_score + left_score + right_score

    best_offset_idx = np.argmax(offset_scores)
    best_offset = offsets[best_offset_idx]
    LOGGER.debug(f"best offset {best_offset}")
    # best_score = np.max(offset_scores)
    horizontal_offset_mask = (idxs2 - idxs1) == best_offset
    vertical_offset_mask = (idxs1 - idxs2) == best_offset
    top_edge, top_score = get_best_edge(
        top_edge_weights * horizontal_offset_mask, top_pts
    )
    bottom_edge, bottom_score = get_best_edge(
        bottom_edge_weights * horizontal_offset_mask, bottom_pts
    )
    left_edge, left_score = get_best_edge(
        left_edge_weights * vertical_offset_mask, left_pts
    )
    right_edge, right_score = get_best_edge(
        right_edge_weights * vertical_offset_mask, right_pts
    )

    score = top_score + bottom_score + left_score + right_score
    LOGGER.debug(f"best score {score}")
    LOGGER.debug(f"best edges {top_edge}, {bottom_edge}, {left_edge}, {right_edge}")
    return top_edge, bottom_edge, left_edge, right_edge


def annotate_image(
    img: AnyArray | PIL.Image.Image,
    contours=None,
    edges: list[IntArray] | tuple[IntArray, ...] | None = None,
) -> AnyArray:
    if isinstance(img, Image.Image):
        img = np.array(img)
    img = img.copy()
    if contours:
        cv2.drawContours(
            img,
            np.array(np.round(contours), dtype=int),  # type: ignore
            -1,
            (0, 0, 255),
            1,  # type: ignore
        )  # type: ignore
    if edges:
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 0, 255),
            (255, 255, 0),
            (0, 255, 255),
        ]
        for edge, color in zip(edges, colors):
            cv2.line(img, (edge[0, 0], edge[0, 1]), (edge[1, 0], edge[1, 1]), color, 1)
    return img


def save_image(file_path: str, img: AnyArray | PIL.Image.Image) -> None:
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    if img.mode == "F":
        # PNG supports greyscale images with 8-bit int pixels.
        img = img.convert("L")

    img.save(file_path)
    LOGGER.info(f"saved: {file_path}")


def refine_bounding_box(
    image: UInt8Array | Image.Image,
    corner_points: BoundingBoxAny,
    reltol: float = 0.05,
    resolution: int = 200,
    enforce_parallel_sides: bool = False,
    debug_dir: str | None = None,
) -> QuadArray:
    if debug_dir is not None:
        pathlib.Path(debug_dir).mkdir(parents=True, exist_ok=True)
        LOGGER.info(f"logging to debug dir {debug_dir}")
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image[:, :, ::-1])
        del image
    else:
        pil_image = image
    image_shape = (pil_image.width, pil_image.height)

    # Initial corner points represent an arbitrary quadrilateral.
    # Bound this with a (not necessarily axis-aligned) rectangle.
    # We'll do all our refinement calculations within this rectangle. Using
    # a rectangle ensures that the transformation into and out of this space
    # preserves parallel lines.
    rect, _ = geometry.minimum_bounding_rectangle(corner_points)
    LOGGER.debug(f"bounding rect {rect}")

    # Reimpose clockwise corner ordering since the previous call may lose it.
    rect = geometry.sort_clockwise(rect)

    # Expand the bounding box slightly to search for edges not contained in the
    # original box. This expansion is done in the box's own coordinate frame, by
    # mapping it to the unit square, and then inverse-mapping an expanded unit
    # square back into image coordinates, so the expansion is proportionate
    # to the bounds in each (not necessarily axis-aligned) direction.
    # TODO: does this make sense? we might actually want to do something like
    # the opposite since actual pixel movement under a rotation (eg, think of
    # rotating a long skinny rectangle around its center point) would be
    # inversely proportional to the relative side length. but it probably
    # doesn't matter much in practice since most photos are within a small
    # factor of being square.
    border_n = int(resolution * reltol)
    coordinates = geometry.PatchCoordinatesConverter(rect)
    expanded_unit_square = np.array(
        [
            (-reltol, -reltol),
            (1 + reltol, -reltol),
            (1 + reltol, 1 + reltol),
            (-reltol, 1 + reltol),
        ]
    )
    expanded_rect = coordinates.unit_square_to_image(expanded_unit_square)
    LOGGER.debug(f"expanded rect {expanded_rect}")

    # Extract the image patch within the expanded bounding box. For simplicity,
    # use the same resolution in both dimensions so the extracted patch is a
    # (probably downscaled) square.
    patch_n = resolution + border_n * 2  # Total output pixels per side.
    extracted_patch = images.extract_perspective_image(
        pil_image, expanded_rect, output_width=patch_n, output_height=patch_n
    )
    LOGGER.debug(
        f"extracted patch dims {extracted_patch.width} {extracted_patch.height}"
    )
    if debug_dir:
        patch_array = np.array(extracted_patch)[:, :, ::-1]
        border_rect = [
            (border_n * 2, border_n * 2),
            (border_n * 2, resolution),
            (resolution, resolution),
            (resolution, border_n * 2),
        ]
        patch_array = annotate_image(patch_array, [border_rect])
        save_image(os.path.join(debug_dir, "patch_with_bounds.png"), patch_array)

    # Find horizontal and vertical edges within the extracted patch
    gray = cv2.cvtColor(np.array(extracted_patch), cv2.COLOR_BGR2GRAY)
    sobel_vertical = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5))
    sobel_horizontal = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5))
    if debug_dir:
        save_image(
            os.path.join(debug_dir, "sobel_vertical.png"),
            annotate_image(sobel_vertical, [border_rect]),
        )
        save_image(
            os.path.join(debug_dir, "sobel_horizontal.png"),
            annotate_image(sobel_horizontal, [border_rect]),
        )

    # If the extracted patch includes areas outside the bounds of the original
    # image, the missing pixels will be filled as black and will create a strong
    # artificial edge corresponding to the original image bounds. This can
    # confound finding the edges of the photo within the image, so we want to
    # suppress the Sobel field when it reaches the boundaries of the original
    # scanned image.
    # (note that the photo may in fact extend outside the scanned image, but we
    # obviously can't get any useful edge information from the part we didn't
    # scan!)

    def _image_to_patch(pts):
        unit_square_points = coordinates.image_to_unit_square(pts)
        return unit_square_points * resolution + border_n

    def _patch_to_image(pts):
        pts = np.asarray(pts)
        return coordinates.unit_square_to_image((pts - border_n) / resolution)

    boundary_mask = geometry.image_boundary_mask(
        image_to_patch_coords=_image_to_patch,
        image_shape=image_shape,
        patch_shape=(patch_n, patch_n),
    )
    sobel_horizontal *= boundary_mask
    sobel_vertical *= boundary_mask
    if debug_dir:
        save_image(os.path.join(debug_dir, "boundary_mask.png"), boundary_mask)

    # Apply a square root transform to the detected edges. This gives less
    # weight to 'outliers' --- individual pixels with strong edge detections ---
    # relative to consistent lines in which every pixel has some edge potential.
    sobel_horizontal = np.sqrt(sobel_horizontal)
    sobel_vertical = np.sqrt(sobel_vertical)

    # Slightly blur the detected edges so we give 'partial credit' to candidate
    # borders that are a pixel or two off from the exact edge.
    sobel_horizontal = cv2.GaussianBlur(sobel_horizontal, (5, 5), 0)
    sobel_vertical = cv2.GaussianBlur(sobel_vertical, (5, 5), 0)
    if debug_dir:
        save_image(
            os.path.join(debug_dir, "sobel_vertical_sqrt_blur.png"),
            annotate_image(sobel_vertical, [border_rect]),
        )
        save_image(
            os.path.join(debug_dir, "sobel_horizontal_sqrt_blur.png"),
            annotate_image(sobel_horizontal, [border_rect]),
        )

    # Search for the edges that maximize the total response integrated across
    # the appropriate Sobel field (horizontal for top/bottom edges, vertical
    # for left/right edges).
    # TODO vectorize this computation?
    N = 2 * border_n
    # Propose potential edge points.
    top_pts, bottom_pts, left_pts, right_pts = _candidate_edge_points(patch_n, border_n)
    # Compute the Sobel response for each candidate edge.
    top_edge_weights = np.zeros([N, N])
    bottom_edge_weights = np.zeros([N, N])
    left_edge_weights = np.zeros([N, N])
    right_edge_weights = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            top_edge_weights[i, j] = geometry.line_integral_simple(
                sobel_horizontal, top_pts[0, i, :], top_pts[1, j, :]
            )
            bottom_edge_weights[i, j] = geometry.line_integral_simple(
                sobel_horizontal, bottom_pts[0, i, :], bottom_pts[1, j, :]
            )
            left_edge_weights[i, j] = geometry.line_integral_simple(
                sobel_vertical, left_pts[0, i, :], left_pts[1, j, :]
            )
            right_edge_weights[i, j] = geometry.line_integral_simple(
                sobel_vertical, right_pts[0, i, :], right_pts[1, j, :]
            )

    if enforce_parallel_sides:
        # Find the set of perpendicular edges that maximize total Sobel
        # response.
        top_edge, bottom_edge, left_edge, right_edge = search_best_rhombus(
            top_pts,
            bottom_pts,
            left_pts,
            right_pts,
            top_edge_weights,
            bottom_edge_weights,
            left_edge_weights,
            right_edge_weights,
        )
    else:
        # Indivually maximize the Sobel response under each edge.
        top_edge, _ = get_best_edge(top_edge_weights, top_pts)
        bottom_edge, _ = get_best_edge(bottom_edge_weights, bottom_pts)
        left_edge, _ = get_best_edge(left_edge_weights, left_pts)
        right_edge, _ = get_best_edge(right_edge_weights, right_pts)

    if debug_dir:
        annotated = annotate_image(
            np.array(extracted_patch)[:, :, ::-1],
            edges=(top_edge, bottom_edge, left_edge, right_edge),
        )
        save_image(os.path.join(debug_dir, "best_edges.png"), annotated)

    # Get corners for the refined bounding box by finding the intersections
    # of the refined edges.
    upper_left_corner = geometry.line_intersection(
        top_edge[0, :], top_edge[1, :], left_edge[0, :], left_edge[1, :]
    )
    lower_left_corner = geometry.line_intersection(
        bottom_edge[0, :], bottom_edge[1, :], left_edge[0, :], left_edge[1, :]
    )
    upper_right_corner = geometry.line_intersection(
        top_edge[0, :], top_edge[1, :], right_edge[0, :], right_edge[1, :]
    )
    lower_right_corner = geometry.line_intersection(
        bottom_edge[0, :], bottom_edge[1, :], right_edge[0, :], right_edge[1, :]
    )

    LOGGER.debug(
        f"patch rect {[upper_left_corner, upper_right_corner, lower_right_corner, lower_left_corner]}"
    )

    # Convert the corner points of our refined bounding box back into image
    # coordinates.
    image_rect = _patch_to_image(
        [upper_left_corner, upper_right_corner, lower_right_corner, lower_left_corner]
    )
    return image_rect


def refine_bounding_box_multiscale(
    image: UInt8Array | Image.Image,
    corner_points: BoundingBoxAny,
    reltol: float = 0.05,
    base_resolution: int = 200,
    scale_factor: int = 5,
    enforce_parallel_sides: bool = False,
    debug_dir: str | None = None,
) -> QuadArray:
    """Coarse-to-fine refinement of bounding box edges."""
    corner_points = bounding_box_as_array(corner_points)
    outer_resolution = int(max(geometry.dimension_bounds(corner_points)))

    resolutions = [base_resolution]
    new_resolution = base_resolution
    reltols = [reltol]
    while new_resolution < outer_resolution:
        new_resolution = min(outer_resolution, new_resolution * scale_factor)
        resolutions.append(new_resolution)
        # if we had a 200x200 image and we were pixel precise
        # now we upscale to a 1000x1000 image, so we are within 5 pixels
        # the appropriate reltol is therefore scale_factor / new_resolution
        # but increase it a bit just for some slack
        reltols.append(1.5 * scale_factor / new_resolution)

    for reltol, resolution in zip(reltols, resolutions):
        debug_subdir = None
        if debug_dir:
            debug_subdir = os.path.join(debug_dir, f"{resolution}_{reltol:.5f}")

        corner_points = refine_bounding_box(
            image,
            corner_points,
            reltol=reltol,
            resolution=resolution,
            enforce_parallel_sides=enforce_parallel_sides,
            debug_dir=debug_subdir,
        )
        LOGGER.debug(
            f"resolution {resolution} reltol {reltol} corner_points {corner_points}"
        )
    return corner_points
