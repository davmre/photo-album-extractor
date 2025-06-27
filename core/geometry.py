from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

# Import semantic types from photo_types for consistency
from core.photo_types import (
    BoundingBoxAny,
    FloatArray,
    QuadArray,
    TransformMatrix,
    bounding_box_as_array,
)
from numpy import ndarray

# Suppress lint errors from uppercase variable names
# ruff: noqa N806, N803


UNIT_SQUARE = np.array([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])


def dimension_bounds(rect: BoundingBoxAny) -> tuple[float, float]:
    rect = bounding_box_as_array(rect)
    width1 = np.linalg.norm(rect[1] - rect[0])
    width2 = np.linalg.norm(rect[2] - rect[3])
    height1 = np.linalg.norm(rect[3] - rect[0])
    height2 = np.linalg.norm(rect[2] - rect[1])

    return float(max(width1, width2)), float(max(height1, height2))


def is_rectangle(corners: BoundingBoxAny, tolerance: float = 1e-6) -> bool:
    """
    Check if a quadrilateral is a rectangle.
    
    A quadrilateral is a rectangle if all its angles are right angles (90 degrees).
    This is equivalent to checking that all dot products of adjacent edges are zero.
    
    Args:
        corners: Quadrilateral corners as array-like of shape (4, 2)
        tolerance: Numerical tolerance for floating point comparison
        
    Returns:
        True if the quadrilateral is a rectangle, False otherwise
    """
    corners = bounding_box_as_array(corners)
    
    # Calculate edge vectors (going around the quadrilateral)
    edges = np.array([
        corners[1] - corners[0],  # edge 0->1
        corners[2] - corners[1],  # edge 1->2
        corners[3] - corners[2],  # edge 2->3
        corners[0] - corners[3],  # edge 3->0
    ])
    
    # Check if all angles are right angles by checking dot products of adjacent edges
    for i in range(4):
        next_i = (i + 1) % 4
        dot_product = np.dot(edges[i], edges[next_i])
        if abs(dot_product) > tolerance:
            return False
    
    return True


def quad_to_unit_square_transform(quad: QuadArray) -> TransformMatrix:
    """
    Compute the homography matrix that transforms a quadrilateral to the unit square.

    Args:
        quad: List of 4 corner points [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
              Points should be ordered (typically clockwise or counter-clockwise)

    Returns:
        H: 3x3 homography matrix

    The unit square corners are:
        (0, 0) -> bottom-left
        (1, 0) -> bottom-right
        (1, 1) -> top-right
        (0, 1) -> top-left
    """

    # Convert quad points to numpy array
    src_points = np.array(quad, dtype=np.float32)

    # Define unit square corners (destination points)
    # Order should match the ordering of input quad points
    dst_points = np.array(
        [
            [0, 0],  # bottom-left
            [1, 0],  # bottom-right
            [1, 1],  # top-right
            [0, 1],  # top-left
        ],
        dtype=np.float32,
    )

    # Set up the system of equations for homography
    # For each point correspondence (xi, yi) -> (ui, vi):
    # ui*h7*xi + ui*h8*yi + ui*h9 = h1*xi + h2*yi + h3
    # vi*h7*xi + vi*h8*yi + vi*h9 = h4*xi + h5*yi + h6

    A = []
    for i in range(4):
        x, y = src_points[i]
        u, v = dst_points[i]

        # First equation: u = (h1*x + h2*y + h3) / (h7*x + h8*y + h9)
        # Rearranged: h1*x + h2*y + h3 - u*h7*x - u*h8*y - u*h9 = 0
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])

        # Second equation: v = (h4*x + h5*y + h6) / (h7*x + h8*y + h9)
        # Rearranged: h4*x + h5*y + h6 - v*h7*x - v*h8*y - v*h9 = 0
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])

    A = np.array(A)

    # Solve the homogeneous system Ah = 0 using SVD
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]  # Last row of V^T (corresponding to smallest singular value)

    # Reshape to 3x3 matrix
    H = h.reshape(3, 3)

    return H


def apply_transform(H: TransformMatrix, points: QuadArray) -> QuadArray:
    """
    Apply homography transformation to a set of points.

    Args:
        H: 3x3 homography matrix
        points: List of (x, y) coordinates or numpy array of shape (N, 2)

    Returns:
        Transformed points as numpy array of shape (N, 2)
    """
    points = np.array(points)
    if points.ndim == 1:
        points = points.reshape(1, -1)

    # Convert to homogeneous coordinates
    ones = np.ones((points.shape[0], 1))
    homogeneous_points = np.hstack([points, ones])

    # Apply transformation
    transformed_homogeneous = (H @ homogeneous_points.T).T

    # Convert back to Cartesian coordinates
    transformed_points = (
        transformed_homogeneous[:, :2] / transformed_homogeneous[:, 2:3]
    )

    return transformed_points


class PatchCoordinatesConverter:
    def __init__(self, rect: QuadArray) -> None:
        self.H = quad_to_unit_square_transform(rect)
        self.H_inv = np.linalg.inv(self.H)

    def image_to_unit_square(self, pts: QuadArray) -> QuadArray:
        pts = np.asarray(pts)
        return apply_transform(self.H, pts)

    def unit_square_to_image(self, pts: QuadArray) -> QuadArray:
        pts = np.asarray(pts)
        return apply_transform(self.H_inv, pts)


def line_intersection(
    p1: ndarray, p2: ndarray, p3: ndarray, p4: ndarray
) -> ndarray | None:
    """
    Find the intersection point of two lines using parametric form.

    Args:
        p1, p2: Two points defining the first line (tuples or lists of [x, y])
        p3, p4: Two points defining the second line (tuples or lists of [x, y])

    Returns:
        tuple: (x, y) coordinates of intersection point
        None: If lines are parallel or coincident

    Raises:
        ValueError: If any point is invalid
    """
    # Convert to numpy arrays for easier computation
    p1, p2, p3, p4 = map(np.array, [p1, p2, p3, p4])

    # Direction vectors
    d1 = p2 - p1  # Direction of line 1
    d2 = p4 - p3  # Direction of line 2

    # Check if lines are parallel (cross product = 0)
    cross_product = d1[0] * d2[1] - d1[1] * d2[0]

    if abs(cross_product) < 1e-10:  # Using small epsilon for floating point comparison
        return None  # Lines are parallel or coincident

    # Solve the system:
    # p1 + t1 * d1 = p3 + t2 * d2
    # Rearranged: t1 * d1 - t2 * d2 = p3 - p1

    # Set up the system as matrix equation: A * t = b
    # [d1[0]  -d2[0]] [t1]   [p3[0] - p1[0]]
    # [d1[1]  -d2[1]] [t2] = [p3[1] - p1[1]]

    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
    b = p3 - p1

    # Solve for parameters t1 and t2
    t = np.linalg.solve(A, b)
    t1 = t[0]

    # Calculate intersection point using either line equation
    intersection = p1 + t1 * d1

    return intersection


def line_integral_chunked(
    image: ndarray,
    start_points: ndarray,
    end_points: ndarray,
    num_samples: int = 100,
    x_spans_full_width=False,
    chunk_size=512,
) -> ndarray:
    """
    Split up vectorized line integral into chunks for cache efficiency.

    Args:
        image: 2D numpy array
        start_points: (N, 2) array of start coordinates
        end_points: (N, 2) array of end coordinates
        num_samples: number of sample points along each line
        x_spans_full_width: if True, all lines start at `x = 0` and end at
          `x = width - 1`.

    Returns:
        (N,) array of line integral values
    """
    n_lines = start_points.shape[0]

    # We could preallocate memory buffers for each intermediate quantity and
    # write a version of the algorithm that updates them in-place, but this makes
    # the code much uglier and doesn't actually improve performance noticeably
    # (malloc is not the bottleneck), so it's simpler to just call the naive function in
    # a loop.
    chunk_results = []
    for i in range(0, n_lines, chunk_size):
        chunk_results.append(
            line_integral_vectorized(
                image,
                start_points[i : i + chunk_size],
                end_points[i : i + chunk_size],
                num_samples=num_samples,
                x_spans_full_width=x_spans_full_width,
            )
        )
    return np.concatenate(chunk_results, axis=0)


def line_integral_vectorized(
    image: ndarray,
    start_points: ndarray,
    end_points: ndarray,
    num_samples: int = 100,
    x_spans_full_width=False,
) -> ndarray:
    """
    Vectorized evaluation of multiple line integrals on an image.

    Args:
        image: 2D numpy array
        start_points: (N, 2) array of start coordinates
        end_points: (N, 2) array of end coordinates
        num_samples: number of sample points along each line
        x_spans_full_width: if True, all lines start at `x = 0` and end at
          `x = width - 1`.

    Returns:
        (N,) array of line integral values
    """
    # Extract coordinates - shape (N,)
    x0, y0 = start_points[:, 0], start_points[:, 1]
    x1, y1 = end_points[:, 0], end_points[:, 1]

    if x_spans_full_width:
        # Represent x as scalar for faster performance.
        x0, x1 = x0[:1], x1[:1]  # assumed `x0 = 0`, `x1 = width - 1`

    # Generate sample points - broadcast to shape (N, num_samples)
    t = np.linspace(0, 1, num_samples)  # shape (num_samples,)
    x_coords = x0[:, None] + t[None, :] * (x1 - x0)[:, None]
    y_coords = y0[:, None] + t[None, :] * (y1 - y0)[:, None]

    # Round to nearest pixel coordinates
    x_indices = np.round(x_coords).astype(int)
    y_indices = np.round(y_coords).astype(int)

    # Create mask for valid coordinates - shape (N, num_samples)
    mask = (y_indices >= 0) & (y_indices < image.shape[0])
    if not x_spans_full_width:
        mask &= (x_indices >= 0) & (x_indices < image.shape[1])

    # Use safe indexing - replace invalid indices with 0 (any valid index)
    safe_y = np.where(mask, y_indices, 0)
    safe_x = x_indices if x_spans_full_width else np.where(mask, x_indices, 0)

    # Sample from image - advanced indexing gives shape (N, num_samples)
    sampled = image[safe_y, safe_x]

    # Zero out samples that were out of bounds
    sampled = np.where(mask, sampled, 0)

    # Sum along the samples dimension to get integral for each line
    results = np.sum(sampled, axis=1)

    return results


def line_integral_simple(image, start_point, end_point, num_samples=100):
    """
    Simple line integral using linear sampling.
    """
    x0, y0 = start_point
    x1, y1 = end_point

    # Generate sample points
    t = np.linspace(0, 1, num_samples)
    x_coords = x0 + t * (x1 - x0)
    y_coords = y0 + t * (y1 - y0)

    # Round to nearest pixel coordinates
    x_indices = np.round(x_coords).astype(int)
    y_indices = np.round(y_coords).astype(int)

    # Filter out-of-bounds indices
    mask = (
        (y_indices >= 0)
        & (y_indices < image.shape[0])
        & (x_indices >= 0)
        & (x_indices < image.shape[1])
    )

    valid_y = y_indices[mask]
    valid_x = x_indices[mask]

    sampled = image[valid_y, valid_x]

    # Sum pixel values
    total_value = np.sum(sampled)

    return total_value


def minimum_bounding_rectangle(points: BoundingBoxAny) -> tuple[QuadArray, float]:
    # Convert QuadArray to CornerPoints for internal processing
    points = bounding_box_as_array(points)
    n = points.shape[0]

    min_area = float("inf")
    best_rect = None

    for i in range(n):
        # Get edge vector
        p1 = points[i]
        p2 = points[(i + 1) % n]
        edge_vector = p2 - p1

        # Normalize edge vector to get unit vector
        length = np.linalg.norm(edge_vector)
        if length == 0:
            continue
        unit_vector = edge_vector / length

        # Perpendicular vector
        perp_vector = np.array((-unit_vector[1], unit_vector[0]))

        # Project all points onto both axes
        u_coords = np.dot(points, unit_vector)
        v_coords = np.dot(points, perp_vector)

        # Get bounding box in this coordinate system
        u_min, u_max = np.min(u_coords), np.max(u_coords)
        v_min, v_max = np.min(v_coords), np.max(v_coords)

        # Calculate area
        area = (u_max - u_min) * (v_max - v_min)

        if area < min_area:
            min_area = area
            # Convert back to original coordinate system
            # Rectangle corners in (u,v) coordinates
            rect_corners_uv = [
                (u_min, v_min),
                (u_max, v_min),
                (u_max, v_max),
                (u_min, v_max),
            ]
            # Transform back to (x,y)
            best_rect = []
            for u, v in rect_corners_uv:
                x = u * unit_vector[0] + v * perp_vector[0]
                y = u * unit_vector[1] + v * perp_vector[1]
                best_rect.append((x, y))

    return np.array(best_rect), min_area


def clockwise_corner_permutation(rect: QuadArray) -> ndarray:
    # Sort corners clockwise in *image coordinates* (y increases downwards).
    # Equivalent to counterclockwise in Cartesian coordinates.
    centroid = np.sum(rect, axis=0) / 4.0
    delta = rect - centroid
    angles = np.atan2(delta[:, 1], delta[:, 0])
    return np.argsort(angles)


def sort_clockwise(rect: BoundingBoxAny) -> QuadArray:
    rect = bounding_box_as_array(rect)
    idxs = clockwise_corner_permutation(rect)
    return rect[idxs]


def get_corner_deviations(rect1: QuadArray, rect2: QuadArray) -> list[float]:
    """Score box pair by avg distance between corresponding corner points."""
    rect1 = sort_clockwise(rect1)
    rect2 = sort_clockwise(rect2)
    return [
        float(np.linalg.norm(corner2 - corner1))
        for corner1, corner2 in zip(rect1, rect2)
    ]


def signed_distance_from_border(
    x_coords: FloatArray,
    y_coords: FloatArray,
    border_pt1: FloatArray,
    border_pt2: FloatArray,
) -> FloatArray:
    x1, y1 = border_pt1
    x2, y2 = border_pt2

    a1 = (y2 - y1) * x_coords
    a2 = (x2 - x1) * y_coords
    a3 = x2 * y1
    a4 = y2 * x1
    n = np.linalg.norm(border_pt2 - border_pt1)
    # Combining a1 (x_coords.shape) and a2 (y_coords.shape) will materialize a big
    # array at their broadcast shape, so group terms to delay that as long as possible.
    return a1 / n + (-a2 + a3 - a4) / n


def image_boundary_mask(
    image_shape: tuple[int, ...],
    patch_shape: tuple[int, ...],
    image_to_patch_coords: Callable[[FloatArray], FloatArray],
    offset: int = -1,
) -> npt.NDArray[np.bool]:
    img_height, img_width = image_shape[:2]
    top_border = image_to_patch_coords(np.array([(0, 0), (img_width - 1, 0)]))
    bottom_border = image_to_patch_coords(
        np.array([(0, img_height - 1), (img_width - 1, img_height - 1)])
    )
    left_border = image_to_patch_coords(np.array([(0, 0), (0, img_height - 1)]))
    right_border = image_to_patch_coords(
        np.array([(img_width - 1, 0), (img_width - 1, img_height - 1)])
    )

    strip_height, strip_width = patch_shape[:2]
    x_coords = np.arange(strip_width, dtype=np.float32)[None, :]
    y_coords = np.arange(strip_height, dtype=np.float32)[:, None]

    top_border_dist = signed_distance_from_border(
        x_coords, y_coords, top_border[0, :], top_border[1, :]
    )
    bottom_border_dist = signed_distance_from_border(
        x_coords, y_coords, bottom_border[0, :], bottom_border[1, :]
    )
    left_border_dist = signed_distance_from_border(
        x_coords, y_coords, left_border[0, :], left_border[1, :]
    )
    right_border_dist = signed_distance_from_border(
        x_coords, y_coords, right_border[0, :], right_border[1, :]
    )

    mask = (
        (top_border_dist < offset)
        & (bottom_border_dist > -offset)
        & (left_border_dist > -offset)
        & (right_border_dist < offset)
    )

    return mask
