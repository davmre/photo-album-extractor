import cv2
from matplotlib import pyplot as plt
import numpy as np

from PIL import Image

import image_processor

def quad_to_unit_square_transform(quad):
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
    dst_points = np.array([
        [0, 0],  # bottom-left
        [1, 0],  # bottom-right
        [1, 1],  # top-right
        [0, 1]   # top-left
    ], dtype=np.float32)

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
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])

        # Second equation: v = (h4*x + h5*y + h6) / (h7*x + h8*y + h9)
        # Rearranged: h4*x + h5*y + h6 - v*h7*x - v*h8*y - v*h9 = 0
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])

    A = np.array(A)

    # Solve the homogeneous system Ah = 0 using SVD
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]  # Last row of V^T (corresponding to smallest singular value)

    # Reshape to 3x3 matrix
    H = h.reshape(3, 3)

    return H

def apply_transform(H, points):
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
    transformed_points = transformed_homogeneous[:, :2] / transformed_homogeneous[:, 2:3]

    return transformed_points

def line_intersection(p1, p2, p3, p4):
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

    A = np.array([[d1[0], -d2[0]],
                  [d1[1], -d2[1]]])
    b = p3 - p1

    # Solve for parameters t1 and t2
    t = np.linalg.solve(A, b)
    t1 = t[0]

    # Calculate intersection point using either line equation
    intersection = p1 + t1 * d1

    return intersection


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
    mask = ((y_indices >= 0) & (y_indices < image.shape[0]) &
            (x_indices >= 0) & (x_indices < image.shape[1]))

    valid_y = y_indices[mask]
    valid_x = x_indices[mask]

    sampled = image[valid_y, valid_x]

    # Sum pixel values
    total_value = np.sum(sampled)

    return total_value


def _candidate_edge_points(patch_n, border_n):
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

def signed_distance_from_border(x_coords, y_coords, border_pt1, border_pt2):
  x1, y1 = border_pt1
  x2, y2 = border_pt2
  return ((y2 - y1) * x_coords - (x2 - x1) * y_coords + x2 * y1 - y2 * x1
          ) / np.linalg.norm(border_pt2 - border_pt1)

def image_boundary_mask(patch_n, image_to_patch_coords, img_shape):
  img_height, img_width = img_shape[:2]
  top_border = image_to_patch_coords(np.array([(0, 0), (img_width-1, 0)]))
  bottom_border = image_to_patch_coords(np.array([(0, img_height-1), (img_width-1, img_height-1)]))
  left_border = image_to_patch_coords(np.array([(0, 0), (0, img_height-1)]))
  right_border = image_to_patch_coords(np.array([(img_width-1, 0), (img_width-1, img_height-1)]))

  x_coords, y_coords = np.meshgrid(np.arange(patch_n), np.arange(patch_n))
  top_border_dist = signed_distance_from_border(
      x_coords, y_coords, top_border[0, :], top_border[1, :])
  bottom_border_dist = signed_distance_from_border(
      x_coords, y_coords, bottom_border[0, :], bottom_border[1, :])
  left_border_dist = signed_distance_from_border(
      x_coords, y_coords, left_border[0, :], left_border[1, :])
  right_border_dist = signed_distance_from_border(
      x_coords, y_coords, right_border[0, :], right_border[1, :])

  mask = np.ones([patch_n, patch_n])
  mask[top_border_dist >= -1] = 0
  mask[bottom_border_dist <= 1] = 0
  mask[left_border_dist <= 1] = 0
  mask[right_border_dist >= -1] = 0
  return mask

def refine_bounding_box(image, rect, reltol=0.05, resolution=200):
    # Expand the bounding box slightly to search for edges not contained in the
    # original box. This expansion is done in the box's own coordinate frame, by
    # mapping it to the unit square, and then inverse-mapping an expanded unit
    # square.
    H = quad_to_unit_square_transform(rect)
    H_inv = np.linalg.inv(H)
    expanded_unit_square = np.array([
        (-reltol, -reltol),
        (1 + reltol, -reltol),
        (1 + reltol, 1 + reltol),
        (-reltol, 1 + reltol)])
    expanded_rect = np.astype(apply_transform(H_inv, expanded_unit_square), int)
    
    # Extract the image patch within the expanded bounding box.
    original_n = resolution # Output pixels for one side of the original box.
    border_n = int(original_n * reltol) # Output pixels of the added border.
    patch_n = original_n + border_n * 2 # Total output pixels per side.
    extracted_patch = image_processor.extract_perspective_image(
        Image.fromarray(image[:, :, ::-1]),
        expanded_rect, output_width=patch_n, output_height=patch_n)
    
    def patch_to_image_coords(pts):
        # Convert coordinates in the extracted patch back to original image.
        pts = np.asarray(pts)
        # First shift and scale patch coords back to the unit square.
        unit_square_coords = (pts - border_n) / original_n
        return apply_transform(H_inv, unit_square_coords)
    
    def image_to_patch_coords(pts):
        # Convert coordinates in the extracted patch back to original image.
        pts = np.asarray(pts)
        # First shift and scale patch coords back to the unit square.
        unit_square_coords = apply_transform(H, pts)
        return unit_square_coords * original_n + border_n
    
    # Find horizontal and vertical edges within the extracted patch
    gray = cv2.cvtColor(np.array(extracted_patch), cv2.COLOR_BGR2GRAY)
    sobel_vertical = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5))
    sobel_horizontal = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5))
    
    # If the extracted patch includes areas outside the bounds of the original
    # image, the missing pixels will be filled as black and will create a strong
    # artificial edge corresponding to the original image bounds. This can
    # confound the edges of the actual photo, so we want to
    # suppress the Sobel field when it reaches the boundaries of the original
    # image.
    # (note that the photo itself may extend outside the scanned image, but we
    # obviously can't get any useful edge information from the part we didn't
    # scan!)
    boundary_mask = image_boundary_mask(
        patch_n, image_to_patch_coords, image.shape)
    sobel_horizontal *= boundary_mask
    sobel_vertical *= boundary_mask

    # Apply a square root transform to the detected edges. This gives less
    # weight to 'outliers' --- individual pixels with strong edge detections ---
    # relative to consistent lines in which every pixel has some edge potential.
    sobel_horizontal = np.sqrt(sobel_horizontal)
    sobel_vertical = np.sqrt(sobel_vertical)

    # Slightly blur the detected edges so we give 'partial credit' to candidate
    # borders that are a pixel or two off from the exact edge.
    sobel_horizontal = cv2.GaussianBlur(sobel_horizontal, (5, 5), 0)
    sobel_vertical = cv2.GaussianBlur(sobel_vertical, (5, 5), 0)


    # Search for the edges that maximize the total response integrated across
    # the appropriate Sobel field (horizontal for top/bottom edges, vertical
    # for left/right edges).
    # TODO vectorize this computation?
    N = 2 * border_n
    # Propose potential edge points.
    top_pts, bottom_pts, left_pts, right_pts = _candidate_edge_points(
        patch_n, border_n)
    # Compute the Sobel response for each candidate edge.
    top_edge_weights = np.zeros([N, N])
    bottom_edge_weights = np.zeros([N, N])
    left_edge_weights = np.zeros([N, N])
    right_edge_weights = np.zeros([N, N])    
    for i in range(N):
        for j in range(N):
            top_edge_weights[i, j] = line_integral_simple(
                sobel_horizontal, top_pts[0, i, :], top_pts[1, j, :])
            bottom_edge_weights[i, j] = line_integral_simple(
                sobel_horizontal, bottom_pts[0, i, :], bottom_pts[1, j, :])
            left_edge_weights[i, j] = line_integral_simple(
                sobel_vertical, left_pts[0, i, :], left_pts[1, j, :])
            right_edge_weights[i, j] = line_integral_simple(
                sobel_vertical, right_pts[0, i, :], right_pts[1, j, :])

    # Get the edge point pairs that maximize the Sobel response.
    def get_best_edge(edge_weights, pts):
        left_idx, right_idx = np.unravel_index(edge_weights.argmax(), edge_weights.shape)
        pt0 = pts[0, left_idx]
        pt1 = pts[1, right_idx]
        return np.astype(np.asarray([pt0, pt1]), int)

    top_edge = get_best_edge(top_edge_weights, top_pts)
    bottom_edge = get_best_edge(bottom_edge_weights, bottom_pts)
    left_edge = get_best_edge(left_edge_weights, left_pts)
    right_edge = get_best_edge(right_edge_weights, right_pts)

    # Get corners for the refined bounding box by finding the intersections
    # of the refined edges.
    upper_left_corner = line_intersection(
        top_edge[0, :], top_edge[1, :], left_edge[0, :], left_edge[1, :])
    lower_left_corner = line_intersection(
        bottom_edge[0, :], bottom_edge[1, :], left_edge[0, :], left_edge[1, :])
    upper_right_corner = line_intersection(
        top_edge[0, :], top_edge[1, :], right_edge[0, :], right_edge[1, :])
    lower_right_corner = line_intersection(
        bottom_edge[0, :], bottom_edge[1, :], right_edge[0, :], right_edge[1, :])

    # Convert from the 

    image_rect = np.round(
        patch_to_image_coords(
            [upper_left_corner,
             upper_right_corner,
             lower_right_corner,
             lower_left_corner])
        ).astype(int)
    return image_rect