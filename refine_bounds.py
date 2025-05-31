import cv2
from matplotlib import pyplot as plt
import numpy as np

from PIL import Image

import image_processor
import geometry



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

def get_best_edge(edge_weights, pts):
    # Get the edge point pairs that maximize the Sobel response.
    left_idx, right_idx = np.unravel_index(edge_weights.argmax(), edge_weights.shape)
    pt0 = pts[0, left_idx]
    pt1 = pts[1, right_idx]
    return np.astype(np.asarray([pt0, pt1]), int), edge_weights[left_idx, right_idx]

def search_best_rhombus(
    top_pts,
    bottom_pts,
    left_pts,
    right_pts,
    top_edge_weights,
    bottom_edge_weights,
    left_edge_weights,
    right_edge_weights):
    N = top_edge_weights.shape[0]
    offsets = np.arange(-N, N)
    offset_scores = np.zeros_like(offsets)
    idxs2, idxs1 = np.meshgrid(np.arange(N), np.arange(N))
    for i, offset in enumerate(offsets):
        horizontal_offset_mask = ((idxs2 - idxs1) == offset)
        vertical_offset_mask = ((idxs1 - idxs2) == offset)

        top_score = np.max(top_edge_weights * horizontal_offset_mask)
        bottom_score = np.max(bottom_edge_weights * horizontal_offset_mask)
        left_score = np.max(left_edge_weights * vertical_offset_mask)
        right_score = np.max(right_edge_weights * vertical_offset_mask)
        offset_scores[i] = top_score + bottom_score + left_score + right_score

    best_offset_idx = np.argmax(offset_scores)
    best_offset = offsets[best_offset_idx]
    print("best offset", best_offset)
    best_score = np.max(offset_scores)
    horizontal_offset_mask = ((idxs2 - idxs1) == best_offset)
    vertical_offset_mask = ((idxs1 - idxs2) == best_offset)
    top_edge, top_score = get_best_edge(
        top_edge_weights  * horizontal_offset_mask, top_pts)
    bottom_edge, bottom_score = get_best_edge(
        bottom_edge_weights * horizontal_offset_mask, bottom_pts)
    left_edge, left_score = get_best_edge(
        left_edge_weights * vertical_offset_mask, left_pts)
    right_edge, right_score = get_best_edge(
        right_edge_weights * vertical_offset_mask, right_pts)

    score = top_score + bottom_score + left_score + right_score
    print("best score", score)
    print("score", score)
    return top_edge, bottom_edge, left_edge, right_edge

def refine_bounding_box(image, rect, reltol=0.05, resolution=200,
                        enforce_parallel_sides=False):
    
    rect, _ = geometry.minimum_bounding_rectangle(rect)
    
    # Expand the bounding box slightly to search for edges not contained in the
    # original box. This expansion is done in the box's own coordinate frame, by
    # mapping it to the unit square, and then inverse-mapping an expanded unit
    # square.
    H = geometry.quad_to_unit_square_transform(rect)
    H_inv = np.linalg.inv(H)
    expanded_unit_square = np.array([
        (-reltol, -reltol),
        (1 + reltol, -reltol),
        (1 + reltol, 1 + reltol),
        (-reltol, 1 + reltol)])
    expanded_rect = np.astype(geometry.apply_transform(H_inv, expanded_unit_square), int)
    
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
        return geometry.apply_transform(H_inv, unit_square_coords)
    
    def image_to_patch_coords(pts):
        # Convert coordinates in the extracted patch back to original image.
        pts = np.asarray(pts)
        # First shift and scale patch coords back to the unit square.
        unit_square_coords = geometry.apply_transform(H, pts)
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
            top_edge_weights[i, j] = geometry.line_integral_simple(
                sobel_horizontal, top_pts[0, i, :], top_pts[1, j, :])
            bottom_edge_weights[i, j] = geometry.line_integral_simple(
                sobel_horizontal, bottom_pts[0, i, :], bottom_pts[1, j, :])
            left_edge_weights[i, j] = geometry.line_integral_simple(
                sobel_vertical, left_pts[0, i, :], left_pts[1, j, :])
            right_edge_weights[i, j] = geometry.line_integral_simple(
                sobel_vertical, right_pts[0, i, :], right_pts[1, j, :])

    if enforce_parallel_sides:
        top_edge, bottom_edge, left_edge, right_edge = search_best_rhombus(
            top_pts, bottom_pts, left_pts, right_pts,
            top_edge_weights, bottom_edge_weights, left_edge_weights,
            right_edge_weights)
    else:
        top_edge, _ = get_best_edge(top_edge_weights, top_pts)
        bottom_edge, _ = get_best_edge(bottom_edge_weights, bottom_pts)
        left_edge, _ = get_best_edge(left_edge_weights, left_pts)
        right_edge, _ = get_best_edge(right_edge_weights, right_pts)

    # Get corners for the refined bounding box by finding the intersections
    # of the refined edges.
    upper_left_corner = geometry.line_intersection(
        top_edge[0, :], top_edge[1, :], left_edge[0, :], left_edge[1, :])
    lower_left_corner = geometry.line_intersection(
        bottom_edge[0, :], bottom_edge[1, :], left_edge[0, :], left_edge[1, :])
    upper_right_corner = geometry.line_intersection(
        top_edge[0, :], top_edge[1, :], right_edge[0, :], right_edge[1, :])
    lower_right_corner = geometry.line_intersection(
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


def refine_bounding_box_multiscale(
    image, rect,
    reltol=0.1,
    base_resolution=200,
    scale_factor=5,
    enforce_parallel_sides=False):
    rect = np.array(rect)
    width1 = np.linalg.norm(rect[1] - rect[0])
    width2 = np.linalg.norm(rect[2] - rect[3])
    height1 = np.linalg.norm(rect[3] - rect[0])
    height2 = np.linalg.norm(rect[2] - rect[1])
    outer_resolution = int(np.max([width1, width2, height1, height2]))
    
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
        rect = refine_bounding_box(
            image, rect, reltol=reltol, resolution=resolution,
            enforce_parallel_sides=enforce_parallel_sides)
        print("resolution", resolution, "reltol", reltol, "rect", rect)
    return rect