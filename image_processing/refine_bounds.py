import pathlib
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np

from PIL import Image

from typing import Union

import image_processing.image_processor as image_processor
import image_processing.geometry as geometry

import logging


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
    edge = np.asarray([pt0, pt1])
    logging.debug(f"best edge {edge}")
    return edge, edge_weights[left_idx, right_idx]

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
    logging.debug("best offset", best_offset)
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
    logging.debug("best score", score)
    logging.debug(f"best edges {top_edge}, {bottom_edge}, {left_edge}, {right_edge}")
    return top_edge, bottom_edge, left_edge, right_edge

def annotate_image(img: np.ndarray, rects=None, edges=None):
  img = img.copy()
  if rects:
      cv2.drawContours(img, np.array(rects), -1, (0, 0, 255), 1)
  if edges:
      colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (255, 255, 0), (0, 255, 255)]
      for edge, color in zip(edges, colors):
        cv2.line(img, (edge[0, 0], edge[0, 1]), (edge[1, 0], edge[1, 1]), color, 1)
  return img

def save_image(file_path: str, img: Union[np.ndarray, Image.Image]):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    if img.mode == 'F':
        # PNG supports greyscale images with 8-bit int pixels.
        img = img.convert("L")
        
    img.save(file_path)
    logging.info("saved: ", file_path)

def refine_bounding_box(image, rect, reltol=0.05, resolution=200,
                        enforce_parallel_sides=False,
                        debug_dir=None):
    if debug_dir is not None:
        pathlib.Path(debug_dir).mkdir(parents=True, exist_ok=True)
        logging.info("logging to debug dir", debug_dir)
        

    # Bound the given shape with a (not necessarily axis-aligned) rectangle.
    # We'll do all our refinement calculations within this rectangle. Using
    # a rectangle ensures that the transformation into and out of this space
    # preserves parallel lines.
    rect, _ = geometry.minimum_bounding_rectangle(rect)
    logging.debug("bounding rect", rect)
    
    # Reimpose our standard corner ordering since the previous call may lose it.
    rect = geometry.sort_clockwise(rect)
    border_n = int(resolution * reltol)
    coordinates = geometry.PatchCoordinatesConverter(
        rect, patch_resolution=resolution, patch_offset=border_n)
    
    # Expand the bounding box slightly to search for edges not contained in the
    # original box. This expansion is done in the box's own coordinate frame, by
    # mapping it to the unit square, and then inverse-mapping an expanded unit
    # square.
    
    expanded_unit_square = np.array([
        (-reltol, -reltol),
        (1 + reltol, -reltol),
        (1 + reltol, 1 + reltol),
        (-reltol, 1 + reltol)])
    expanded_rect = coordinates.unit_square_to_image(expanded_unit_square)
    logging.debug("expanded rect", expanded_rect)
        
    # Extract the image patch within the expanded bounding box.
    patch_n = resolution + border_n * 2 # Total output pixels per side.
    extracted_patch = image_processor.extract_perspective_image(
        Image.fromarray(image[:, :, ::-1]),
        expanded_rect, output_width=patch_n, output_height=patch_n)
    logging.debug(f"extracted patch dims {extracted_patch.width} {extracted_patch.height}")
    if debug_dir:
        patch_array = np.array(extracted_patch)[:, :, ::-1]
        border_rect = [(border_n * 2, border_n * 2), 
                       (border_n * 2, resolution),
                       (resolution, resolution),
                       (resolution, border_n * 2)]
        patch_array = annotate_image(patch_array, [border_rect])
        save_image(os.path.join(debug_dir, "patch_with_bounds.png"),
                   patch_array)

    # Find horizontal and vertical edges within the extracted patch
    gray = cv2.cvtColor(np.array(extracted_patch), cv2.COLOR_BGR2GRAY)
    sobel_vertical = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5))
    sobel_horizontal = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5))
    if debug_dir:
        save_image(os.path.join(debug_dir, "sobel_vertical.png"), 
                   annotate_image(sobel_vertical, [border_rect]))
        save_image(os.path.join(debug_dir, "sobel_horizontal.png"), 
                   annotate_image(sobel_horizontal, [border_rect]))

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
        patch_n, image_to_patch_coords=coordinates.image_to_patch, 
        img_shape=image.shape)
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
    #sobel_horizontal = cv2.GaussianBlur(sobel_horizontal, (5, 5), 0)
    #sobel_vertical = cv2.GaussianBlur(sobel_vertical, (5, 5), 0)
    if debug_dir:
        save_image(os.path.join(debug_dir, "sobel_vertical_sqrt_blur.png"), 
                   annotate_image(sobel_vertical, [border_rect]))
        save_image(os.path.join(debug_dir, "sobel_horizontal_sqrt_blur.png"),
                   annotate_image(sobel_horizontal, [border_rect]))
    

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
        
    if debug_dir:
        annotated = annotate_image(
            np.array(extracted_patch)[:, :, ::-1],
            edges=(top_edge, bottom_edge, left_edge, right_edge))
        save_image(os.path.join(debug_dir, "best_edges.png"), annotated)


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

    logging.debug(
        f"patch rect {[upper_left_corner, upper_right_corner,
        lower_right_corner, lower_left_corner]}")

    image_rect = coordinates.patch_to_image(
            [upper_left_corner,
             upper_right_corner,
             lower_right_corner,
             lower_left_corner])
    return image_rect


def refine_bounding_box_multiscale(
    image, rect,
    reltol=0.05,
    base_resolution=200,
    scale_factor=5,
    enforce_parallel_sides=False,
    debug_dir=None):
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
        debug_subdir = None
        if debug_dir:
            debug_subdir = os.path.join(debug_dir, f"{resolution}_{reltol:.5f}")
            
        rect = refine_bounding_box(
            image, rect, reltol=reltol, resolution=resolution,
            enforce_parallel_sides=enforce_parallel_sides,
            debug_dir=debug_subdir)
        logging.debug(f"resolution {resolution} reltol {reltol} rect {rect}")
    return rect