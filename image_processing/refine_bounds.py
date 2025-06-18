from __future__ import annotations

import logging
import os
import pathlib
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt
import PIL.Image
from PIL import Image

import image_processing.geometry as geometry
import image_processing.image_processor as image_processor
from photo_types import (AnyArray, BGRImage, BoundaryRefinementStrategy,
                         DirectoryPath, FloatArray, IntArray, UInt8Array, QuadArray, BoundingBoxAny, bounding_box_as_array)

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
LOGGER = logging.getLogger("logger")

# Registry of all available strategies
# Registry uses flexible typing since lambdas don't implement full Protocol
REFINEMENT_STRATEGIES: Dict[str, Callable[..., QuadArray]] = {
    "Original (200px res)": 
        (lambda image, corner_points, debug_dir=None: 
         refine_bounding_box(image, corner_points, resolution=200, 
                             enforce_parallel_sides=True, 
                             debug_dir=debug_dir)),
    "Original (multiscale)":
         (lambda image, corner_points, debug_dir=None: 
         refine_bounding_box_multiscale(image, corner_points, 
                                        enforce_parallel_sides=True,
                                        debug_dir=debug_dir)),
    "Original (multiscale, independent sides)": 
        (lambda image, corner_points, debug_dir=None: 
         refine_bounding_box_multiscale(image, corner_points, 
                                        enforce_parallel_sides=False,
                                        debug_dir=debug_dir)),
    "Strips (native res)": (
        lambda image, corner_points, debug_dir=None: 
         refine_bounding_box_strips(image, corner_points, 
                                    enforce_parallel_sides=True,
                                    debug_dir=debug_dir)),
    "Strips (10x downscaled)": (
        lambda image, corner_points, debug_dir=None: 
         refine_bounding_box_strips(image, corner_points,
                                    resolution_scale_factor=0.1, 
                                    enforce_parallel_sides=True, 
                                    debug_dir=debug_dir)),
    "Strips (multiscale)" : (
        lambda image, corner_points, debug_dir=None: 
         refine_bounding_box_strips_multiscale(
             image, corner_points, 
             enforce_parallel_sides=True,
             debug_dir=debug_dir))
}

def _candidate_edge_points(patch_n: int, border_n: int) -> Tuple[IntArray, IntArray, IntArray, IntArray]:
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

def signed_distance_from_border(x_coords: FloatArray, y_coords: FloatArray, border_pt1: FloatArray, border_pt2: FloatArray) -> FloatArray:
  x1, y1 = border_pt1
  x2, y2 = border_pt2
  return ((y2 - y1) * x_coords - (x2 - x1) * y_coords + x2 * y1 - y2 * x1
          ) / np.linalg.norm(border_pt2 - border_pt1)

def image_boundary_mask(image_shape: Tuple[int, ...], patch_shape: Tuple[int, ...], image_to_patch_coords: Callable[[FloatArray], FloatArray], offset: int = -1) -> FloatArray:
  img_height, img_width = image_shape[:2]
  top_border = image_to_patch_coords(np.array([(0, 0), (img_width-1, 0)]))
  bottom_border = image_to_patch_coords(np.array([(0, img_height-1), (img_width-1, img_height-1)]))
  left_border = image_to_patch_coords(np.array([(0, 0), (0, img_height-1)]))
  right_border = image_to_patch_coords(np.array([(img_width-1, 0), (img_width-1, img_height-1)]))

  strip_width, strip_height = patch_shape[:2]
  x_coords, y_coords = np.meshgrid(
      np.arange(strip_height), np.arange(strip_width))
  top_border_dist = signed_distance_from_border(
      x_coords, y_coords, top_border[0, :], top_border[1, :])
  bottom_border_dist = signed_distance_from_border(
      x_coords, y_coords, bottom_border[0, :], bottom_border[1, :])
  left_border_dist = signed_distance_from_border(
      x_coords, y_coords, left_border[0, :], left_border[1, :])
  right_border_dist = signed_distance_from_border(
      x_coords, y_coords, right_border[0, :], right_border[1, :])
  
  mask = np.ones([strip_width, strip_height])
  mask[top_border_dist >= offset] = 0
  mask[bottom_border_dist <= -offset] = 0
  mask[left_border_dist <= -offset] = 0
  mask[right_border_dist >= offset] = 0
  return mask

def get_best_edge(edge_weights: FloatArray, pts: IntArray) -> Tuple[IntArray, float]:
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
    LOGGER.debug(f"best offset {best_offset}")
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
    LOGGER.debug(f"best score {score}")
    LOGGER.debug(f"best edges {top_edge}, {bottom_edge}, {left_edge}, {right_edge}")
    return top_edge, bottom_edge, left_edge, right_edge

def annotate_image(img: AnyArray, contours: Optional[List[Any]] = None, edges: Optional[Union[List[IntArray], Tuple[IntArray, ...]]] = None) -> AnyArray:
  img = img.copy()
  if contours:
      cv2.drawContours(img, np.array(np.round(contours), dtype=int), -1, (0, 0, 255), 1)  # type: ignore
  if edges:
      colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (255, 255, 0), (0, 255, 255)]
      for edge, color in zip(edges, colors):
        cv2.line(img, (edge[0, 0], edge[0, 1]), (edge[1, 0], edge[1, 1]), color, 1)
  return img

def save_image(file_path: str, img: Union[AnyArray, PIL.Image.Image]) -> None:
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    if img.mode == 'F':
        # PNG supports greyscale images with 8-bit int pixels.
        img = img.convert("L")
        
    img.save(file_path)
    LOGGER.info(f"saved: {file_path}")

def refine_bounding_box(image: Union[UInt8Array, Image.Image], 
                        corner_points: BoundingBoxAny, reltol: float = 0.05, resolution: int = 200,
                        enforce_parallel_sides: bool = False,
                        debug_dir: Optional[str] = None) -> QuadArray:
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
    coordinates = geometry.PatchCoordinatesConverter(
        rect)
    expanded_unit_square = np.array([
        (-reltol, -reltol),
        (1 + reltol, -reltol),
        (1 + reltol, 1 + reltol),
        (-reltol, 1 + reltol)])
    expanded_rect = coordinates.unit_square_to_image(expanded_unit_square)
    LOGGER.debug(f"expanded rect {expanded_rect}")
        
    # Extract the image patch within the expanded bounding box. For simplicity,
    # use the same resolution in both dimensions so the extracted patch is a
    # (probably downscaled) square.
    patch_n = resolution + border_n * 2 # Total output pixels per side.
    extracted_patch = image_processor.extract_perspective_image(
        pil_image,
        expanded_rect, output_width=patch_n, output_height=patch_n)
    LOGGER.debug(f"extracted patch dims {extracted_patch.width} {extracted_patch.height}")
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
    
    boundary_mask = image_boundary_mask(
        image_to_patch_coords=_image_to_patch, 
        image_shape=image_shape,
        patch_shape=(patch_n, patch_n))
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
        # Find the set of perpendicular edges that maximize total Sobel
        # response.
        top_edge, bottom_edge, left_edge, right_edge = search_best_rhombus(
            top_pts, bottom_pts, left_pts, right_pts,
            top_edge_weights, bottom_edge_weights, left_edge_weights,
            right_edge_weights)
    else:
        # Indivually maximize the Sobel response under each edge.
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

    LOGGER.debug(
        f"patch rect {[upper_left_corner, upper_right_corner, lower_right_corner, lower_left_corner]}")

    # Convert the corner points of our refined bounding box back into image
    # coordinates.
    image_rect = _patch_to_image(
            [upper_left_corner,
             upper_right_corner,
             lower_right_corner,
             lower_left_corner])
    return image_rect


def refine_bounding_box_multiscale(
    image: Union[UInt8Array, Image.Image],
    corner_points: BoundingBoxAny,
    reltol: float = 0.05,
    base_resolution: int = 200,
    scale_factor: int = 5,
    enforce_parallel_sides: bool = False,
    debug_dir: Optional[str] = None) -> QuadArray:
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
            image, corner_points, reltol=reltol, resolution=resolution,
            enforce_parallel_sides=enforce_parallel_sides,
            debug_dir=debug_subdir)
        LOGGER.debug(
            f"resolution {resolution} reltol {reltol} corner_points {corner_points}")
    return corner_points


# Strip-based edge detection implementation

class StripData:
    """Container for a border strip and its coordinate transformations."""
    def __init__(self, pixels: UInt8Array, edge_response: FloatArray, image_to_strip_transform: Callable[[FloatArray], FloatArray], strip_to_image_transform: Callable[[FloatArray], FloatArray]) -> None:
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

def detect_edges_sharp(pixels: UInt8Array, horizontal: bool = True, debug_dir: Optional[str] = None) -> FloatArray:
    # Convert to numpy array and compute edge response
    gray = np.asarray(cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY), dtype=np.float32)
    
    filter = np.array([[-1, -2, -1],
                [-1, -2, -1], 
                [ 1,  2,  1],
                [ 1,  2,  1]], dtype=np.float32)
    
    
    if not horizontal:
        filter = np.array(filter.T)

    convolved = cv2.filter2D(gray, -1, filter)  # type: ignore
    edge_response = np.sqrt(np.abs(convolved))
    #edge_response = cv2.GaussianBlur(edge_response, (5, 5), 0)
    return edge_response


def detect_edges_color(pixels: UInt8Array, horizontal: bool = True, mask: Optional[FloatArray] = None) -> FloatArray:
    # Convert to numpy array and compute edge response
    if pixels.shape[-1] != 3:
        raise ValueError(f"Expected color image, got array of shape {pixels.shape}")
    r, g, b = [np.astype(pixels[..., i], np.float32) for i in range(3)]
    
    
    filter = np.array([[-1, -2, -1],
                [-1, -2, -1], 
                [ 1,  2,  1],
                [ 1,  2,  1]], dtype=np.float32)
    
    if not horizontal:
        filter = np.array(filter.T)

    cr, cg, cb = [
        cv2.filter2D(channel, -1, filter) for channel in [r, g, b]]
    edge_response = np.sqrt(np.linalg.norm([cr, cg, cb], axis=0))
    if mask is not None:
        edge_response *= mask
    edge_response = cv2.GaussianBlur(edge_response, (5, 5), 0)
    return edge_response

def extract_border_strips(image: Union[Image.Image, UInt8Array], 
                          rect: QuadArray,
                          reltol: float, 
                          resolution_scale_factor: float = 1., 
                          min_image_pixels: int = 8,
                          debug_dir: Optional[str] = None) -> Dict[str, 'StripData']:
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
    top_strip_normalized = np.array([
        [0 - reltol_x, -reltol_y],
        [1 + reltol_x, -reltol_y], 
        [1 + reltol_x, reltol_y],
        [0 - reltol_x, reltol_y]
    ])
    
    # Bottom strip: full width, bottom portion
    bottom_strip_normalized = np.array([
        [0 - reltol_x, 1 - reltol_y],
        [1 + reltol_x, 1 - reltol_y],
        [1 + reltol_x, 1 + reltol_y],
        [0 - reltol_x, 1 + reltol_y]
    ])
    
    # Left strip: left portion, full height
    left_strip_normalized = np.array([
        [- reltol_x, 0 - reltol_y],
        [reltol_x, 0 - reltol_y],
        [reltol_x, 1 + reltol_y],
        [- reltol_x, 1 + reltol_y]
    ])
    
    # Right strip: right portion, full height  
    right_strip_normalized = np.array([
        [1 - reltol_x, 0 - reltol_y],
        [1 + reltol_x, 0 - reltol_y],
        [1 + reltol_x, 1 + reltol_y],
        [1 - reltol_x, 1 + reltol_y]
    ])
    
    # Convert to image coordinates and extract each strip
    converter = geometry.PatchCoordinatesConverter(rect)
    
    for name, strip_normalized in [
        ('top', top_strip_normalized),
        ('bottom', bottom_strip_normalized),
        ('left', left_strip_normalized),
        ('right', right_strip_normalized)
    ]:
        # Convert normalized coords to image coords
        strip_corners_image = converter.unit_square_to_image(strip_normalized)
        strip_corners_image = np.round(strip_corners_image)
        width, height = geometry.dimension_bounds(strip_corners_image)
        strip_width = int(np.ceil(width * resolution_scale_factor))
        strip_height = int(np.ceil(height * resolution_scale_factor))
        
        # Extract the strip
        LOGGER.debug(f"extracting strip {name} with corners {strip_corners_image} image width {width} height {height} strip width {strip_width} height {strip_height}")
        strip_pixels = image_processor.extract_perspective_image(
            pil_image,
            strip_corners_image,
            output_width=strip_width,
            output_height=strip_height,
            mode=Image.Resampling.BICUBIC)
        pixels_array = np.array(strip_pixels)
        
        # Create coordinate converters for this strip
        strip_converter = geometry.PatchCoordinatesConverter(
            strip_corners_image)
        image_to_strip_transform=(
            lambda pts, sc=strip_converter, sw=strip_width, sh=strip_height:
                sc.image_to_unit_square(pts) * np.array([sw, sh]))
        strip_to_image_transform=(
            lambda pts, sc=strip_converter, sw=strip_width, sh=strip_height:
                sc.unit_square_to_image(pts / np.array([sw, sh])))
        
        strip_mask = image_boundary_mask(
            image_shape=image_shape,
            patch_shape=pixels_array.shape,
            image_to_patch_coords=image_to_strip_transform)

        edge_response = detect_edges_color(
            pixels_array,
            horizontal=(name in ['top', 'bottom']),
            mask=strip_mask)
                
        # Store strip data
        strips[name] = StripData(
            pixels=pixels_array,
            edge_response=edge_response,
            image_to_strip_transform=image_to_strip_transform,
            strip_to_image_transform=strip_to_image_transform)
        
    return strips


def evaluate_edges_at_angle(strip, angle, strip_type, debug_dir=None):
    """Evaluate all possible edge positions at given angle in a strip."""
    if debug_dir is not None:
        debug_dir = os.path.join(debug_dir, "candidates")
        pathlib.Path(debug_dir).mkdir(parents=True, exist_ok=True)
    
    h, w = strip.edge_response.shape
    
    if strip_type in ['top', 'bottom']:
        # Horizontal strips - edges go left to right with slight angle        
        ys = np.arange(h)
        edge_pts1 = np.column_stack(
            (np.zeros_like(ys), ys - w/2 * np.tan(angle)))
        edge_pts2 = np.column_stack(
            (np.full_like(ys, w - 1), ys + w/2 * np.tan(angle)))
    else:  # left or right
        # Vertical strips - edges go top to bottom with slight angle
        xs = np.arange(w)
        edge_pts1 = np.column_stack(
            (xs + h/2 * np.tan(angle), np.zeros_like(xs)))
        edge_pts2 = np.column_stack(
            (xs - h/2 * np.tan(angle), np.full_like(xs, h - 1)))
            
    scores = geometry.line_integral_vectorized(
                strip.edge_response, 
                edge_pts1, 
                edge_pts2)
    return scores


def search_rectangle_factored(strips, initial_rect, debug_dir=None):
    """Search for best rectangle using factored approach."""
    # Compute initial angle
    edge_vector = initial_rect[1] - initial_rect[0]
    base_angle = np.arctan2(edge_vector[1], edge_vector[0])
    
    # Use enough angles that we can hit any pixel on the edge of the strip,
    # and ensure that the number is odd so that we always consider angle==0.
    num_angles = strips['top'].edge_response.shape[0]
    num_angles = num_angles + 1 if (num_angles % 2 == 0) else num_angles
    
    # Limit angle search to ensure edges span strips
    # Maximum angle where edges still span the strip
    h, w = strips['top'].edge_response.shape
    max_angle_deviation = np.arctan(h / w)
    
    strip_angles = np.linspace(
        - max_angle_deviation, 
        max_angle_deviation, 
        num_angles)

    best_score = -np.inf
    best_edges = None
    best_angle = None
    
    for angle in strip_angles:
        # Evaluate all edges at this angle
        edge_scores = {}
        
        # Top and bottom edges are at angle θ
        edge_scores['top'] = evaluate_edges_at_angle(
            strips['top'], angle, 'top',
            debug_dir=debug_dir)
        edge_scores['bottom'] = evaluate_edges_at_angle(
            strips['bottom'], angle, 'bottom')
        
        # Left and right edges are perpendicular (angle θ + π/2)
        edge_scores['left'] = evaluate_edges_at_angle(
            strips['left'], angle , 'left')
        edge_scores['right'] = evaluate_edges_at_angle(
            strips['right'], angle , 'right')

        score = (np.max(edge_scores['top']) +
                np.max(edge_scores['bottom']) +
                np.max(edge_scores['left']) +
                np.max(edge_scores['right']))
        

        edges = {
                'top': np.argmax(edge_scores['top']),
                'bottom': np.argmax(edge_scores['bottom']),
                'left': np.argmax(edge_scores['left']),
                'right': np.argmax(edge_scores['right'])}
        
        if score > best_score:
            best_score = score
            best_edges = edges
            best_angle = base_angle + angle
            
        LOGGER.debug(f"angle {angle:.3f}: best score {score:.1f}")
    
    return best_edges, best_angle, best_score


def find_corner_intersections(edges):
    # Find corner intersections
    upper_left = geometry.line_intersection(
        edges['top'][0], edges['top'][1],
        edges['left'][0], edges['left'][1]
    )
    upper_right = geometry.line_intersection(
        edges['top'][0], edges['top'][1],
        edges['right'][0], edges['right'][1]
    )
    lower_right = geometry.line_intersection(
        edges['bottom'][0], edges['bottom'][1],
        edges['right'][0], edges['right'][1]
    )
    lower_left = geometry.line_intersection(
        edges['bottom'][0], edges['bottom'][1],
        edges['left'][0], edges['left'][1]
    )
    return np.array([upper_left, upper_right, lower_right, lower_left])

def get_edges_as_image_coordinates(edge_indices, strips, angle_offset, debug_dir=None):
    """Convert edge indices to image coordinates."""
    # Get the edge lines in strip coordinates
    edges_strip = {}
    
    # Top edge
    top_y = edge_indices['top']
    top_h, top_w = strips['top'].edge_response.shape
    top_y0 = top_y - top_w/2 * np.tan(angle_offset)
    top_y1 = top_y + top_w/2 * np.tan(angle_offset)
    edges_strip['top'] = (
        np.array([0, top_y0]),
        np.array([top_w-1, top_y1])
    )
    
    # Bottom edge
    bottom_y = edge_indices['bottom']
    bottom_y0 = bottom_y - top_w/2 * np.tan(angle_offset)
    bottom_y1 = bottom_y + top_w/2 * np.tan(angle_offset)
    edges_strip['bottom'] = (
        np.array([0, bottom_y0]),
        np.array([top_w-1, bottom_y1])
    )
    
    # Left edge
    left_x = edge_indices['left']
    left_h, left_w = strips['left'].edge_response.shape
    left_x0 = left_x + left_h/2 * np.tan(angle_offset)
    left_x1 = left_x - left_h/2 * np.tan(angle_offset)
    edges_strip['left'] = (
        np.array([left_x0, 0]),
        np.array([left_x1, left_h-1])
    )
    
    # Right edge
    right_x = edge_indices['right']
    right_x0 = right_x + left_h/2 * np.tan(angle_offset)
    right_x1 = right_x - left_h/2 * np.tan(angle_offset)
    edges_strip['right'] = (
        np.array([right_x0, 0]),
        np.array([right_x1, left_h-1])
    )
    
            
    if debug_dir:
        for name, strip in strips.items():
            edge = edges_strip[name]
            save_image(
                os.path.join(debug_dir, f"edge_{name}.png"),
                annotate_image(strip.edge_response, edges=[np.array(edge, dtype=int)])
            )

    # Convert to image coordinates
    edges_image = {}
    for name, (pt1, pt2) in edges_strip.items():
        strip = strips[name]
        edges_image[name] = (
            strip.strip_to_image(pt1.reshape(1, -1))[0],
            strip.strip_to_image(pt2.reshape(1, -1))[0]
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
            strip_pt2 = np.array([w-1, y])
            score = geometry.line_integral_simple(
                strip.edge_response, strip_pt1, strip_pt2
            )
            if score > best_score:
                best_score = score
                # Convert to image coordinates
                best_edge = (
                    strip.strip_to_image(strip_pt1.reshape(1, -1))[0],
                    strip.strip_to_image(strip_pt2.reshape(1, -1))[0]
                )
    else:  # left, right
        # Vertical edge - spans height
        x_values = np.arange(w)
        best_score = -np.inf
        best_edge = None
        
        for x in x_values:
            strip_pt1 = np.array([x, 0])
            strip_pt2 = np.array([x, h-1])
            score = geometry.line_integral_simple(
                strip.edge_response, strip_pt1, strip_pt2
            )
            if score > best_score:
                best_score = score
                # Convert to image coordinates
                best_edge = (
                    strip.strip_to_image(strip_pt1.reshape(1, -1))[0],
                    strip.strip_to_image(strip_pt2.reshape(1, -1))[0]
                )
            
    return best_edge


def refine_bounding_box_strips(image: Union[Image.Image, UInt8Array],
                               corner_points: BoundingBoxAny, 
                               reltol: float = 0.05,
                               resolution_scale_factor: float = 1.,
                              enforce_parallel_sides: bool = False, debug_dir: Optional[str] = None) -> QuadArray:
    """Refine bounding box using strip-based edge detection."""
    corner_points = bounding_box_as_array(corner_points)
    if debug_dir is not None:
        pathlib.Path(debug_dir).mkdir(parents=True, exist_ok=True)
        LOGGER.info(f"logging to debug dir {debug_dir}")
        save_image(
            os.path.join(debug_dir, "init.png"),
            annotate_image(image, [corner_points])
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
        image, rect, reltol=reltol, 
        resolution_scale_factor=resolution_scale_factor, 
        debug_dir=debug_dir)
    
    # Apply boundary masking to each strip
    # TODO: Implement boundary masking for strips
    
    # Save debug images if requested
    if debug_dir:
        for name, strip in strips.items():
            save_image(
                os.path.join(debug_dir, f"strip_{name}.png"),
                strip.pixels[:, :, ::-1]
            )
            save_image(
                os.path.join(debug_dir, f"edge_response_{name}.png"),
                strip.edge_response
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
            best_edges_strip, strips, 
            best_angle - np.arctan2(rect[1][1] - rect[0][1], rect[1][0] - rect[0][0]),
            debug_dir=debug_dir)
    else:
        # Find best edge in each strip independently
        best_edges = {}
        for edge_name in ['top', 'bottom', 'left', 'right']:
            best_edges[edge_name] = search_best_edge(
                strips[edge_name],
                edge_is_horizontal=(edge_name in ["top", "bottom"]))
            LOGGER.debug(f"best {edge_name} edge: {best_edges[edge_name]}")

    LOGGER.debug(f"got best edges {best_edges}")
    corners = find_corner_intersections(best_edges)
    LOGGER.debug(f"returning refined corners {corners}")
    if debug_dir is not None:
        save_image(
            os.path.join(debug_dir, "result.png"),
            annotate_image(image, [corners])
        )
    return corners


def refine_bounding_box_strips_multiscale(
    image: UInt8Array,
    corner_points: BoundingBoxAny,
    reltol: float = 0.05,
    base_resolution: int = 200,
    scale_step: int = 4,
    enforce_parallel_sides: bool = False,
    debug_dir: Optional[str] = None) -> QuadArray:
    """Coarse-to-fine refinement using strip-based edge detection."""
    corner_points = bounding_box_as_array(corner_points)
    
    outer_resolution = int(max(geometry.dimension_bounds(corner_points)))
    
    scale_factors = [base_resolution / outer_resolution]
    reltols = [reltol]
    
    new_scale_factor = scale_factors[-1]
    
    while new_scale_factor < 1.:
        new_scale_factor = min(1., new_scale_factor * scale_step)
        scale_factors.append(new_scale_factor)
        reltols.append(
            min(reltol / 2., scale_step / (new_scale_factor * base_resolution)))
        
    scale_factors[-1] = 1.
    
    LOGGER.debug("SCALE FACTORS", scale_factors)
    LOGGER.debug("RELTOLS", reltols)
    
    
    for reltol, scale_factor in zip(reltols, scale_factors):
        debug_subdir = None
        if debug_dir:
            debug_subdir = os.path.join(debug_dir, f"strips_{scale_factor}_{reltol:.5f}")
            
        corner_points = refine_bounding_box_strips(
            image, corner_points, reltol=reltol, resolution_scale_factor=scale_factor,
            enforce_parallel_sides=enforce_parallel_sides,
            debug_dir=debug_subdir)
        LOGGER.debug(
            f"strips: resolution {scale_factor} reltol {reltol} corner_points {corner_points}")
    
    return corner_points