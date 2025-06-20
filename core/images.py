"""
Image processing utilities for loading, cropping, and saving photos.
"""

# ruff: noqa N806, N803

import os
from datetime import datetime
from typing import Optional

import core.photo_types as photo_types
import numpy as np
import piexif
import PIL.Image
from core import geometry
from core.photo_types import BoundingBoxData, PhotoAttributes
from PIL import Image

# Semantic type aliases
PILImage = PIL.Image.Image  # PIL/Pillow images


def extract_perspective_image(
    image: PILImage,
    corner_points: photo_types.BoundingBoxAny,
    output_width: Optional[int] = None,
    output_height: Optional[int] = None,
    mode: PIL.Image.Resampling = Image.Resampling.BICUBIC,
) -> PILImage:
    """Crop image using four corner points."""
    # Convert corner points to numpy array
    corners = photo_types.bounding_box_as_array(corner_points)

    # Calculate the width and height of the output rectangle
    # Use the maximum dimensions to avoid losing any image data
    max_width, max_height = geometry.dimension_bounds(corners)
    output_width = output_width if output_width else int(max_width)
    output_height = output_height if output_height else int(max_height)

    # Define the output rectangle corners (in order: top-left, top-right, bottom-right, bottom-left)
    output_corners = np.array(
        [[0, 0], [output_width, 0], [output_width, output_height], [0, output_height]],
        dtype=np.float32,
    )

    # Calculate transformation matrix
    # We need to map from output_corners to corners
    def find_coeffs(pa, pb):
        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])
        mat_A = np.array(matrix, dtype=np.float32)
        vec_B = np.array(corners.flatten(), dtype=np.float32)
        try:
            res = np.dot(np.linalg.inv(mat_A.T @ mat_A) @ mat_A.T, vec_B)
            return np.array(res).flatten()
        except Exception:
            return None

    coeffs = find_coeffs(output_corners, corners)

    # Apply perspective transformation
    return image.transform(
        (output_width, output_height), Image.Transform.PERSPECTIVE, coeffs, mode
    )


def load_image(filepath: str) -> PILImage:
    return Image.open(filepath)


def save_cropped_images(
    image: PILImage,
    bounding_box_data_list: list[BoundingBoxData],
    output_dir: str,
    base_name: str = "photo",
) -> list[str]:
    """Save multiple cropped images to the specified directory.

    Args:
        image: image to crop from.
        bounding_box_data_list: List of BoundingBoxData objects containing
            corners and attributes for each photo to extract.
        output_dir: Directory to save images
        base_name: Base name for files
    """
    saved_files = []

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for i, bbox_data in enumerate(bounding_box_data_list):
        # Extract the image using the corners
        cropped = extract_perspective_image(image, bbox_data.corners)

        # Get attributes from the bounding box data
        attributes = bbox_data.attributes

        # Generate filename
        filename = f"{base_name}_{i:03d}.jpg"
        filepath = os.path.join(output_dir, filename)

        # Ensure unique filename
        counter = 1
        while os.path.exists(filepath):
            filename = f"{base_name}_{i:03d}_{counter}.jpg"
            filepath = os.path.join(output_dir, filename)
            counter += 1

        try:
            # Convert to RGB if necessary
            if cropped.mode != "RGB":
                cropped = cropped.convert("RGB")

            save_image_with_exif(cropped, filepath, attributes)
            saved_files.append(filepath)
            print(f"Saved: {filename}")
        except Exception as e:
            print(f"Error saving {filename}: {e}")

    return saved_files


def save_image_with_exif(
    image: PILImage, filepath: str, attributes: PhotoAttributes, jpeg_quality: int = 95
) -> None:
    """Save image with EXIF data from attributes."""
    try:
        # Create EXIF dictionary
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

        # Add date/time if available
        if attributes.date_time:
            try:
                # Parse ISO date string and convert to EXIF format
                dt = datetime.fromisoformat(attributes.date_time.replace("Z", "+00:00"))
                exif_datetime = dt.strftime("%Y:%m:%d %H:%M:%S")

                # Set multiple date fields for maximum compatibility
                exif_dict["0th"][piexif.ImageIFD.DateTime] = exif_datetime
                exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = exif_datetime
                exif_dict["Exif"][piexif.ExifIFD.DateTimeDigitized] = exif_datetime
            except (ValueError, AttributeError) as e:
                print(f"Warning: Could not parse date '{attributes.date_time}': {e}")

        # Add comments if available
        if attributes.comments:
            comments = attributes.comments[:65535]  # EXIF comment limit
            # Use UserComment for better unicode support
            exif_dict["Exif"][piexif.ExifIFD.UserComment] = comments.encode("utf-8")
            # Also set ImageDescription for wider compatibility
            exif_dict["0th"][piexif.ImageIFD.ImageDescription] = comments

        # Add software tag
        exif_dict["0th"][piexif.ImageIFD.Software] = "Photo Album Extractor"

        # Convert EXIF dictionary to bytes
        exif_bytes = piexif.dump(exif_dict)

        # Save image with EXIF data
        image.save(filepath, "JPEG", quality=jpeg_quality, exif=exif_bytes)

    except Exception as e:
        print(f"Warning: Could not write EXIF data to {filepath}: {e}")
        # Fall back to saving without EXIF
        image.save(filepath, "JPEG", quality=jpeg_quality)
