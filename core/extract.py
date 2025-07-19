"""
Image processing utilities for loading, cropping, and saving photos.
"""

from __future__ import annotations

import os
import platform
from datetime import datetime
from enum import Enum, IntEnum
from typing import Generator, Literal

import numpy as np
import piexif
import PIL.Image
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import core.photo_types as photo_types
from core import date_utils, geometry
from core.bounding_box import BoundingBox, PhotoAttributes
from core.photo_types import PhotoOrientation

# Allow math variables with uppercase letters.
# ruff: noqa N806


# Semantic type aliases
PILImage = PIL.Image.Image  # PIL/Pillow images


class FileExistsBehavior(Enum):
    OVERWRITE = 0
    SKIP = 1
    INCREMENT = 2


class OutputFormat(Enum):
    JPEG = "jpg"
    PNG = "png"
    TIFF = "tif"


class TiffTag(IntEnum):
    IMAGE_DESCRIPTION = 270
    DATE_TIME = 306
    DATE_TIME_ORIGINAL = 36867  # Date/time when original image was taken
    DATE_TIME_DIGITIZED = 36868  # Date/time when image was digitized
    ARTIST = 315
    COPYRIGHT = 33432
    SOFTWARE = 305


def get_file_creation_time(filepath: str) -> datetime | None:
    """Get the creation time of a file as a datetime object.

    On Windows, this returns the actual creation time (st_ctime).
    On macOS, this returns the birth time (st_birthtime) if available,
    otherwise falls back to modification time.
    On Linux and other Unix systems, this falls back to modification time
    since creation time is not reliably available.

    Args:
        filepath: Path to the file

    Returns:
        datetime object representing file creation time, or None if error
    """
    try:
        stat = os.stat(filepath)

        # Try platform-specific creation time fields
        if platform.system() == "Windows":
            # On Windows, st_ctime is creation time
            ctime = stat.st_ctime
        elif hasattr(stat, "st_birthtime"):
            # On macOS and some other systems, st_birthtime is creation time
            ctime = stat.st_birthtime
        else:
            # Fall back to modification time on Linux and other systems
            # where creation time is not reliably available
            ctime = stat.st_mtime

        return datetime.fromtimestamp(ctime)
    except (OSError, ValueError, AttributeError) as e:
        print(f"Warning: Could not read creation time for {filepath}: {e}")
        return None


def extract_perspective_image(
    image: PILImage,
    corner_points: photo_types.BoundingBoxAny,
    output_width: int | None = None,
    output_height: int | None = None,
    mode: PIL.Image.Resampling = Image.Resampling.BICUBIC,
    orientation: PhotoOrientation = PhotoOrientation.NORMAL,
) -> PILImage:
    """Crop image using four corner points."""
    # Convert corner points to numpy array
    corners = photo_types.bounding_box_as_array(corner_points)

    # Calculate the width and height of the output rectangle
    # Use the maximum dimensions to avoid losing any image data
    max_width, max_height = geometry.dimension_bounds(corners)
    output_width = output_width if output_width else int(max_width)
    output_height = output_height if output_height else int(max_height)

    # For 90° and 270° rotations, swap dimensions
    if orientation in (PhotoOrientation.ROTATED_90_CW, PhotoOrientation.ROTATED_90_CCW):
        output_width, output_height = output_height, output_width

    # Define the output rectangle corners (in order: top-left, top-right, bottom-right, bottom-left)
    # These corners define where the input corners should map to in the output
    if orientation == PhotoOrientation.NORMAL:
        output_corners = np.array(
            [
                [0, 0],
                [output_width, 0],
                [output_width, output_height],
                [0, output_height],
            ],
            dtype=np.float32,
        )
    elif orientation == PhotoOrientation.ROTATED_90_CCW:
        # rotate 90° CW: top-left → top-right, top-right → bottom-right, etc.
        output_corners = np.array(
            [
                [output_width, 0],
                [output_width, output_height],
                [0, output_height],
                [0, 0],
            ],
            dtype=np.float32,
        )
    elif orientation == PhotoOrientation.UPSIDE_DOWN:
        # rotate 180°: top-left → bottom-right, top-right → bottom-left, etc.
        output_corners = np.array(
            [
                [output_width, output_height],
                [0, output_height],
                [0, 0],
                [output_width, 0],
            ],
            dtype=np.float32,
        )
    elif orientation == PhotoOrientation.ROTATED_90_CW:
        # rotate 270° CW (90° CCW): top-left → bottom-left, top-right → top-left, etc.
        output_corners = np.array(
            [
                [0, output_height],
                [0, 0],
                [output_width, 0],
                [output_width, output_height],
            ],
            dtype=np.float32,
        )
    else:
        raise ValueError(f"Invalid orientation! {orientation}")

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
        (output_width, output_height),
        Image.Transform.PERSPECTIVE,
        coeffs,
        mode,
    )


def load_image(filepath: str) -> PILImage:
    return Image.open(filepath)


def save_cropped_images(
    image: PILImage,
    bounding_box_data_list: list[BoundingBox],
    output_dir: str,
    base_name: str = "photo",
    source_image_path: str | None = None,
    file_exists_behavior: FileExistsBehavior = FileExistsBehavior.OVERWRITE,
    output_format: OutputFormat = OutputFormat.JPEG,
    save_date_hint_in_description: bool = True,
) -> Generator[
    tuple[Literal["skipped"], str]
    | tuple[Literal["saved"], str]
    | tuple[Literal["error"], Exception]
]:
    """Save multiple cropped images to the specified directory.

    Args:
        image: image to crop from.
        bounding_box_data_list: List of BoundingBoxData objects containing
            corners and attributes for each photo to extract.
        output_dir: Directory to save images
        base_name: Base name for files
        source_image_path: Optional path to source image file (for EXIF
        DateTimeDigitized creation time)

    Yields:
        For each box to save, one of:
        ("saved", filename : str)
        ("skipped", filename: str)
        ("error", error: Exception)
    """
    saved_files = []

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read source file creation time if path provided
    source_file_time = None
    if source_image_path:
        source_file_time = get_file_creation_time(source_image_path)

    # Order boxes by their location on the page, top-to-bottom.
    bounding_box_data_list = sorted(
        bounding_box_data_list, key=lambda b: np.min(b.corners, axis=0)[1]
    )
    for i, bbox_data in enumerate(bounding_box_data_list):
        # Extract the image using the corners, incorporating rotation into the perspective transform
        attributes = bbox_data.attributes
        cropped = extract_perspective_image(
            image, bbox_data.corners, orientation=attributes.orientation
        )

        # Generate filename
        filename = f"{base_name}_{i:03d}.{output_format.value}"
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
            if file_exists_behavior == FileExistsBehavior.SKIP:
                yield ("skipped", filename)
                continue
            elif file_exists_behavior == FileExistsBehavior.INCREMENT:
                # Ensure unique filename
                counter = 1
                while os.path.exists(filepath):
                    filename = f"{base_name}_{i:03d}_{counter}.{output_format.value}"
                    filepath = os.path.join(output_dir, filename)
                    counter += 1

        try:
            # Convert to RGB if necessary
            if cropped.mode != "RGB":
                cropped = cropped.convert("RGB")

            save_image_with_exif(
                cropped, filepath, attributes, source_file_time=source_file_time
            )
            yield ("saved", filename)
            saved_files.append(filepath)
        except Exception as e:
            yield ("error", e)


def save_image_with_exif(
    image: PILImage,
    filepath: str,
    attributes: PhotoAttributes,
    jpeg_quality: int = 95,
    source_file_time: datetime | None = None,
    save_date_hint_in_description: bool = True,
) -> None:
    """Save image with EXIF data from attributes.

    Args:
        image: PIL image to save
        filepath: Output file path
        attributes: Photo attributes containing EXIF data
        jpeg_quality: JPEG compression quality (default 95)
        source_file_time: Optional datetime for DateTimeDigitized (source image creation time)
    """
    # Create EXIF dictionary
    exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
    pnginfo = PngInfo()
    tiffinfo = {}

    # Add date/time if available
    date_to_parse = attributes.exif_date or attributes.date_hint
    if date_to_parse:
        try:
            # Parse ISO date string and convert to EXIF format
            dt = date_utils.parse_flexible_date_as_datetime(date_to_parse)
            if dt is None:
                raise ValueError()

            exif_datetime = dt.strftime("%Y:%m:%d %H:%M:%S")

            # Set DateTime and DateTimeOriginal to the inferred photo date
            exif_dict["0th"][piexif.ImageIFD.DateTime] = exif_datetime
            exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = exif_datetime
            pnginfo.add_text("DateTime", exif_datetime)
            tiffinfo[TiffTag.DATE_TIME] = exif_datetime
            tiffinfo[TiffTag.DATE_TIME_ORIGINAL] = exif_datetime

            # Set DateTimeDigitized to source file time if available, otherwise photo date
            if source_file_time:
                digitized_datetime = source_file_time.strftime("%Y:%m:%d %H:%M:%S")
                exif_dict["Exif"][piexif.ExifIFD.DateTimeDigitized] = digitized_datetime
                pnginfo.add_text("DateTimeDigitized", digitized_datetime)
                tiffinfo[TiffTag.DATE_TIME_DIGITIZED] = digitized_datetime
        except (ValueError, AttributeError) as e:
            print(f"Warning: Could not parse date '{date_to_parse}': {e}")

    # Add comments if available
    comments = ""
    if attributes.comments:
        comments = attributes.comments[:65535]  # EXIF comment limit
    if attributes.date_hint and save_date_hint_in_description:
        if comments:
            comments += "\n"
        comments += f"Date: {attributes.date_hint}"
    if comments:
        exif_dict["0th"][piexif.ImageIFD.ImageDescription] = comments
        pnginfo.add_text("Description", comments)
        tiffinfo[TiffTag.IMAGE_DESCRIPTION] = comments

    # Add software tag
    exif_dict["0th"][piexif.ImageIFD.Software] = "Photo Album Extractor"
    pnginfo.add_text("Software", "Photo Album Extractor")
    tiffinfo[TiffTag.SOFTWARE] = "Photo Album Extractor"

    # Convert EXIF dictionary to bytes
    exif_bytes = piexif.dump(exif_dict)
    pnginfo.add_text("Exif", exif_bytes)

    # Save image with EXIF data
    extension = os.path.splitext(filepath)[1][1:].lower()
    if extension in ("jpg", "jpeg"):
        image.save(filepath, "JPEG", quality=jpeg_quality, exif=exif_bytes)
    elif extension == "png":
        image.save(filepath, "PNG", pnginfo=pnginfo)
    elif extension in ("tif", "tiff"):
        image.save(filepath, "TIFF", tiffinfo=tiffinfo)
    else:
        raise Exception(f"Unrecognized extension {extension}!")
