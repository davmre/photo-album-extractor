"""
Info command implementation for CLI.
"""

from __future__ import annotations

import logging

from cli.utils import get_image_files
from core.bounding_box_storage import BoundingBoxStorage


def cmd_info(paths: list[str]) -> int:
    """Show bounding box information for specified images."""
    logger = logging.getLogger(__name__)
    logger.info(f"Getting info for {len(paths)} path(s)")

    image_files = get_image_files(paths)
    if image_files is None:
        return 1

    if not image_files:
        print("No image files found")
        return 0

    for image_path in image_files:
        directory = image_path.parent
        filename = image_path.name

        storage = BoundingBoxStorage(str(directory))
        bounding_boxes = storage.load_bounding_boxes(filename)

        print(f"\n{image_path}:")
        if not bounding_boxes:
            print("  No bounding boxes found")
        else:
            print(f"  {len(bounding_boxes)} bounding box(es):")
            for i, bbox in enumerate(bounding_boxes, 1):
                print(f"    Box {i} (ID: {bbox.box_id}):")
                print(f"      Corners: {bbox.corners.tolist()}")
                if bbox.attributes.date_hint:
                    print(f"      Date: {bbox.attributes.date_hint}")
                if bbox.attributes.exif_date:
                    print(f"      Date: {bbox.attributes.exif_date}")
                if bbox.attributes.comments:
                    print(f"      Comments: {bbox.attributes.comments}")

    return 0
