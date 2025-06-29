"""
Detection command implementation for CLI.
"""

from __future__ import annotations

import logging

import PIL.Image

from cli.utils import get_image_files
from core.bounding_box_storage import BoundingBoxStorage
from core.detection_strategies import configure_detection_strategy
from core.settings import app_settings


def cmd_detect(paths: list[str], force: bool = False) -> int:
    """Run photo detection on specified images."""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting detection on {len(paths)} path(s)")

    image_files = get_image_files(paths)
    if image_files is None:
        return 1

    if not image_files:
        print("No image files found")
        return 0

    # Configure detection strategy using global settings
    try:
        detection_strategy = configure_detection_strategy(app_settings)
        logger.info(f"Configured detection strategy: {detection_strategy.name}")
    except Exception as e:
        print(f"Error configuring detection strategy: {e}")
        return 1

    processed_count = 0
    skipped_count = 0
    error_count = 0

    for image_path in image_files:
        directory = image_path.parent
        filename = image_path.name

        # Check if bounding boxes already exist
        storage = BoundingBoxStorage(str(directory))
        existing_boxes = storage.get_bounding_boxes(filename)

        if existing_boxes and not force:
            print(
                f"Skipping {image_path} (already has {len(existing_boxes)} bounding boxes, use --force to overwrite)"
            )
            skipped_count += 1
            continue

        # Load and process image
        try:
            logger.info(f"Processing image: {image_path}")
            print(f"Processing {image_path}...")
            with PIL.Image.open(image_path) as image:
                # Run detection
                detected_boxes = detection_strategy.detect_photos(image)

                if detected_boxes:
                    # Save results
                    storage.set_bounding_boxes(filename, detected_boxes)
                    print(f"  Found and saved {len(detected_boxes)} bounding box(es)")
                    processed_count += 1
                else:
                    print("  No photos detected")
                    # Save empty list to mark as processed
                    storage.set_bounding_boxes(filename, [])
                    processed_count += 1

        except Exception as e:
            print(f"  Error processing {image_path}: {e}")
            error_count += 1

    # Print summary
    print("\nSummary:")
    print(f"  Processed: {processed_count} images")
    if skipped_count > 0:
        print(f"  Skipped: {skipped_count} images (already had data)")
    if error_count > 0:
        print(f"  Errors: {error_count} images")

    return 0 if error_count == 0 else 1
