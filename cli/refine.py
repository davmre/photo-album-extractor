"""
Detection command implementation for CLI.
"""

from __future__ import annotations

import logging

import PIL.Image

from cli.utils import get_image_files
from core.bounding_box_storage import BoundingBoxStorage
from core.settings import AppSettings
from photo_detection.refinement_strategies import (
    REFINEMENT_STRATEGIES,
    configure_refinement_strategy,
)


def cmd_refine(
    paths: list[str], strategy: str | None, debug_dir: str | None = None
) -> int:
    """Run photo detection on specified images."""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting refinement on {len(paths)} path(s)")

    if not debug_dir:  # Standardize empty string as None
        debug_dir = None

    image_files = get_image_files(paths)
    if image_files is None:
        return 1

    if not image_files:
        print("No image files found")
        return 0

    # Load settings and configure detection strategy
    try:
        if strategy:
            refinement_strategy = None
            for k in REFINEMENT_STRATEGIES.keys():
                if strategy in k:
                    refinement_strategy = REFINEMENT_STRATEGIES[k]
                    break
            if not refinement_strategy:
                raise ValueError(
                    f"Unrecognized strategy {k}. Options (can specify by any substring): {list(REFINEMENT_STRATEGIES.keys())}"
                )
        else:
            settings = AppSettings.load_from_file()
            refinement_strategy = configure_refinement_strategy(settings)
        logger.info(f"Configured refinement strategy: {refinement_strategy.name}")
    except Exception as e:
        print(f"Error configuring refinement strategy: {e}")
        return 1

    processed_count = 0
    skipped_count = 0
    error_count = 0

    for image_path in image_files:
        directory = image_path.parent
        filename = image_path.name

        print(f"Processing {image_path}...")

        # Load and process image
        try:
            image = PIL.Image.open(image_path)

            storage = BoundingBoxStorage(str(directory))
            boxes = storage.get_bounding_boxes(filename)
            refined_boxes = []
            for box in boxes:
                # Run detection
                refined = refinement_strategy.refine(
                    image, corner_points=box.corners, debug_dir=debug_dir
                )
                box.corners = refined
                refined_boxes.append(box)
                print(f"  Refined box id {box.box_id}")

            storage.set_bounding_boxes(filename, refined_boxes)
            print(f"  Refined {len(refined_boxes)} bounding box(es)")
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
