"""
Extract command implementation for CLI.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import PIL.Image

from cli.utils import get_image_files
from core.bounding_box_storage import BoundingBoxStorage
from core.images import save_cropped_images


def cmd_extract(paths: list[str], output_dir: str, base_name: str | None) -> int:
    """Extract photos from images using stored bounding box data."""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting extraction for {len(paths)} path(s) to {output_dir}")

    image_files = get_image_files(paths)
    if image_files is None:
        return 1

    if not image_files:
        print("No image files found")
        return 0

    # Validate and create output directory
    try:
        p = Path(output_dir).expanduser()
        if not p.exists():
            p.mkdir(parents=True)
        output_dir = str(p)
        print(f"Output directory: {output_dir}")
    except OSError as e:
        print(f"Error creating output directory '{output_dir}': {e}")
        return 1

    total_extracted = 0
    processed_count = 0
    skipped_count = 0
    error_count = 0

    for image_path in image_files:
        directory = image_path.parent
        filename = image_path.name
        source_base_name = image_path.stem  # filename without extension

        # Load bounding box data
        storage = BoundingBoxStorage(str(directory))
        bounding_boxes = storage.get_bounding_boxes(filename)

        if not bounding_boxes:
            print(f"Skipping {image_path} (no bounding boxes found)")
            skipped_count += 1
            continue

        # Load and process image
        try:
            print(f"Processing {image_path} ({len(bounding_boxes)} photos)...")
            with PIL.Image.open(image_path) as image:
                # Generate base name for extracted files
                extract_base_name = base_name if base_name else source_base_name

                # Extract and save photos
                saved_files = save_cropped_images(
                    image=image,
                    bounding_box_data_list=bounding_boxes,
                    output_dir=output_dir,
                    base_name=extract_base_name,
                    source_image_path=str(image_path),
                )

                if saved_files:
                    print(f"  Extracted {len(saved_files)} photos:")
                    for saved_file in saved_files:
                        print(f"    {os.path.basename(saved_file)}")
                    total_extracted += len(saved_files)
                    processed_count += 1
                else:
                    print("  No photos were extracted")
                    error_count += 1

        except Exception as e:
            print(f"  Error processing {image_path}: {e}")
            error_count += 1

    # Print summary
    print("\nSummary:")
    print(f"  Source images processed: {processed_count}")
    print(f"  Total photos extracted: {total_extracted}")
    if skipped_count > 0:
        print(f"  Skipped: {skipped_count} images (no bounding boxes)")
    if error_count > 0:
        print(f"  Errors: {error_count} images")

    return 0 if error_count == 0 else 1
