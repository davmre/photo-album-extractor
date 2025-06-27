"""
Date inference and consistency checking for bounding box data.
"""

from __future__ import annotations

from datetime import datetime
from typing import NamedTuple

import numpy as np

from core import date_utils
from core.bounding_box_data import BoundingBoxData
from core.bounding_box_storage import BoundingBoxStorage


class DateInferenceResult(NamedTuple):
    """Result of running date inference on a directory."""

    updated_files: set[str]  # Files that had date changes
    total_boxes_updated: int
    inconsistent_boxes_updated: int


def infer_dates_for_directory(
    storage: BoundingBoxStorage, trial_run: bool = False
) -> DateInferenceResult:
    """
    Infer dates and check consistency for all bounding boxes in a directory.

    This implements the same logic as cli/interpolate_dates.py but works with
    an existing BoundingBoxStorage object to avoid reloading from disk.

    Args:
        storage: BoundingBoxStorage object for the directory
        trial_run: If True, don't save changes, just return what would be updated

    Returns:
        DateInferenceResult with information about what was updated
    """
    storage.clear_nonexistent_images()
    image_filenames = sorted(storage.load_image_filenames())

    if not image_filenames:
        return DateInferenceResult(set(), 0, 0)

    # Construct a list of intervals giving possible dates for each photo
    photo_filenames: list[str] = []
    photo_boxes: list[BoundingBoxData] = []
    date_intervals: list[tuple[datetime, datetime] | None] = []

    for image_filename in image_filenames:
        boxes = storage.get_bounding_boxes(image_filename)
        # Order boxes on each page by their minimum y-coord, top-to-bottom
        boxes = sorted(boxes, key=lambda b: np.min(b.corners, axis=0)[1])

        # Parse a date interval (or None) for each image
        for box in boxes:
            date_interval = date_utils.parse_flexible_date_as_interval(
                box.attributes.date_hint
            )
            photo_filenames.append(image_filename)
            photo_boxes.append(box)
            date_intervals.append(date_interval)

    if not photo_boxes:
        return DateInferenceResult(set(), 0, 0)

    # Run date interpolation to get consistent dates
    interpolated_date_segments = date_utils.interpolate_dates_segmented(date_intervals)
    interpolated_dates = [d for segment in interpolated_date_segments for d in segment]

    # Calculate segment boundaries (these mark date inconsistencies)
    segment_boundaries = set(
        np.cumsum([len(segment) for segment in interpolated_date_segments])
    )
    segment_boundary_indicators = [
        i in segment_boundaries for i in range(len(interpolated_dates))
    ]

    # Track what gets updated
    updated_files = set()
    total_boxes_updated = 0
    inconsistent_boxes_updated = 0

    # Update the bounding boxes with interpolated dates and consistency flags
    for filename, box, date, is_segment_boundary in zip(
        photo_filenames, photo_boxes, interpolated_dates, segment_boundary_indicators
    ):
        updated_date_str = date.strftime("%Y-%m-%d %H:%M:%S")

        # Check if anything actually changed
        date_changed = box.attributes.exif_date != updated_date_str
        consistency_changed = box.attributes.date_inconsistent != is_segment_boundary

        if date_changed or consistency_changed:
            updated_files.add(filename)
            total_boxes_updated += 1

            if consistency_changed and is_segment_boundary:
                inconsistent_boxes_updated += 1

            if not trial_run:
                # Update the box attributes
                box.attributes.exif_date = updated_date_str
                box.attributes.date_inconsistent = is_segment_boundary

                # Save to storage (but defer file write)
                storage.update_box_data(filename, box, save_data=False)

    # Save all changes to disk if not a trial run
    if not trial_run and updated_files:
        storage.save_data()

    return DateInferenceResult(
        updated_files=updated_files,
        total_boxes_updated=total_boxes_updated,
        inconsistent_boxes_updated=inconsistent_boxes_updated,
    )


def infer_dates_for_current_directory(
    storage: BoundingBoxStorage,
) -> DateInferenceResult:
    """
    Convenience function to infer dates for the current directory.

    Args:
        storage: BoundingBoxStorage object for the directory

    Returns:
        DateInferenceResult with information about what was updated
    """
    return infer_dates_for_directory(storage, trial_run=False)
