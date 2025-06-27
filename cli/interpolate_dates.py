"""
Date interpolation command for CLI.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from core import date_utils
from core.bounding_box_data import BoundingBoxData
from core.bounding_box_storage import BoundingBoxStorage


def cmd_interpolate_dates(directory: str, trial_run=False):
    path = Path(directory)
    if not path.is_dir():
        raise ValueError(f"Specified location {directory} is not a directory!")

    storage = BoundingBoxStorage(directory)
    storage.clear_nonexistent_images()
    image_filenames = sorted(storage.load_image_filenames())

    # Construct a list of intervals giving possible dates for each photo.
    photo_filenames: list[str] = []
    photo_boxes: list[BoundingBoxData] = []
    date_intervals: list[tuple[datetime, datetime] | None] = []

    for image_filename in image_filenames:
        boxes = storage.load_bounding_boxes(image_filename)
        # Order boxes on each page by their minimum y-coord, top-to-bottom.
        boxes = sorted(boxes, key=lambda b: np.min(b.corners, axis=0)[1])
        # Parse a date interval (or None) for each image.
        for box in boxes:
            date_interval = date_utils.parse_flexible_date_as_interval(
                box.attributes.date_hint
            )
            photo_filenames.append(image_filename)
            photo_boxes.append(box)
            date_intervals.append(date_interval)

    interpolated_date_segments = date_utils.interpolate_dates_segmented(date_intervals)
    interpolated_dates = [d for segment in interpolated_date_segments for d in segment]
    segment_boundaries = set(
        np.cumsum([len(segment) for segment in interpolated_date_segments])
    )
    segment_boundary_indicators = [
        i in segment_boundaries for i in range(len(interpolated_dates))
    ]

    # Save the interpolated dates in the Exif field (the original hints stay in the
    # `date_hint` field).
    for filename, box, date, is_segment_boundary in zip(
        photo_filenames, photo_boxes, interpolated_dates, segment_boundary_indicators
    ):
        updated_date_str = date.strftime("%Y-%m-%d %H:%M:%S")
        if trial_run:
            print(
                f"{filename}:{box.box_id[:6]}: interpolated date {updated_date_str} (hint: {box.attributes.date_hint}) {'NEW SEGMENT' if is_segment_boundary else ''}"
            )
        else:
            print(
                f"{filename}:{box.box_id[:6]}: saving new date {updated_date_str} (hint: {box.attributes.date_hint}) {'NEW SEGMENT' if is_segment_boundary else ''}"
            )
            box.attributes.exif_date = updated_date_str
            box.attributes.date_inconsistent = is_segment_boundary
            # Defer saving so we only write the JSON file at the end once everything's
            # updated.
            storage.update_box_data(filename, box, save_data=False)
    storage._save_data()
