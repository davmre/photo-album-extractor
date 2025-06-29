"""
Persistent storage for bounding box data per directory.
"""

import json
import os
from typing import Any

from core.bounding_box import BoundingBox


class BoundingBoxStorage:
    """Handles saving and loading bounding box data for images in a directory."""

    def __init__(self, directory: str) -> None:
        self.directory = directory
        self.data_file = os.path.join(directory, ".photo_extractor_data.json")
        self.data: dict[str, list[dict[str, Any]]] = self._load_data()

    def _load_data(self) -> dict[str, list[dict[str, Any]]]:
        """Load bounding box data from JSON file."""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file) as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                return {}
        return {}

    def save_data(self) -> None:
        """Save bounding box data to JSON file."""
        try:
            with open(self.data_file, "w") as f:
                json.dump(self.data, f, indent=2)
        except OSError:
            print(f"Warning: Could not save bounding box data to {self.data_file}")

    def set_bounding_boxes(
        self,
        image_filename: str,
        bounding_boxes: list[BoundingBox],
        save_data: bool = True,
    ) -> None:
        """Save bounding boxes for a specific image."""
        if not bounding_boxes:
            # Remove entry if no bounding boxes
            self.data.pop(image_filename, None)
        else:
            # Convert bounding boxes to serializable format
            boxes_dicts = [bbox_data.to_dict() for bbox_data in bounding_boxes]
            self.data[image_filename] = boxes_dicts
        if save_data:
            self.save_data()

    def load_image_filenames(self) -> list[str]:
        return list(self.data.keys())

    def clear_nonexistent_images(self):
        filenames = self.load_image_filenames()
        for filename in filenames:
            if not os.path.exists(os.path.join(self.directory, filename)):
                del self.data[filename]
        self.save_data()

    def get_bounding_boxes(self, image_filename: str) -> list[BoundingBox]:
        """Load bounding box data with IDs for a specific image."""
        boxes = self.data.get(image_filename, [])
        return [BoundingBox.from_dict(box_dict) for box_dict in boxes]

    def update_box_data(
        self, image_filename: str, bounding_box_data: BoundingBox, save_data=True
    ) -> bool:
        """Update complete bounding box data for a specific box."""
        if image_filename not in self.data:
            return False

        updated_dict = bounding_box_data.to_dict()

        for i, saved_box_data in enumerate(self.data[image_filename]):
            if saved_box_data.get("id") == bounding_box_data.box_id:
                self.data[image_filename][i] = updated_dict
                if save_data:
                    self.save_data()
                return True
        return False
