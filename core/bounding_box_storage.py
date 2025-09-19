"""
Persistent storage for bounding box data per directory.
"""

import json
import platform
from pathlib import Path
from typing import Any, Optional

from core.bounding_box import BoundingBox


class BoundingBoxStorage:
    """Handles saving and loading bounding box data for images in a directory."""

    def __init__(self, directory: str, json_file_name: Optional[str] = None) -> None:
        self.directory = directory
        self._directory_path = Path(directory)

        # Use platform-specific filename to avoid Windows permission issues
        if json_file_name is None:
            if "Windows" in platform.system():
                json_file_name = "photo_extractor_data.json"
            else:
                json_file_name = ".photo_extractor_data.json"

        self._data_file_path = self._directory_path / json_file_name
        self.data_file = str(self._data_file_path)

        self.data: dict[str, list[dict[str, Any]]] = self._load_data()

    def _load_data(self) -> dict[str, list[dict[str, Any]]]:
        """Load bounding box data from JSON file."""
        if self._data_file_path.exists():
            try:
                with open(self.data_file) as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                return {}
        return {}

    def save_data(self) -> None:
        """Save bounding box data to JSON file."""
        with open(self.data_file, "w") as f:
            json.dump(self.data, f, indent=2)

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
            if not (self._directory_path / filename).exists():
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
            self.data[image_filename] = []

        updated_dict = bounding_box_data.to_dict()

        box_exists = False
        for i, saved_box_data in enumerate(self.data[image_filename]):
            if saved_box_data.get("id") == bounding_box_data.box_id:
                self.data[image_filename][i] = updated_dict
                box_exists = True
                break
        if not box_exists:
            self.data[image_filename].append(updated_dict)
        if save_data:
            self.save_data()
        return True
