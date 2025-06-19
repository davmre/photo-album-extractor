"""
Persistent storage for bounding box data per directory.
"""

import json
import os
import uuid
from typing import Any

from core.photo_types import PhotoAttributes
from gui.quad_bounding_box import QuadBoundingBox


class BoundingBoxStorage:
    """Handles saving and loading bounding box data for images in a directory."""

    def __init__(self, directory: str) -> None:
        self.directory = directory
        self.data_file = os.path.join(directory, ".photo_extractor_data.json")
        self.data: dict[str, list[dict[str, Any]]] = self.load_data()

    def load_data(self) -> dict[str, list[dict[str, Any]]]:
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

    def save_bounding_boxes(
        self, image_filename: str, bounding_boxes: list[QuadBoundingBox]
    ) -> None:
        """Save bounding boxes for a specific image."""
        if not bounding_boxes:
            # Remove entry if no bounding boxes
            self.data.pop(image_filename, None)
        else:
            # Convert bounding boxes to serializable format
            box_data = []
            for box in bounding_boxes:
                if isinstance(box, QuadBoundingBox):
                    corners = box.get_ordered_corners_for_extraction()
                    corner_coords = [[corner[0], corner[1]] for corner in corners]

                    # Build box data with attributes
                    box_entry = {"type": "quad", "corners": corner_coords}

                    # Add attributes if they exist
                    if hasattr(box, "box_id") and box.box_id:
                        box_entry["id"] = box.box_id

                    if hasattr(box, "attributes") and box.attributes:
                        box_entry["attributes"] = box.attributes.to_dict()

                    box_data.append(box_entry)
            self.data[image_filename] = box_data
        self.save_data()

    def load_bounding_boxes(self, image_filename: str) -> list[dict[str, Any]]:
        """Load bounding boxes for a specific image."""
        return self.data.get(image_filename, [])

    def generate_box_id(self):
        """Generate a unique box ID."""
        return str(uuid.uuid4())

    def get_box_attributes(self, image_filename: str, box_id: str) -> PhotoAttributes:
        """Get attributes for a specific box."""
        boxes = self.load_bounding_boxes(image_filename)
        for box_data in boxes:
            if box_data.get("id") == box_id:
                attributes_dict = box_data.get("attributes", {})
                return PhotoAttributes.from_dict(attributes_dict)
        return PhotoAttributes()

    def update_box_attributes(
        self, image_filename: str, box_id: str, attributes: PhotoAttributes
    ) -> bool:
        """Update attributes for a specific box."""
        if image_filename not in self.data:
            return False

        attributes_dict = attributes.to_dict()

        for box_data in self.data[image_filename]:
            if box_data.get("id") == box_id:
                box_data["attributes"] = attributes_dict
                self.save_data()
                return True
        return False
