"""
Persistent storage for bounding box data per directory.
"""

import json
import os
from typing import Any

from core.photo_types import BoundingBoxData, PhotoAttributes


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
        self, image_filename: str, bounding_boxes: list[BoundingBoxData]
    ) -> None:
        """Save bounding boxes for a specific image."""
        if not bounding_boxes:
            # Remove entry if no bounding boxes
            self.data.pop(image_filename, None)
        else:
            # Convert bounding boxes to serializable format
            boxes_dicts = []
            for bbox_data in bounding_boxes:
                # Get unified bounding box data
                corner_coords = [[corner[0], corner[1]] for corner in bbox_data.corners]

                # Build box data entry
                box_entry = {
                    "type": "quad",
                    "corners": corner_coords,
                    "attributes": bbox_data.attributes.to_dict(),
                    "id": bbox_data.box_id,
                }
                boxes_dicts.append(box_entry)
            self.data[image_filename] = boxes_dicts
        self.save_data()

    def load_bounding_boxes(self, image_filename: str) -> list[BoundingBoxData]:
        """Load bounding box data with IDs for a specific image."""
        boxes = self.data.get(image_filename, [])
        result = []

        for box_dict in boxes:
            # Extract corners and convert to numpy array
            corners = box_dict.get("corners", [])
            import numpy as np

            corners_array = np.array(corners, dtype=np.float64)

            # Extract attributes
            attributes_dict = box_dict.get("attributes", {})
            attributes = PhotoAttributes.from_dict(attributes_dict)

            # Create BoundingBoxData
            bbox_data = BoundingBoxData(
                corners=corners_array,
                box_id=box_dict["id"],
                attributes=attributes,
            )

            result.append(bbox_data)

        return result

    def update_box_data(
        self, image_filename: str, bounding_box_data: BoundingBoxData
    ) -> bool:
        """Update complete bounding box data for a specific box."""
        if image_filename not in self.data:
            return False

        corner_coords = [[corner[0], corner[1]] for corner in bounding_box_data.corners]
        attributes_dict = bounding_box_data.attributes.to_dict()

        for saved_box_data in self.data[image_filename]:
            if saved_box_data.get("id") == bounding_box_data.box_id:
                saved_box_data["corners"] = corner_coords
                saved_box_data["attributes"] = attributes_dict
                self.save_data()
                return True
        return False
