"""
Persistent storage for bounding box data per directory.
"""

import os
import json
from gui.quad_bounding_box import QuadBoundingBox


class BoundingBoxStorage:
    """Handles saving and loading bounding box data for images in a directory."""
    
    def __init__(self, directory):
        self.directory = directory
        self.data_file = os.path.join(directory, '.photo_extractor_data.json')
        self.data = self.load_data()
        
    def load_data(self):
        """Load bounding box data from JSON file."""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}
        
    def save_data(self):
        """Save bounding box data to JSON file."""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.data, f, indent=2)
        except IOError:
            print(f"Warning: Could not save bounding box data to {self.data_file}")
            
    def save_bounding_boxes(self, image_filename, bounding_boxes):
        """Save bounding boxes for a specific image."""
        if not bounding_boxes:
            # Remove entry if no bounding boxes
            self.data.pop(image_filename, None)
        else:
            # Convert bounding boxes to serializable format
            box_data = []
            for box in bounding_boxes:
                if isinstance(box, QuadBoundingBox):
                    corners = box.get_corner_points_for_extraction()
                    corner_coords = [[corner.x(), corner.y()] for corner in corners]
                    box_data.append({'type': 'quad', 'corners': corner_coords})
            self.data[image_filename] = box_data
        self.save_data()
        
    def load_bounding_boxes(self, image_filename):
        """Load bounding boxes for a specific image."""
        return self.data.get(image_filename, [])