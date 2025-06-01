"""
Persistent storage for bounding box data per directory.
"""

import os
import json
import uuid
from datetime import datetime
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
                    
                    # Build box data with attributes
                    box_entry = {
                        'type': 'quad', 
                        'corners': corner_coords
                    }
                    
                    # Add attributes if they exist
                    if hasattr(box, 'box_id') and box.box_id:
                        box_entry['id'] = box.box_id
                    
                    if hasattr(box, 'attributes') and box.attributes:
                        box_entry['attributes'] = box.attributes.copy()
                    
                    box_data.append(box_entry)
            self.data[image_filename] = box_data
        self.save_data()
        
    def load_bounding_boxes(self, image_filename):
        """Load bounding boxes for a specific image."""
        return self.data.get(image_filename, [])
        
    def generate_box_id(self):
        """Generate a unique box ID."""
        return str(uuid.uuid4())
        
    def get_box_attributes(self, image_filename, box_id):
        """Get attributes for a specific box."""
        boxes = self.load_bounding_boxes(image_filename)
        for box_data in boxes:
            if box_data.get('id') == box_id:
                return box_data.get('attributes', {})
        return {}
        
    def update_box_attributes(self, image_filename, box_id, attributes):
        """Update attributes for a specific box."""
        if image_filename not in self.data:
            return False
            
        for box_data in self.data[image_filename]:
            if box_data.get('id') == box_id:
                box_data['attributes'] = attributes.copy()
                self.save_data()
                return True
        return False