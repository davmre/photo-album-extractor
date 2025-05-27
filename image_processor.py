"""
Image processing utilities for loading, cropping, and saving photos.
"""

import os
import numpy as np
from PIL import Image
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import QRectF

class ImageProcessor:
    """Handles image loading, cropping, and batch saving operations."""
    
    def __init__(self):
        self.current_image = None
        self.image_path = None
        
    def load_image(self, file_path):
        """Load an image from file path."""
        try:
            self.current_image = Image.open(file_path)
            self.image_path = file_path
            return True
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
            
    def get_pixmap(self):
        """Convert current image to QPixmap for display."""
        if self.current_image is None:
            return None
            
        # Convert PIL image to QPixmap using temporary file approach
        import io
        img_io = io.BytesIO()
        
        # Convert to RGB if needed and save as PNG
        if self.current_image.mode in ('RGBA', 'LA'):
            # Handle transparency by converting to RGB with white background
            background = Image.new('RGB', self.current_image.size, (255, 255, 255))
            if self.current_image.mode == 'RGBA':
                background.paste(self.current_image, mask=self.current_image.split()[-1])
            else:
                background.paste(self.current_image)
            background.save(img_io, format='PNG')
        elif self.current_image.mode != 'RGB':
            # Convert other modes to RGB
            rgb_image = self.current_image.convert('RGB')
            rgb_image.save(img_io, format='PNG')
        else:
            # RGB image
            self.current_image.save(img_io, format='PNG')
            
        img_io.seek(0)
        pixmap = QPixmap()
        success = pixmap.loadFromData(img_io.getvalue())
        
        if not success:
            print(f"Failed to load pixmap from image data")
            return None
            
        return pixmap
        
    def crop_image(self, crop_rect, scale_factor=1.0):
        """Crop image using the provided rectangle."""
        if self.current_image is None:
            return None
            
        # Convert QRectF to PIL crop box (left, top, right, bottom)
        # Apply scale factor to convert from display coordinates to image coordinates
        left = int(crop_rect.x() * scale_factor)
        top = int(crop_rect.y() * scale_factor)
        right = int((crop_rect.x() + crop_rect.width()) * scale_factor)
        bottom = int((crop_rect.y() + crop_rect.height()) * scale_factor)
        
        # Ensure coordinates are within image bounds
        img_width, img_height = self.current_image.size
        left = max(0, min(left, img_width))
        top = max(0, min(top, img_height))
        right = max(left, min(right, img_width))
        bottom = max(top, min(bottom, img_height))
        
        # Crop the image
        cropped = self.current_image.crop((left, top, right, bottom))
        return cropped
        
    def crop_rotated_image(self, corner_points):
        """Crop image using four corner points (for rotated rectangles)."""
        if self.current_image is None:
            return None
            
        # Convert corner points to numpy array
        corners = np.array(corner_points, dtype=np.float32)
        
        # Calculate the width and height of the output rectangle
        # Use the maximum dimensions to avoid losing any image data
        width1 = np.linalg.norm(corners[1] - corners[0])
        width2 = np.linalg.norm(corners[2] - corners[3])
        height1 = np.linalg.norm(corners[3] - corners[0])
        height2 = np.linalg.norm(corners[2] - corners[1])
        
        output_width = int(max(width1, width2))
        output_height = int(max(height1, height2))
        
        # Define the output rectangle corners (in order: top-left, top-right, bottom-right, bottom-left)
        output_corners = np.array([
            [0, 0],
            [output_width, 0],
            [output_width, output_height],
            [0, output_height]
        ], dtype=np.float32)
        
        # Use PIL's transform with perspective correction
        from PIL import Image
        
        # Calculate transformation matrix
        # We need to map from output_corners to corners
        def find_coeffs(pa, pb):
            matrix = []
            for p1, p2 in zip(pa, pb):
                matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
                matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
            A = np.matrix(matrix, dtype=np.float32)
            B = np.array(corners.flatten(), dtype=np.float32)
            try:
                res = np.dot(np.linalg.inv(A.T @ A) @ A.T, B)
                return np.array(res).flatten()
            except:
                return None
                
        coeffs = find_coeffs(output_corners, corners)
        if coeffs is None:
            return None
            
        # Apply perspective transformation
        transformed = self.current_image.transform(
            (output_width, output_height),
            Image.Transform.PERSPECTIVE,
            coeffs,
            Image.Resampling.BICUBIC
        )
        
        return transformed

    def save_cropped_images(self, crop_data, output_dir, base_name="photo"):
        """Save multiple cropped images to the specified directory."""
        if self.current_image is None:
            return []
            
        saved_files = []
        img_width, img_height = self.current_image.size
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        for i, data in enumerate(crop_data, 1):
            crop_type, crop_info = data
            cropped = None
            
            if crop_type == 'rotated':
                # Convert relative coordinates to absolute pixel coordinates
                abs_corners = []
                for rel_x, rel_y in crop_info:
                    abs_x = rel_x * img_width
                    abs_y = rel_y * img_height
                    abs_corners.append((abs_x, abs_y))
                cropped = self.crop_rotated_image(abs_corners)
            elif crop_type == 'rect':
                # Handle regular rectangle
                rect = crop_info
                abs_rect = QRectF(
                    rect.x() * img_width,
                    rect.y() * img_height,
                    rect.width() * img_width,
                    rect.height() * img_height
                )
                cropped = self.crop_image(abs_rect)
                
            if cropped is None:
                continue
                
            # Generate filename
            filename = f"{base_name}_{i:03d}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            # Ensure unique filename
            counter = 1
            while os.path.exists(filepath):
                filename = f"{base_name}_{i:03d}_{counter}.jpg"
                filepath = os.path.join(output_dir, filename)
                counter += 1
                
            try:
                # Convert to RGB if necessary and save
                if cropped.mode != 'RGB':
                    cropped = cropped.convert('RGB')
                cropped.save(filepath, 'JPEG', quality=95)
                saved_files.append(filepath)
                print(f"Saved: {filename}")
            except Exception as e:
                print(f"Error saving {filename}: {e}")
                
        return saved_files
        
    def get_image_size(self):
        """Get the size of the current image."""
        if self.current_image is None:
            return (0, 0)
        return self.current_image.size