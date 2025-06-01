"""
Image processing utilities for loading, cropping, and saving photos.
"""

import os
import numpy as np
from datetime import datetime
from PIL import Image, ImageQt
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import QRectF
import piexif


def extract_perspective_image(image, corner_points, output_width=None, output_height=None):
    """Crop image using four corner points."""
    # Convert corner points to numpy array
    corners = np.array(corner_points, dtype=np.float32)
    
    # Calculate the width and height of the output rectangle
    # Use the maximum dimensions to avoid losing any image data
    if output_width is None:
        width1 = np.linalg.norm(corners[1] - corners[0])
        width2 = np.linalg.norm(corners[2] - corners[3])
        output_width = int(max(width1, width2))
    if output_height is None:
        height1 = np.linalg.norm(corners[3] - corners[0])
        height2 = np.linalg.norm(corners[2] - corners[1])
        output_height = int(max(height1, height2))
    
    # Define the output rectangle corners (in order: top-left, top-right, bottom-right, bottom-left)
    output_corners = np.array([
        [0, 0],
        [output_width, 0],
        [output_width, output_height],
        [0, output_height]
    ], dtype=np.float32)
    
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

    # Apply perspective transformation
    return image.transform(
        (output_width, output_height),
        Image.Transform.PERSPECTIVE,
        coeffs,
        Image.Resampling.BICUBIC
    )


def load_image(filepath):
    return Image.open(filepath)


def pil_image_as_pixmap(image: Image.Image) -> QPixmap:
    """Convert image to QPixmap for display."""
    image_qt = ImageQt.ImageQt(image)
    return QPixmap.fromImage(image_qt)
    
    # Convert PIL image to QPixmap using temporary file approach
    import io
    img_io = io.BytesIO()

    # Convert to RGB if needed and save as PNG
    if image.mode in ('RGBA', 'LA'):
        # Handle transparency by converting to RGB with white background
        background = Image.new('RGB', self.current_image.size, (255, 255, 255))
        if image.mode == 'RGBA':
            background.paste(image, mask=image.split()[-1])
        else:
            background.paste(image)
        background.save(img_io, format='PNG')
    elif image.mode != 'RGB':
        # Convert other modes to RGB
        rgb_image = image.convert('RGB')
        rgb_image.save(img_io, format='PNG')
    else:
        # RGB image
        image.save(img_io, format='PNG')

    img_io.seek(0)
    pixmap = QPixmap()
    success = pixmap.loadFromData(img_io.getvalue())

    if not success:
        print(f"Failed to load pixmap from image data")
        return None
        
    return pixmap

def save_cropped_images(image: Image.Image,
                        crop_data, output_dir, base_name="photo",
                        attributes_list=None):
    """Save multiple cropped images to the specified directory.
    
    Args:
        image: image to crop from.
        crop_data: List of quadrilaterals to crop. Each quadrilateral is
            a list of four `(x, y)` corner points in relative coordinates.
        output_dir: Directory to save images
        base_name: Base name for files
        attributes_list: List of attribute dictionaries (one per crop)
    """
    saved_files = []
    img_width, img_height = image.size

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    for i, crop_rect in enumerate(crop_data):
        cropped = None
        
        # Get attributes for this crop if available
        attributes = {}
        if attributes_list and i-1 < len(attributes_list):
            attributes = attributes_list[i-1] or {}
        
        # Convert relative coordinates to absolute pixel coordinates for quadrilateral
        abs_corners = []
        for rel_x, rel_y in crop_rect:
            abs_x = rel_x * img_width
            abs_y = rel_y * img_height
            abs_corners.append((abs_x, abs_y))
        cropped = extract_perspective_image(image, abs_corners)
            
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
            # Convert to RGB if necessary
            if cropped.mode != 'RGB':
                cropped = cropped.convert('RGB')

            save_image_with_exif(cropped, filepath, attributes)
            saved_files.append(filepath)
            print(f"Saved: {filename}")
        except Exception as e:
            print(f"Error saving {filename}: {e}")
            
    return saved_files
    
def save_image_with_exif(image, filepath, attributes, jpeg_quality=95):
    """Save image with EXIF data from attributes."""
    try:
        # Create EXIF dictionary
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
        
        # Add date/time if available
        if 'date_time' in attributes and attributes['date_time']:
            try:
                # Parse ISO date string and convert to EXIF format
                dt = datetime.fromisoformat(attributes['date_time'].replace('Z', '+00:00'))
                exif_datetime = dt.strftime("%Y:%m:%d %H:%M:%S")
                
                # Set multiple date fields for maximum compatibility
                exif_dict["0th"][piexif.ImageIFD.DateTime] = exif_datetime
                exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = exif_datetime
                exif_dict["Exif"][piexif.ExifIFD.DateTimeDigitized] = exif_datetime
            except (ValueError, AttributeError) as e:
                print(f"Warning: Could not parse date '{attributes['date_time']}': {e}")
        
        # Add comments if available
        if 'comments' in attributes and attributes['comments']:
            comments = attributes['comments'][:65535]  # EXIF comment limit
            # Use UserComment for better unicode support
            exif_dict["Exif"][piexif.ExifIFD.UserComment] = comments.encode('utf-8')
            # Also set ImageDescription for wider compatibility
            exif_dict["0th"][piexif.ImageIFD.ImageDescription] = comments
        
        # Add software tag
        exif_dict["0th"][piexif.ImageIFD.Software] = "Photo Album Extractor"
        
        # Convert EXIF dictionary to bytes
        exif_bytes = piexif.dump(exif_dict)
        
        # Save image with EXIF data
        image.save(filepath, 'JPEG', quality=jpeg_quality, exif=exif_bytes)
        
    except Exception as e:
        print(f"Warning: Could not write EXIF data to {filepath}: {e}")
        # Fall back to saving without EXIF
        image.save(filepath, 'JPEG', quality=jpeg_quality)
