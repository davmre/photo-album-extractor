"""
Photo detection strategies for automatically identifying photos in scanned album pages.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import cv2
import numpy as np
from PIL import Image
from PyQt6.QtCore import QPointF
import base64
import io
import google.generativeai as genai
import json


class DetectionStrategy(ABC):
    """Base class for photo detection strategies."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the strategy."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Brief description of what the strategy does."""
        pass
    
    @abstractmethod
    def detect_photos(self, image_width: int, image_height: int, image_path: str = None) -> List[List[QPointF]]:
        """
        Detect photos in an image and return their bounding quadrilaterals.
        
        Args:
            image_width: Width of the image in pixels
            image_height: Height of the image in pixels
            image_path: Optional path to the image file (for strategies that need to load the image)
            
        Returns:
            List of quadrilaterals, where each quadrilateral is a list of 4 QPointF
            coordinates in relative coordinates (0.0 to 1.0)
        """
        pass


class OneBoxStrategy(DetectionStrategy):
    """Single box covering the entire image."""
    
    @property
    def name(self) -> str:
        return "One Box"
    
    @property
    def description(self) -> str:
        return "Single bounding box covering entire image"
    
    def detect_photos(self, image_width: int, image_height: int, image_path: str = None) -> List[List[QPointF]]:
        # Create a box with small margin from edges
        margin = 0.05  # 5% margin on all sides
        return [[
            QPointF(margin, margin),                    # Top-left
            QPointF(1.0 - margin, margin),             # Top-right
            QPointF(1.0 - margin, 1.0 - margin),       # Bottom-right
            QPointF(margin, 1.0 - margin)              # Bottom-left
        ]]


class TwoBoxStrategy(DetectionStrategy):
    """Two boxes splitting the image horizontally."""
    
    @property
    def name(self) -> str:
        return "Two Boxes (Horizontal)"
    
    @property
    def description(self) -> str:
        return "Split image horizontally into two equal boxes"
    
    def detect_photos(self, image_width: int, image_height: int, image_path: str = None) -> List[List[QPointF]]:
        margin = 0.05
        center = 0.5
        gap = 0.02  # Small gap between boxes
        
        # Left box
        left_box = [
            QPointF(margin, margin),
            QPointF(center - gap, margin),
            QPointF(center - gap, 1.0 - margin),
            QPointF(margin, 1.0 - margin)
        ]
        
        # Right box
        right_box = [
            QPointF(center + gap, margin),
            QPointF(1.0 - margin, margin),
            QPointF(1.0 - margin, 1.0 - margin),
            QPointF(center + gap, 1.0 - margin)
        ]
        
        return [left_box, right_box]


class FourBoxStrategy(DetectionStrategy):
    """Four boxes in a 2x2 grid."""
    
    @property
    def name(self) -> str:
        return "Four Boxes (2x2 Grid)"
    
    @property
    def description(self) -> str:
        return "Split image into 2x2 grid of four equal boxes"
    
    def detect_photos(self, image_width: int, image_height: int, image_path: str = None) -> List[List[QPointF]]:
        margin = 0.05
        center_x = 0.5
        center_y = 0.5
        gap = 0.02
        
        boxes = []
        
        # Top-left box
        boxes.append([
            QPointF(margin, margin),
            QPointF(center_x - gap, margin),
            QPointF(center_x - gap, center_y - gap),
            QPointF(margin, center_y - gap)
        ])
        
        # Top-right box
        boxes.append([
            QPointF(center_x + gap, margin),
            QPointF(1.0 - margin, margin),
            QPointF(1.0 - margin, center_y - gap),
            QPointF(center_x + gap, center_y - gap)
        ])
        
        # Bottom-left box
        boxes.append([
            QPointF(margin, center_y + gap),
            QPointF(center_x - gap, center_y + gap),
            QPointF(center_x - gap, 1.0 - margin),
            QPointF(margin, 1.0 - margin)
        ])
        
        # Bottom-right box
        boxes.append([
            QPointF(center_x + gap, center_y + gap),
            QPointF(1.0 - margin, center_y + gap),
            QPointF(1.0 - margin, 1.0 - margin),
            QPointF(center_x + gap, 1.0 - margin)
        ])
        
        return boxes


class SixBoxStrategy(DetectionStrategy):
    """Six boxes in a 2x3 grid (common photo album layout)."""
    
    @property
    def name(self) -> str:
        return "Six Boxes (2x3 Grid)"
    
    @property
    def description(self) -> str:
        return "Split image into 2x3 grid of six equal boxes"
    
    def detect_photos(self, image_width: int, image_height: int, image_path: str = None) -> List[List[QPointF]]:
        margin = 0.05
        center_x = 0.5
        third_y = 1.0 / 3.0
        two_third_y = 2.0 / 3.0
        gap = 0.015
        
        boxes = []
        
        # Top row
        # Top-left
        boxes.append([
            QPointF(margin, margin),
            QPointF(center_x - gap, margin),
            QPointF(center_x - gap, third_y - gap),
            QPointF(margin, third_y - gap)
        ])
        
        # Top-right
        boxes.append([
            QPointF(center_x + gap, margin),
            QPointF(1.0 - margin, margin),
            QPointF(1.0 - margin, third_y - gap),
            QPointF(center_x + gap, third_y - gap)
        ])
        
        # Middle row
        # Middle-left
        boxes.append([
            QPointF(margin, third_y + gap),
            QPointF(center_x - gap, third_y + gap),
            QPointF(center_x - gap, two_third_y - gap),
            QPointF(margin, two_third_y - gap)
        ])
        
        # Middle-right
        boxes.append([
            QPointF(center_x + gap, third_y + gap),
            QPointF(1.0 - margin, third_y + gap),
            QPointF(1.0 - margin, two_third_y - gap),
            QPointF(center_x + gap, two_third_y - gap)
        ])
        
        # Bottom row
        # Bottom-left
        boxes.append([
            QPointF(margin, two_third_y + gap),
            QPointF(center_x - gap, two_third_y + gap),
            QPointF(center_x - gap, 1.0 - margin),
            QPointF(margin, 1.0 - margin)
        ])
        
        # Bottom-right
        boxes.append([
            QPointF(center_x + gap, two_third_y + gap),
            QPointF(1.0 - margin, two_third_y + gap),
            QPointF(1.0 - margin, 1.0 - margin),
            QPointF(center_x + gap, 1.0 - margin)
        ])
        
        return boxes


class OpenCVContourStrategy(DetectionStrategy):
    """OpenCV-based strategy using contour detection to find rectangular shapes."""
    
    @property
    def name(self) -> str:
        return "OpenCV Contour Detection"
    
    @property
    def description(self) -> str:
        return "Use OpenCV to detect rectangular contours that could be photos"
    
    def detect_photos(self, image_width: int, image_height: int, image_path: str = None) -> List[List[QPointF]]:
        if not image_path:
            return []
            
        try:
            # Load image using OpenCV
            img = cv2.imread(image_path)
            if img is None:
                return []
                
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply adaptive threshold to get binary image
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            rectangles = []
            
            for contour in contours:
                # Calculate contour area
                area = cv2.contourArea(contour)
                
                # Filter by area (should be reasonably large to be a photo)
                min_area = (image_width * image_height) * 0.02  # At least 2% of image
                max_area = (image_width * image_height) * 0.8   # At most 80% of image
                
                print("CONTOUR", area, "MIN", min_area, "MAX", max_area)

                
                if area < min_area or area > max_area:
                    continue
                
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                print("approximated with", approx)
                
                # Look for rectangles (4 corners) or approximate rectangles
                if len(approx) >= 4:
                    # If more than 4 points, find the bounding rectangle
                    if len(approx) > 4:
                        x, y, w, h = cv2.boundingRect(contour)
                        corners = [
                            [x, y],
                            [x + w, y],
                            [x + w, y + h],
                            [x, y + h]
                        ]
                    else:
                        # Use the 4 approximated corners
                        corners = approx.reshape(4, 2).tolist()
                    
                    # Convert to relative coordinates and QPointF
                    relative_corners = []
                    for corner in corners:
                        rel_x = corner[0] / image_width
                        rel_y = corner[1] / image_height
                        relative_corners.append(QPointF(rel_x, rel_y))
                    
                    rectangles.append(relative_corners)
            
            # Sort by area (largest first) and limit to reasonable number
            rectangles.sort(key=lambda rect: self._calculate_area(rect), reverse=True)
            return rectangles[:12]  # Limit to 12 photos max
            
        except Exception as e:
            print(f"OpenCV detection error: {e}")
            return []
    
    def _calculate_area(self, corners):
        """Calculate approximate area of a quadrilateral."""
        if len(corners) < 4:
            return 0
        # Simple approximation using bounding box
        min_x = min(c.x() for c in corners)
        max_x = max(c.x() for c in corners)
        min_y = min(c.y() for c in corners)
        max_y = max(c.y() for c in corners)
        return (max_x - min_x) * (max_y - min_y)


class PaddleOCRLayoutStrategy(DetectionStrategy):
    """PaddleOCR-based strategy using layout analysis to detect image regions."""
    
    def __init__(self):
        self._ocr = None
    
    @property
    def ocr(self):
        """Lazy initialization of PaddleOCR to avoid startup delays."""
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR
                # Initialize PaddleOCR with layout analysis enabled
                # use_angle_cls=False and rec=False since we only want layout detection
                self._ocr = PaddleOCR(use_angle_cls=False, lang='en', rec=False, 
                                     use_gpu=False, show_log=False)
            except Exception as e:
                print(f"Failed to initialize PaddleOCR: {e}")
                self._ocr = None
        return self._ocr
    
    @property
    def name(self) -> str:
        return "PaddleOCR Layout Analysis"
    
    @property
    def description(self) -> str:
        return "Use PaddleOCR layout analysis to detect image regions in documents"
    
    def detect_photos(self, image_width: int, image_height: int, image_path: str = None) -> List[List[QPointF]]:
        if not image_path or not self.ocr:
            return []
            
        try:
            # Load image with PIL first
            pil_image = Image.open(image_path)
            
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert PIL image to numpy array for PaddleOCR
            img_array = np.array(pil_image)
            
            # Run OCR with layout analysis
            # This returns text regions, but we can use the same logic to find image-like regions
            results = self.ocr.ocr(img_array, cls=False)
            print("RESULTS", results)
            
            if not results or not results[0]:
                # If no OCR results, fall back to simple grid strategy
                return self._fallback_grid_detection(image_width, image_height)
            
            rectangles = []
            
            # Process OCR results to find potential photo regions
            for line in results[0]:
                if line is None:
                    continue
                    
                # Each line contains [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], (text, confidence)]
                bbox = line[0]  # Bounding box coordinates
                
                # Convert bbox to our quadrilateral format
                corners = []
                for point in bbox:
                    rel_x = point[0] / image_width
                    rel_y = point[1] / image_height
                    corners.append(QPointF(rel_x, rel_y))
                
                # Calculate area to filter small regions
                area = self._calculate_area(corners)
                min_area = 0.01  # At least 1% of image
                max_area = 0.7   # At most 70% of image
                
                if min_area <= area <= max_area:
                    rectangles.append(corners)
            
            # If we found very few regions, supplement with a grid approach
            if len(rectangles) < 2:
                grid_regions = self._fallback_grid_detection(image_width, image_height)
                rectangles.extend(grid_regions)
            
            # Sort by area (largest first) and limit
            rectangles.sort(key=lambda rect: self._calculate_area(rect), reverse=True)
            return rectangles[:8]  # Limit to 8 photos max
            
        except Exception as e:
            print(f"PaddleOCR detection error: {e}")
            # Fall back to grid strategy
            return self._fallback_grid_detection(image_width, image_height)
    
    def _fallback_grid_detection(self, image_width: int, image_height: int) -> List[List[QPointF]]:
        """Fallback to a 2x2 grid if OCR doesn't find regions."""
        margin = 0.1
        center_x = 0.5
        center_y = 0.5
        gap = 0.05
        
        return [
            # Top-left
            [QPointF(margin, margin), QPointF(center_x - gap, margin),
             QPointF(center_x - gap, center_y - gap), QPointF(margin, center_y - gap)],
            # Top-right  
            [QPointF(center_x + gap, margin), QPointF(1.0 - margin, margin),
             QPointF(1.0 - margin, center_y - gap), QPointF(center_x + gap, center_y - gap)],
            # Bottom-left
            [QPointF(margin, center_y + gap), QPointF(center_x - gap, center_y + gap),
             QPointF(center_x - gap, 1.0 - margin), QPointF(margin, 1.0 - margin)],
            # Bottom-right
            [QPointF(center_x + gap, center_y + gap), QPointF(1.0 - margin, center_y + gap),
             QPointF(1.0 - margin, 1.0 - margin), QPointF(center_x + gap, 1.0 - margin)]
        ]
    
    def _calculate_area(self, corners):
        """Calculate approximate area of a quadrilateral."""
        if len(corners) < 4:
            return 0
        # Simple approximation using bounding box
        min_x = min(c.x() for c in corners)
        max_x = max(c.x() for c in corners)
        min_y = min(c.y() for c in corners)
        max_y = max(c.y() for c in corners)
        return (max_x - min_x) * (max_y - min_y)


class GeminiDetectionStrategy(DetectionStrategy):
    """Gemini AI-based strategy for detecting photos in album pages."""
    
    def __init__(self):
        self._model = None
        self._setup_gemini()
    
    def _setup_gemini(self):
        """Initialize the Gemini model."""
        try:
            # Configure the API key
            genai.configure(api_key="AIzaSyBbt3NkCk91s_jb0nOjGUvSeNLdJ9m1OFA")
            self._model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
        except Exception as e:
            print(f"Failed to initialize Gemini: {e}")
            self._model = None
    
    @property
    def name(self) -> str:
        return "Gemini AI Detection"
    
    @property
    def description(self) -> str:
        return "Use Gemini AI to intelligently detect photos in album pages"
    
    def _parse_as_json(self, response):
        # Extract JSON from response (remove any markdown formatting)
        json_text = response.text.strip()
        if json_text.startswith('```json'):
            json_text = json_text[7:]
        if json_text.endswith('```'):
            json_text = json_text[:-3]
        json_text = json_text.strip()
        return json.loads(json_text)
    
    def _rect_from_bbox(self, box_2d):
        y_min, x_min, y_max, x_max = box_2d
        return [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
    
    def _unnormalize_coords(self, coords, normalized_width=1000, normalized_height=1000):
        xscale = 1. / normalized_width
        yscale = 1. / normalized_height
        return [ QPointF(p[0] * xscale, p[1] * yscale) for p in coords]
    
    def detect_photos_with_corners(self, image_width: int, image_height: int, image_path: str = None) -> List[List[QPointF]]:
        if not image_path or not self._model:
            return []
            
        try:
            # Load and encode image
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # Create PIL image for Gemini
            pil_image = Image.open(io.BytesIO(image_data))
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            #pil_image.resize(())
            
            prompt = """This is a scanned page from a photo album. Your task 
to detect the locations of the photos on the page. Output a JSON list of
bounding boxes, one for each photo, where each entry contains the 2D bounding
box in "box_2d" and a label 'photo01', 'photo02', etc.

We are also interested in actually extracting the exact quadrilateral containing
each photo. This quadrilateral will be similar to the bounding box, but not
identical because the photos may be skewed relative to the page orientation.
Unfortunately your technical wrapper is only trained to output bounding boxes.
Thus I would like you to recognize the corners of each photo and return a
trivial bounding box (with no area) for each corner of each photo. That is, if
the upper right corner of the first photo is at position [294, 512], you would
return the bounding box for that corner as `{"box_2d": [294, 512, 294, 512],
"label": "photo01_upper_right_corner"}`. The corner bounding boxes, in the
format `[x_min, y_min, x_max, y_max]`, should always have `x_min=x_max` and
`y_min=y_max`, since they are really serving as a hack to specify just a
specific point in the image.

Return only the JSON response, no additional text."""

            # Generate response
            response = self._model.generate_content([prompt, pil_image])
            
            if not response.text:
                return []
            
            # Parse the JSON response
            try:
                result = self._parse_as_json(response)
                print("GOT RESPONSE")
                print(result)
                
                rectangles = []
                bboxes = {}
                result_by_label = {}
                for entry in result:
                    label = entry.get('label', '')
                    result_by_label[label] = entry
                    
                    if label.startswith('photo') and "corner" not in label:
                        bboxes[label] = entry['box_2d']
                
                print("Parsed bboxes:", bboxes)
                
                for label in bboxes.keys():
                    try:
                        r = [
                            result_by_label[label + "_upper_left_corner"]['box_2d'][:2],
                            result_by_label[label + "_upper_right_corner"]['box_2d'][:2],
                            result_by_label[label + "_lower_right_corner"]['box_2d'][:2],
                            result_by_label[label + "_lower_left_corner"]['box_2d'][:2],
                        ]
                        rectangles.append(r)
                    except Exception as e:
                        print(f"Could not extract corners for {label}: {e}")
                        rectangles.append(self._rect_from_bbox(bboxes[label]))
                print("Parsed rectangles", rectangles)
                image_coords = [self._unnormalize_coords(r) for r in rectangles]
                print("As image coords", image_coords)
                return image_coords
                
            except json.JSONDecodeError as e:
                print(f"Failed to parse Gemini JSON response: {e}")
                print(f"Response was: {response.text}")
                return []
                
        except Exception as e:
            print(f"Gemini detection error: {e}")
            return []

    def detect_photos(self, image_width: int, image_height: int, image_path: str = None) -> List[List[QPointF]]:
        if not image_path or not self._model:
            return []
            
        try:
            # Load and encode image
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # Create PIL image for Gemini
            pil_image = Image.open(io.BytesIO(image_data))
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            # If the image is very large, no need to send the whole thing.
            # We're just getting approximate bounding boxes here.
            pil_image = pil_image.resize(size=(768, 768))
            
            prompt = """This is a scanned page from a photo album. Your task is
to detect the locations of the photos on the page. Output a JSON list of
bounding boxes, one for each photo, where each entry contains the 2D bounding
box in the format `{ "box_2d": [y_min, x_min, y_max, x_max] }`. Return only the
JSON response, no additional text."""

            # Generate response
            response = self._model.generate_content([pil_image, prompt])
            
            if not response.text:
                return []
            
            # Parse the JSON response
            try:
                result = self._parse_as_json(response)
                print("GOT RESPONSE")
                print(result)
                
                rectangles = []
                bboxes = {}
                for entry in result:
                    if 'box_2d' in entry:
                        rectangles.append(self._rect_from_bbox(entry['box_2d']))
                print("Parsed rectangles", rectangles)
                image_coords = [self._unnormalize_coords(r) for r in rectangles]
                print("As image coords", image_coords)
                return image_coords
                
            except json.JSONDecodeError as e:
                print(f"Failed to parse Gemini JSON response: {e}")
                print(f"Response was: {response.text}")
                return []
                
        except Exception as e:
            print(f"Gemini detection error: {e}")
            return []




# Registry of all available strategies
DETECTION_STRATEGIES = [
    OneBoxStrategy(),
    TwoBoxStrategy(),
    FourBoxStrategy(),
    SixBoxStrategy(),
    OpenCVContourStrategy(),
    PaddleOCRLayoutStrategy(),
    GeminiDetectionStrategy()
]