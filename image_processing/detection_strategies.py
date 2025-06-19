"""
Photo detection strategies for automatically identifying photos in scanned album pages.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Union

import google.generativeai as genai  # type: ignore
import numpy as np
import PIL.Image

from photo_types import QuadArray


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
    def detect_photos(self, image: PIL.Image.Image) -> list[QuadArray]:
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


class GeminiDetectionStrategy(DetectionStrategy):
    """Gemini AI-based strategy for detecting photos in album pages."""

    def __init__(self) -> None:
        self._model: Any | None = None
        self._api_key: str | None = None

    def set_api_key(self, api_key: str) -> None:
        """Set the API key and reinitialize the model."""
        self._api_key = api_key
        self._setup_gemini()

    def _setup_gemini(self) -> None:
        """Initialize the Gemini model."""
        if not self._api_key:
            print("Gemini API key not set. Please configure it in Settings.")
            self._model = None
            return

        try:
            # Configure the API key
            genai.configure(api_key=self._api_key)  # type: ignore
            self._model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")  # type: ignore
        except Exception as e:
            print(f"Failed to initialize Gemini: {e}")
            self._model = None

    @property
    def name(self) -> str:
        return "Gemini AI Detection"

    @property
    def description(self) -> str:
        return "Use Gemini AI to intelligently detect photos in album pages"

    def _parse_as_json(self, response: Any) -> list[dict[str, Any]]:
        # Extract JSON from response (remove any markdown formatting)
        json_text = response.text.strip()
        if json_text.startswith("```json"):
            json_text = json_text[7:]
        if json_text.endswith("```"):
            json_text = json_text[:-3]
        json_text = json_text.strip()
        return json.loads(json_text)

    def _corner_points_from_bbox(self, box_2d: list[float]) -> QuadArray:
        y_min, x_min, y_max, x_max = box_2d
        return np.array(
            [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        )

    def detect_photos(self, image: PIL.Image.Image) -> list[QuadArray]:
        if not image or not self._model:
            return []

        image_width, image_height = image.width, image.height

        try:
            # If the image is very large, no need to send the whole thing.
            # We're just getting approximate bounding boxes here.
            image = image.resize(size=(768, 768))

            if image.mode != "RGB":
                image = image.convert("RGB")

            prompt = """This is a scanned page from a photo album. Your task is
to detect the locations of the photos on the page. Output a JSON list of
bounding boxes, one for each photo, where each entry contains the 2D bounding
box in the format `{ "box_2d": [y_min, x_min, y_max, x_max] }`. Return only the
JSON response, no additional text."""

            response = self._model.generate_content([image, prompt])

            if not response.text:
                return []

            # Parse the JSON response
            try:
                result = self._parse_as_json(response)
                print("GOT Gemini response")
                print(result)

                rectangles = []
                for entry in result:
                    if "box_2d" in entry:
                        rectangles.append(
                            self._corner_points_from_bbox(entry["box_2d"])
                        )
                print("Parsed rectangles", rectangles)

                unnormalize_coords = np.array(
                    [image_width / 1000.0, image_height / 1000.0]
                )

                image_coords = [unnormalize_coords * r for r in rectangles]
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
DETECTION_STRATEGIES = [GeminiDetectionStrategy()]
