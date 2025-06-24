"""
Photo detection strategies for automatically identifying photos in scanned album pages.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

import google.generativeai as genai  # type: ignore
import numpy as np
import PIL.Image

from core.errors import AppError
from core.photo_types import BoundingBoxData, PhotoAttributes, QuadArray
from core.settings import AppSettings


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
    def detect_photos(self, image: PIL.Image.Image) -> list[BoundingBoxData]:
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

    def _get_bounding_box_data_from_json(
        self, entry, rescale_coordinates: np.ndarray | None = None
    ) -> BoundingBoxData:
        if "box_2d" in entry:
            corners = self._corner_points_from_bbox(entry["box_2d"])
            if rescale_coordinates is not None:
                corners = rescale_coordinates * corners
        else:
            raise ValueError("Entry does not contain a bounding box.")

        attributes = PhotoAttributes()
        if "date" in entry:
            attributes.date_string = entry["date"]
        if "caption" in entry:
            attributes.comments = entry["caption"]
        return BoundingBoxData.new(corners=corners, attributes=attributes)

    def _corner_points_from_bbox(self, box_2d: list[float]) -> QuadArray:
        y_min, x_min, y_max, x_max = box_2d
        return np.array(
            [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        )

    def detect_photos(self, image: PIL.Image.Image) -> list[BoundingBoxData]:
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
detected photos. Each entry contains:

 - **Required:** the 2D bounding box of the photo `"box_2d": [y_min, x_min, y_max, x_max]`.
   This bounding box contains the photo only; no annotations or associated text.

 - **Optional:**: if there is additional writing on the page that appears to be
   associated with this photo, you may include additional string fields `date` (if
   there appears to be a date written for this photo) and/or `caption` (for any non-date
   text related to this photo). If there is no date or caption written, simply omit
   these fields.

This project has sentimental value and your help is appreciated!

Example response for a page with two photos:

{
    [
      "box_2d": [photo1_y_min, photo1_x_min, photo1_y_max, photo1_x_max],
      "date": "May 1997",
    ],
    [
      "box_2d": [photo2_y_min, photo2_x_min, photo2_y_max, photo2_x_max],
      "caption": "Dinner with Susan",
    ]
}

Return only the JSON response, no additional text."""

            response = self._model.generate_content([image, prompt])

            if not response.text:
                return []

            unnormalize_gemini_coords = np.array(
                [image_width / 1000.0, image_height / 1000.0]
            )

            # Parse the JSON response
            try:
                result = self._parse_as_json(response)
                print("GOT Gemini response")
                print(result)

                detected_bboxes = []
                for entry in result:
                    try:
                        bbox_data = self._get_bounding_box_data_from_json(
                            entry, rescale_coordinates=unnormalize_gemini_coords
                        )
                        detected_bboxes.append(bbox_data)
                    except ValueError as e:
                        print("NO BBOX??", e)
                        continue
                return detected_bboxes

            except json.JSONDecodeError as e:
                print(f"Failed to parse Gemini JSON response: {e}")
                print(f"Response was: {response.text}")
                return []

        except Exception as e:
            print(f"Gemini detection error: {e}")
            return []


class GeminiWithRefinementDetectionStrategy(DetectionStrategy):
    def __init__(self):
        self.gemini_strategy = GeminiDetectionStrategy()

    @property
    def name(self) -> str:
        return "Auto-refine Gemini"

    @property
    def description(self) -> str:
        return "Gemini AI Detection with auto-refined corners."

    def set_api_key(self, api_key: str):
        self.gemini_strategy.set_api_key(api_key)

    def set_refinement_strategy(self, refine_fn):
        self.refine_fn = refine_fn

    def detect_photos(self, image: PIL.Image.Image) -> list[BoundingBoxData]:
        detected_boxes = self.gemini_strategy.detect_photos(image)
        for box in detected_boxes:
            refined_corners = self.refine_fn(image, box.corners)
            box.corners = refined_corners
        return detected_boxes


# Registry of all available strategies
_DETECTION_STRATEGIES: list[DetectionStrategy] = [
    GeminiDetectionStrategy(),
    GeminiWithRefinementDetectionStrategy(),
]

DETECTION_STRATEGIES: dict[str, DetectionStrategy] = {
    s.name: s for s in _DETECTION_STRATEGIES
}


def configure_detection_strategy(settings: AppSettings) -> DetectionStrategy:
    # Get the selected strategy from settings
    strategy_name = settings.detection_strategy
    selected_strategy = DETECTION_STRATEGIES.get(
        strategy_name, _DETECTION_STRATEGIES[0]
    )

    # Configure API key for strategies that need it
    if hasattr(selected_strategy, "set_api_key"):
        if settings.gemini_api_key:
            selected_strategy.set_api_key(settings.gemini_api_key)  # type: ignore
        else:
            raise AppError(
                msg=f"The {selected_strategy.name} strategy requires an API key. "
                "Please configure it in Edit > Settings.",
                title="API Key Required",
            )
    return selected_strategy
