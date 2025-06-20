"""
Tests for bounding box storage and loading functionality.
"""

import os
import sys

import pytest

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.bounding_box_storage import BoundingBoxStorage


class TestBoundingBoxStorage:
    """Test the BoundingBoxStorage class for saving and loading bounding box data."""

    @pytest.fixture
    def test_directory(self):
        """Path to test data directory."""
        return os.path.join(os.path.dirname(__file__), "..", "test_data", "album1")

    @pytest.fixture
    def storage(self, test_directory):
        """Create a BoundingBoxStorage instance for testing."""
        return BoundingBoxStorage(test_directory)

    def test_init_loads_existing_data(self, storage, test_directory):
        """Test that initializing storage loads existing data file."""
        expected_data_file = os.path.join(test_directory, ".photo_extractor_data.json")

        assert storage.directory == test_directory
        assert storage.data_file == expected_data_file
        assert storage.data is not None
        assert isinstance(storage.data, dict)

    def test_load_bounding_boxes_for_existing_image(self, storage: BoundingBoxStorage):
        """Test loading bounding boxes for images that have saved data."""
        # Test loading for album_page1.jpg which has test data
        boxes = storage.load_bounding_boxes("album_page1.jpg")

        assert isinstance(boxes, list)
        assert len(boxes) == 3  # Based on our test data

        # Check first box structure
        first_box = boxes[0]
        assert first_box.box_id == "photo1-page1"
        assert len(first_box.corners) == 4  # Four corners for quad

    def test_load_bounding_boxes_for_nonexistent_image(
        self, storage: BoundingBoxStorage
    ):
        """Test loading bounding boxes for images with no saved data."""
        boxes = storage.load_bounding_boxes("nonexistent_image.jpg")

        assert isinstance(boxes, list)
        assert len(boxes) == 0

    def test_load_bounding_boxes_structure(self, storage: BoundingBoxStorage):
        """Test the structure of loaded bounding box data."""
        boxes = storage.load_bounding_boxes("album_page1.jpg")

        for box in boxes:
            # Check required fields

            # Check corners format
            corners = box.corners
            assert len(corners) == 4
            for corner in corners:
                assert len(corner) == 2  # x, y coordinates
                assert isinstance(corner[0], (int, float))
                assert isinstance(corner[1], (int, float))
