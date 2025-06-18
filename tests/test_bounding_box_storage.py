"""
Tests for bounding box storage and loading functionality.
"""

import json
import os
import sys

import pytest

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from storage.bounding_box_storage import BoundingBoxStorage


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

    def test_load_bounding_boxes_for_existing_image(self, storage):
        """Test loading bounding boxes for images that have saved data."""
        # Test loading for album_page1.jpg which has test data
        boxes = storage.load_bounding_boxes("album_page1.jpg")

        assert isinstance(boxes, list)
        assert len(boxes) == 3  # Based on our test data

        # Check first box structure
        first_box = boxes[0]
        assert first_box["type"] == "quad"
        assert first_box["id"] == "photo1-page1"
        assert "corners" in first_box
        assert "attributes" in first_box
        assert len(first_box["corners"]) == 4  # Four corners for quad

    def test_load_bounding_boxes_for_nonexistent_image(self, storage):
        """Test loading bounding boxes for images with no saved data."""
        boxes = storage.load_bounding_boxes("nonexistent_image.jpg")

        assert isinstance(boxes, list)
        assert len(boxes) == 0

    def test_load_bounding_boxes_structure(self, storage):
        """Test the structure of loaded bounding box data."""
        boxes = storage.load_bounding_boxes("album_page1.jpg")

        for box in boxes:
            # Check required fields
            assert "type" in box
            assert "corners" in box
            assert box["type"] == "quad"

            # Check corners format
            corners = box["corners"]
            assert len(corners) == 4
            for corner in corners:
                assert len(corner) == 2  # x, y coordinates
                assert isinstance(corner[0], (int, float))
                assert isinstance(corner[1], (int, float))

            # Check optional fields if present
            if "id" in box:
                assert isinstance(box["id"], str)
            if "attributes" in box:
                assert isinstance(box["attributes"], dict)

    def test_get_box_attributes(self, storage):
        """Test retrieving attributes for specific boxes."""
        # Test getting attributes for a box that exists
        attributes = storage.get_box_attributes("album_page1.jpg", "photo1-page1")

        assert isinstance(attributes, dict)
        assert "date_time" in attributes
        assert "comments" in attributes
        assert attributes["date_time"] == "1985-06-20"
        assert attributes["comments"] == "Birthday party - Sarah blowing out candles"

    def test_get_box_attributes_nonexistent(self, storage):
        """Test retrieving attributes for non-existent boxes."""
        # Non-existent image
        attributes = storage.get_box_attributes("fake.jpg", "fake-id")
        assert attributes == {}

        # Existing image, non-existent box ID
        attributes = storage.get_box_attributes("album_page1.jpg", "fake-id")
        assert attributes == {}

    def test_generate_box_id(self, storage):
        """Test box ID generation."""
        id1 = storage.generate_box_id()
        id2 = storage.generate_box_id()

        assert isinstance(id1, str)
        assert isinstance(id2, str)
        assert id1 != id2  # Should be unique
        assert len(id1) > 0
        assert len(id2) > 0


class TestBoundingBoxStorageReadWrite:
    """Test reading and writing bounding box data."""

    @pytest.fixture
    def temp_directory(self, tmp_path):
        """Create a temporary directory for testing."""
        return str(tmp_path)

    @pytest.fixture
    def temp_storage(self, temp_directory):
        """Create storage in temporary directory."""
        return BoundingBoxStorage(temp_directory)

    def test_save_and_load_bounding_boxes(self, temp_storage):
        """Test saving and loading bounding boxes."""
        # Create some test bounding boxes
        test_boxes = [
            {
                "type": "quad",
                "id": "test-box-1",
                "corners": [[10, 10], [100, 10], [100, 100], [10, 100]],
                "attributes": {"date_time": "2024-01-01", "comments": "Test box 1"},
            },
            {
                "type": "quad",
                "id": "test-box-2",
                "corners": [[200, 200], [300, 200], [300, 300], [200, 300]],
                "attributes": {"date_time": "2024-01-02", "comments": "Test box 2"},
            },
        ]

        # Save the boxes
        temp_storage.data["test_image.jpg"] = test_boxes
        temp_storage.save_data()

        # Create new storage instance to test loading
        new_storage = BoundingBoxStorage(temp_storage.directory)
        loaded_boxes = new_storage.load_bounding_boxes("test_image.jpg")

        assert len(loaded_boxes) == 2
        assert loaded_boxes == test_boxes

    def test_update_box_attributes(self, temp_storage):
        """Test updating attributes for existing boxes."""
        # Set up initial data
        test_boxes = [
            {
                "type": "quad",
                "id": "test-box-1",
                "corners": [[10, 10], [100, 10], [100, 100], [10, 100]],
                "attributes": {
                    "date_time": "2024-01-01",
                    "comments": "Original comment",
                },
            }
        ]
        temp_storage.data["test_image.jpg"] = test_boxes

        # Update attributes
        new_attributes = {
            "date_time": "2024-01-15",
            "comments": "Updated comment",
            "location": "New field",
        }
        result = temp_storage.update_box_attributes(
            "test_image.jpg", "test-box-1", new_attributes
        )

        assert result is True

        # Verify update
        updated_attributes = temp_storage.get_box_attributes(
            "test_image.jpg", "test-box-1"
        )
        assert updated_attributes == new_attributes

    def test_update_box_attributes_nonexistent(self, temp_storage):
        """Test updating attributes for non-existent boxes."""
        # Try to update non-existent image
        result = temp_storage.update_box_attributes("fake.jpg", "fake-id", {})
        assert result is False

        # Try to update non-existent box in existing image
        temp_storage.data["test_image.jpg"] = []
        result = temp_storage.update_box_attributes("test_image.jpg", "fake-id", {})
        assert result is False


class TestBoundingBoxAttributes:
    """Test handling of bounding box attributes (dates, comments, etc.)."""

    @pytest.fixture
    def test_directory(self):
        return os.path.join(os.path.dirname(__file__), "..", "test_data", "album1")

    @pytest.fixture
    def storage(self, test_directory):
        return BoundingBoxStorage(test_directory)

    def test_attributes_date_format(self, storage):
        """Test that date attributes are in expected format."""
        test_images = ["album_page1.jpg", "album_page2.png", "album_page3.tiff"]

        for image in test_images:
            boxes = storage.load_bounding_boxes(image)
            for box in boxes:
                if "attributes" in box and "date_time" in box["attributes"]:
                    date_str = box["attributes"]["date_time"]
                    # Check date format (YYYY-MM-DD)
                    assert len(date_str) == 10
                    assert date_str.count("-") == 2
                    parts = date_str.split("-")
                    assert len(parts) == 3
                    assert len(parts[0]) == 4  # Year
                    assert len(parts[1]) == 2  # Month
                    assert len(parts[2]) == 2  # Day

    def test_attributes_comment_content(self, storage):
        """Test that comment attributes contain expected content."""
        boxes = storage.load_bounding_boxes("album_page1.jpg")

        comments = [box["attributes"]["comments"] for box in boxes]

        assert len(comments) > 0
        for comment in comments:
            assert isinstance(comment, str)
            assert len(comment) > 0
            # Comments should be descriptive
            assert len(comment.split()) >= 2  # At least 2 words

    def test_all_boxes_have_attributes(self, storage):
        """Test that all saved bounding boxes have attributes."""
        test_images = ["album_page1.jpg", "album_page2.png", "album_page3.tiff"]

        for image in test_images:
            boxes = storage.load_bounding_boxes(image)
            for box in boxes:
                assert "attributes" in box, f"Box in {image} missing attributes: {box}"
                attributes = box["attributes"]
                assert "date_time" in attributes, (
                    f"Box in {image} missing date attribute"
                )
                assert "comments" in attributes, (
                    f"Box in {image} missing comment attribute"
                )

    def test_specific_attribute_values(self, storage):
        """Test specific attribute values from our test data."""
        # Test specific known values
        attrs1 = storage.get_box_attributes("album_page1.jpg", "photo1-page1")
        assert attrs1["date_time"] == "1985-06-20"
        assert "Birthday party" in attrs1["comments"]

        attrs2 = storage.get_box_attributes("album_page2.png", "photo1-page2")
        assert attrs2["date_time"] == "1985-07-04"
        assert "fireworks" in attrs2["comments"]

        attrs3 = storage.get_box_attributes("album_page3.tiff", "photo1-page3")
        assert attrs3["date_time"] == "1985-08-15"
        assert "school" in attrs3["comments"].lower()


class TestBoundingBoxIntegration:
    """Test integration between storage and the main application."""

    @pytest.fixture
    def test_directory(self):
        return os.path.join(os.path.dirname(__file__), "..", "test_data", "album1")

    def test_data_file_exists(self, test_directory):
        """Test that the data file exists in test directory."""
        data_file = os.path.join(test_directory, ".photo_extractor_data.json")
        assert os.path.exists(data_file)

    def test_data_file_valid_json(self, test_directory):
        """Test that the data file contains valid JSON."""
        data_file = os.path.join(test_directory, ".photo_extractor_data.json")

        with open(data_file) as f:
            data = json.load(f)

        assert isinstance(data, dict)

        # Check structure for each image
        for image_name, boxes in data.items():
            assert isinstance(image_name, str)
            assert image_name.endswith((".jpg", ".png", ".tiff"))
            assert isinstance(boxes, list)

            for box in boxes:
                assert isinstance(box, dict)
                assert "type" in box
                assert "corners" in box

    def test_box_coordinates_within_image_bounds(self, test_directory):
        """Test that bounding box coordinates are within expected image bounds."""
        storage = BoundingBoxStorage(test_directory)

        # Expected image dimensions from our test data
        image_dimensions = {
            "album_page1.jpg": (2400, 1800),
            "album_page2.png": (2048, 1536),
            "album_page3.tiff": (1920, 1440),
        }

        for image_name, (width, height) in image_dimensions.items():
            boxes = storage.load_bounding_boxes(image_name)

            for box in boxes:
                corners = box["corners"]
                for corner in corners:
                    x, y = corner
                    assert 0 <= x <= width, (
                        f"X coordinate {x} out of bounds for {image_name}"
                    )
                    assert 0 <= y <= height, (
                        f"Y coordinate {y} out of bounds for {image_name}"
                    )
