"""
Tests for image extraction, perspective correction, and EXIF preservation.
"""

import os
import shutil
import sys
import tempfile

import piexif
import pytest
from PIL import Image

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from image_processing import image_processor


class TestPerspectiveExtraction:
    """Test perspective correction and image extraction."""

    @pytest.fixture
    def test_image(self):
        """Create a test image for extraction testing."""
        # Create a 400x300 test image
        img = Image.new("RGB", (400, 300), color="white")
        # Add some visual markers to test extraction
        pixels = img.load()
        # Add colored corners for verification
        for i in range(50):
            for j in range(50):
                pixels[i, j] = (255, 0, 0)  # Red top-left
                pixels[350 + i, j] = (0, 255, 0)  # Green top-right
                pixels[i, 250 + j] = (0, 0, 255)  # Blue bottom-left
                pixels[350 + i, 250 + j] = (255, 255, 0)  # Yellow bottom-right
        return img

    def test_extract_rectangle(self, test_image):
        """Test extracting a rectangular region."""
        # Define a rectangular region (100x100 square in center)
        corners = [
            [150, 100],  # top-left
            [250, 100],  # top-right
            [250, 200],  # bottom-right
            [150, 200],  # bottom-left
        ]

        extracted = image_processor.extract_perspective_image(test_image, corners)

        assert extracted is not None
        assert isinstance(extracted, Image.Image)
        # Should maintain aspect ratio
        assert extracted.width == 100
        assert extracted.height == 100

    def test_extract_skewed_quadrilateral(self, test_image):
        """Test extracting a skewed quadrilateral and correcting perspective."""
        # Define a skewed quadrilateral
        corners = [
            [100, 80],  # top-left (slightly skewed)
            [280, 120],  # top-right
            [260, 220],  # bottom-right
            [120, 180],  # bottom-left
        ]

        extracted = image_processor.extract_perspective_image(test_image, corners)

        assert extracted is not None
        assert isinstance(extracted, Image.Image)
        assert extracted.width > 0
        assert extracted.height > 0

    def test_extract_with_specified_dimensions(self, test_image):
        """Test extraction with specified output dimensions."""
        corners = [[100, 100], [200, 100], [200, 200], [100, 200]]

        extracted = image_processor.extract_perspective_image(
            test_image, corners, output_width=150, output_height=150
        )

        assert extracted.width == 150
        assert extracted.height == 150

    def test_extract_edge_cases(self, test_image):
        """Test extraction with edge case coordinates."""
        # Test with coordinates at image boundaries
        corners = [[0, 0], [399, 0], [399, 299], [0, 299]]

        extracted = image_processor.extract_perspective_image(test_image, corners)

        assert extracted is not None
        # Should be close to original dimensions
        assert abs(extracted.width - 400) <= 2
        assert abs(extracted.height - 300) <= 2


class TestImageProcessor:
    """Test image extraction and saving."""

    @pytest.fixture
    def test_images_dir(self):
        """Path to test images directory."""
        return os.path.join(os.path.dirname(__file__), "..", "test_data", "album1")

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary directory for output testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_save_cropped_images_without_attributes(
        self, test_images_dir, temp_output_dir
    ):
        """Test saving cropped images without EXIF attributes."""
        # Load a test image
        test_image_path = os.path.join(test_images_dir, "album_page1.jpg")
        image = image_processor.load_image(test_image_path)

        # Define crop data (simple rectangle)
        crop_data = [[[100, 150], [400, 150], [400, 450], [100, 450]]]
        # Save cropped images
        saved_files = image_processor.save_cropped_images(
            image, crop_data, temp_output_dir, base_name="test_photo"
        )

        assert len(saved_files) == 1
        assert os.path.exists(saved_files[0])

        # Check the saved image
        saved_image = Image.open(saved_files[0])
        assert saved_image.width > 0
        assert saved_image.height > 0

    def test_save_cropped_images_with_attributes(
        self, test_images_dir, temp_output_dir
    ):
        """Test saving cropped images with EXIF attributes."""
        # Load test image
        test_image_path = os.path.join(test_images_dir, "album_page1.jpg")
        image = image_processor.load_image(test_image_path)

        # Define crop data with attributes
        h, w = 2400.0, 1800.0
        crop_data = [[[100, 150], [400, 150], [400, 450], [100, 450]]]
        attributes_list = [
            {"date_time": "1985-06-20", "comments": "Test extraction with EXIF data"}
        ]

        # Save cropped images with attributes
        saved_files = image_processor.save_cropped_images(
            image,
            crop_data,
            temp_output_dir,
            base_name="test_photo",
            attributes_list=attributes_list,
        )

        assert len(saved_files) == 1
        assert os.path.exists(saved_files[0])

        # Verify EXIF data was written
        try:
            exif_dict = piexif.load(saved_files[0])

            # Check software tag
            assert (
                exif_dict["0th"][piexif.ImageIFD.Software] == b"Photo Album Extractor"
            )

            # Check date fields
            expected_date = "1985:06:20 00:00:00"
            assert exif_dict["0th"][piexif.ImageIFD.DateTime] == expected_date.encode()
            assert (
                exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal]
                == expected_date.encode()
            )

            # Check comment
            assert (
                b"Test extraction with EXIF data"
                in exif_dict["Exif"][piexif.ExifIFD.UserComment]
            )

        except (KeyError, piexif.InvalidImageDataError):
            pytest.fail("EXIF data not properly written to extracted image")

    def test_save_multiple_cropped_images(self, test_images_dir, temp_output_dir):
        """Test saving multiple cropped images from one source."""
        # Load test image
        test_image_path = os.path.join(test_images_dir, "album_page1.jpg")
        image = image_processor.load_image(test_image_path)

        # Define multiple crop regions
        crop_data = [
            [[100, 150], [400, 150], [400, 450], [100, 450]],
            [[500, 150], [800, 150], [800, 450], [500, 450]],
            [[100, 500], [400, 500], [400, 800], [100, 800]],
        ]

        # Save all crops
        saved_files = image_processor.save_cropped_images(
            image, crop_data, temp_output_dir, base_name="multi_photo"
        )
        assert len(saved_files) == 3

        # Check all files exist and have different names
        for i, filepath in enumerate(saved_files):
            assert os.path.exists(filepath)
            assert f"multi_photo_{i:03d}" in os.path.basename(filepath)

            # Check image is valid
            img = Image.open(filepath)
            assert img.width > 0
            assert img.height > 0


class TestEXIFHandling:
    """Test EXIF data preservation and metadata handling."""

    @pytest.fixture
    def temp_output_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def test_image(self):
        """Create a simple test image."""
        return Image.new("RGB", (300, 300), color="white")

    def test_save_image_with_exif_date_conversion(self, temp_output_dir, test_image):
        """Test date format conversion in EXIF data."""
        test_filepath = os.path.join(temp_output_dir, "test_exif.jpg")

        attributes = {
            "date_time": "1985-12-25",  # ISO format
            "comments": "Christmas photo",
        }

        image_processor.save_image_with_exif(test_image, test_filepath, attributes)

        # Verify EXIF data
        exif_dict = piexif.load(test_filepath)

        # Check date conversion (ISO to EXIF format)
        expected_datetime = "1985:12:25 00:00:00"
        assert exif_dict["0th"][piexif.ImageIFD.DateTime] == expected_datetime.encode()
        assert (
            exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal]
            == expected_datetime.encode()
        )
        assert (
            exif_dict["Exif"][piexif.ExifIFD.DateTimeDigitized]
            == expected_datetime.encode()
        )

    def test_save_image_with_exif_comment_handling(self, temp_output_dir, test_image):
        """Test comment handling in EXIF data."""
        test_filepath = os.path.join(temp_output_dir, "test_comment.jpg")

        attributes = {
            "date_time": "2024-01-01",
            "comments": "Test comment with special chars: äöü & symbols!",
        }

        image_processor.save_image_with_exif(test_image, test_filepath, attributes)

        # Verify comment in EXIF
        exif_dict = piexif.load(test_filepath)

        # Check UserComment (should be UTF-8 encoded)
        user_comment = exif_dict["Exif"][piexif.ExifIFD.UserComment]
        # Skip the ASCII/Unicode header bytes and decode
        comment_text = user_comment.decode("utf-8")
        assert "Test comment with special chars: äöü & symbols!" in comment_text

        # Check ImageDescription
        description = exif_dict["0th"][piexif.ImageIFD.ImageDescription]
        assert b"Test comment with special chars" in description

    def test_save_image_with_exif_software_tag(self, temp_output_dir, test_image):
        """Test software tag in EXIF data."""
        test_filepath = os.path.join(temp_output_dir, "test_software.jpg")

        attributes = {"date_time": "2024-01-01", "comments": "Test"}

        image_processor.save_image_with_exif(test_image, test_filepath, attributes)

        # Verify software tag
        exif_dict = piexif.load(test_filepath)
        assert exif_dict["0th"][piexif.ImageIFD.Software] == b"Photo Album Extractor"

    def test_save_image_exif_error_handling(self, temp_output_dir, test_image):
        """Test error handling when EXIF writing fails."""
        test_filepath = os.path.join(temp_output_dir, "test_error.jpg")

        # Create malformed attributes to trigger errors
        attributes = {"date": "invalid-date-format", "comment": "Test"}

        # Should not raise exception, should fall back to saving without EXIF
        try:
            image_processor.save_image_with_exif(test_image, test_filepath, attributes)
            # File should still be created
            assert os.path.exists(test_filepath)

            # Image should be readable
            saved_image = Image.open(test_filepath)
            assert saved_image.width == 300
            assert saved_image.height == 300

        except Exception as e:
            pytest.fail(f"Should not raise exception, but got: {e}")

    def test_save_image_without_attributes(self, temp_output_dir, test_image):
        """Test saving image when no attributes are provided."""
        test_filepath = os.path.join(temp_output_dir, "test_no_attrs.jpg")

        # Save without attributes - should work fine
        image_processor.save_image_with_exif(test_image, test_filepath, {})

        assert os.path.exists(test_filepath)
        saved_image = Image.open(test_filepath)
        assert saved_image.width == 300
        assert saved_image.height == 300
