"""
Tests for image extraction, perspective correction, and EXIF preservation.
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import piexif
import pytest
from PIL import Image

from core.bounding_box import BoundingBox, PhotoAttributes
from core.photo_types import PhotoOrientation

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import extract


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
        corners = np.array(
            [
                [150, 100],  # top-left
                [250, 100],  # top-right
                [250, 200],  # bottom-right
                [150, 200],  # bottom-left
            ],
            dtype=float,
        )

        extracted = extract.extract_perspective_image(test_image, corners)

        assert extracted is not None
        assert isinstance(extracted, Image.Image)
        # Should maintain aspect ratio
        assert extracted.width == 100
        assert extracted.height == 100

    def test_extract_skewed_quadrilateral(self, test_image):
        """Test extracting a skewed quadrilateral and correcting perspective."""
        # Define a skewed quadrilateral
        corners = np.array(
            [
                [100, 80],  # top-left (slightly skewed)
                [280, 120],  # top-right
                [260, 220],  # bottom-right
                [120, 180],  # bottom-left
            ],
            dtype=float,
        )

        extracted = extract.extract_perspective_image(test_image, corners)

        assert extracted is not None
        assert isinstance(extracted, Image.Image)
        assert extracted.width > 0
        assert extracted.height > 0

    def test_extract_with_specified_dimensions(self, test_image):
        """Test extraction with specified output dimensions."""
        corners = np.array(
            [[100, 100], [200, 100], [200, 200], [100, 200]], dtype=float
        )

        extracted = extract.extract_perspective_image(
            test_image, corners, output_width=150, output_height=150
        )

        assert extracted.width == 150
        assert extracted.height == 150

    def test_extract_edge_cases(self, test_image):
        """Test extraction with edge case coordinates."""
        # Test with coordinates at image boundaries
        corners = np.array([[0, 0], [399, 0], [399, 299], [0, 299]], dtype=float)

        extracted = extract.extract_perspective_image(test_image, corners)

        assert extracted is not None
        # Should be close to original dimensions
        assert abs(extracted.width - 400) <= 2
        assert abs(extracted.height - 300) <= 2

    def test_extract_with_90_degree_rotation_is_transpose(self, test_image):
        """Test that 90-degree rotation produces the transpose of the normal extraction."""
        # Create a non-square region to make rotation effects clearly visible
        corners = np.array(
            [[150, 100], [250, 100], [250, 150], [150, 150]],
            dtype=float,  # 100x50 rectangle
        )

        # Extract without rotation
        extracted_normal = extract.extract_perspective_image(test_image, corners)

        # Extract with 90-degree clockwise rotation
        extracted_rotated = extract.extract_perspective_image(
            test_image, corners, orientation=PhotoOrientation.ROTATED_90_CW
        )

        # After 90° CW rotation, dimensions should be swapped
        assert extracted_normal.width == extracted_rotated.height
        assert extracted_normal.height == extracted_rotated.width

        # Convert to numpy arrays for comparison
        pixels_normal = np.array(extracted_normal)
        pixels_rotated = np.array(extracted_rotated)

        expected_rotated = np.transpose(
            pixels_normal, (1, 0, 2)
        )  # transpose width/height

        # Due to interpolation differences, we check that most pixels are close
        # rather than requiring exact equality
        pixel_diff = np.abs(
            pixels_rotated.astype(float) - expected_rotated.astype(float)
        )
        mean_diff = np.mean(pixel_diff)

        # Allow for some interpolation error but should be very close
        assert mean_diff < 5.0, (
            f"Mean pixel difference {mean_diff} too large for proper rotation"
        )

    def test_extract_with_180_degree_rotation_is_double_flip(self, test_image):
        """Test that 180-degree rotation produces a double-flipped version."""
        corners = np.array(
            [[150, 100], [250, 100], [250, 200], [150, 200]], dtype=float
        )

        # Extract without rotation
        extracted_normal = extract.extract_perspective_image(test_image, corners)

        # Extract with 180-degree rotation
        extracted_rotated = extract.extract_perspective_image(
            test_image, corners, orientation=PhotoOrientation.UPSIDE_DOWN
        )

        # Dimensions should remain the same
        assert extracted_normal.size == extracted_rotated.size

        # Convert to numpy arrays
        pixels_normal = np.array(extracted_normal)
        pixels_rotated = np.array(extracted_rotated)

        # 180° rotation should be equivalent to flipping both horizontally and vertically
        expected_rotated = np.flipud(np.fliplr(pixels_normal))

        # Check similarity allowing for interpolation differences
        pixel_diff = np.abs(
            pixels_rotated.astype(float) - expected_rotated.astype(float)
        )
        mean_diff = np.mean(pixel_diff)

        assert mean_diff < 5.0, (
            f"Mean pixel difference {mean_diff} too large for 180° rotation"
        )


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
        image = extract.load_image(test_image_path)

        # Define crop data (simple rectangle)
        crop_data = [
            BoundingBox(
                corners=np.array(
                    [[100, 150], [400, 150], [400, 450], [100, 450]], dtype=float
                ),
                box_id="new",
                attributes=PhotoAttributes(),
            )
        ]
        # Save cropped images
        saved_files = []
        for status, v in extract.save_cropped_images(
            image, crop_data, temp_output_dir, base_name="test_photo"
        ):
            if status == extract.ExtractionStatus.SAVED:
                saved_files.append(v)

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
        image = extract.load_image(test_image_path)

        # Define crop data with attributes
        crop_data = [
            BoundingBox.new(
                corners=[(100, 150), (400, 150), (400, 450), (100, 450)],
                attributes=PhotoAttributes(
                    date_hint="1985-06-20", comments="Test extraction with EXIF data"
                ),
            )
        ]

        # Save cropped images with attributes
        saved_files = [
            str(v)
            for (s, v) in extract.save_cropped_images(
                image,
                crop_data,
                temp_output_dir,
                base_name="test_photo",
            )
            if s == extract.ExtractionStatus.SAVED
        ]

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
                in exif_dict["0th"][piexif.ImageIFD.ImageDescription]
            )

        except (KeyError, piexif.InvalidImageDataError):
            pytest.fail("EXIF data not properly written to extracted image")

    def test_save_multiple_cropped_images(self, test_images_dir, temp_output_dir):
        """Test saving multiple cropped images from one source."""
        # Load test image
        test_image_path = os.path.join(test_images_dir, "album_page1.jpg")
        image = extract.load_image(test_image_path)

        # Define multiple crop regions
        corners = [
            [[100, 150], [400, 150], [400, 450], [100, 450]],
            [[500, 150], [800, 150], [800, 450], [500, 450]],
            [[100, 500], [400, 500], [400, 800], [100, 800]],
        ]
        crop_data = [BoundingBox.new(corners=np.array(c, dtype=float)) for c in corners]

        # Save all crops
        saved_files = [
            str(v)
            for (s, v) in extract.save_cropped_images(
                image, crop_data, temp_output_dir, base_name="multi_photo"
            )
            if s == extract.ExtractionStatus.SAVED
        ]
        assert len(saved_files) == 3

        # Check all files exist and have different names
        print("saved files", saved_files)
        for i, filepath in enumerate(saved_files):
            assert os.path.exists(filepath)
            assert f"multi_photo_{i:03d}" in os.path.basename(filepath)

            # Check image is valid
            img = Image.open(filepath)
            assert img.width > 0
            assert img.height > 0

    def test_save_cropped_images_with_orientation(
        self, test_images_dir, temp_output_dir
    ):
        """Test saving cropped images with different orientations applied."""
        # Load test image
        test_image_path = os.path.join(test_images_dir, "album_page1.jpg")
        image = extract.load_image(test_image_path)

        # Create same bounding box with different orientations
        corners = np.array(
            [[100, 150], [400, 150], [400, 300], [100, 300]], dtype=float
        )

        crop_data = [
            BoundingBox.new(
                corners=corners,
                attributes=PhotoAttributes(
                    orientation=PhotoOrientation.NORMAL, comments="Normal orientation"
                ),
            ),
            BoundingBox.new(
                corners=corners,
                attributes=PhotoAttributes(
                    orientation=PhotoOrientation.ROTATED_90_CW,
                    comments="90° CW rotation",
                ),
            ),
        ]

        # Save cropped images
        saved_files = [
            str(v)
            for (s, v) in extract.save_cropped_images(
                image, crop_data, temp_output_dir, base_name="orientation_test"
            )
            if s == extract.ExtractionStatus.SAVED
        ]

        assert len(saved_files) == 2

        # Load both images and verify they have different dimensions
        normal_img = Image.open(saved_files[0])
        rotated_img = Image.open(saved_files[1])

        # For 90° rotation, width and height should be swapped
        assert normal_img.width == rotated_img.height
        assert normal_img.height == rotated_img.width

        # Images should be different due to rotation
        normal_pixels = np.array(normal_img)
        rotated_pixels = np.array(rotated_img)
        assert not np.array_equal(normal_pixels, rotated_pixels)


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

        attributes = PhotoAttributes(
            date_hint="1985-12-25",  # ISO format
            comments="Christmas photo",
        )

        extract.save_image_with_exif(test_image, test_filepath, attributes)

        # Verify EXIF data
        exif_dict = piexif.load(test_filepath)

        # Check date conversion (ISO to EXIF format)
        expected_datetime = "1985:12:25 00:00:00"
        assert exif_dict["0th"][piexif.ImageIFD.DateTime] == expected_datetime.encode()
        assert (
            exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal]
            == expected_datetime.encode()
        )

    def test_save_image_with_exif_comment_handling(self, temp_output_dir, test_image):
        """Test comment handling in EXIF data."""
        test_filepath = os.path.join(temp_output_dir, "test_comment.jpg")

        attributes = PhotoAttributes(
            date_hint="2024-01-01",
            comments="Test comment with special chars: äöü & symbols!",
        )

        extract.save_image_with_exif(test_image, test_filepath, attributes)

        # Verify comment in EXIF
        exif_dict = piexif.load(test_filepath)

        # Check ImageDescription
        description = exif_dict["0th"][piexif.ImageIFD.ImageDescription]
        assert b"Test comment with special chars" in description

    def test_save_image_with_exif_software_tag(self, temp_output_dir, test_image):
        """Test software tag in EXIF data."""
        test_filepath = os.path.join(temp_output_dir, "test_software.jpg")

        attributes = PhotoAttributes(date_hint="2024-01-01", comments="Test")

        extract.save_image_with_exif(test_image, test_filepath, attributes)

        # Verify software tag
        exif_dict = piexif.load(test_filepath)
        assert exif_dict["0th"][piexif.ImageIFD.Software] == b"Photo Album Extractor"
