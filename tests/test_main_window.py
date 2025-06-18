"""
Tests for image loading functionality in the photo extractor app.
"""

import os
import sys

import pytest

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gui.main_window import PhotoExtractorApp


class TestMainWindow:
    """Test image loading through the main application window."""

    @pytest.fixture
    def app(self, qtbot):
        """Create application instance for testing."""
        test_app = PhotoExtractorApp()
        qtbot.addWidget(test_app)
        return test_app

    @pytest.fixture
    def test_images_dir(self):
        """Path to test images directory."""
        return os.path.join(os.path.dirname(__file__), "..", "test_data", "album1")

    def test_load_image_from_path(self, app, test_images_dir, qtbot):
        """Test loading image from path through main window."""
        jpeg_path = os.path.join(test_images_dir, "album_page1.jpg")

        app.load_image_from_path(jpeg_path)

        # Wait for any async operations
        qtbot.wait(100)

        assert app.current_image_path == jpeg_path
        assert app.current_image is not None
        assert app.image_view.scene.items()  # Should have image item in scene

    def test_load_directory(self, app, test_images_dir, qtbot):
        """Test loading a directory of images."""
        app.load_directory(test_images_dir)

        # Wait for directory loading
        qtbot.wait(100)

        assert app.current_directory == test_images_dir
        assert app.bounding_box_storage is not None
        assert app.bounding_box_storage.directory == test_images_dir

        # Should load first image in directory
        expected_files = [
            "album_page1.jpg",
            "album_page2.png",
            "album_page3.tiff",
            "album_page4.jpg",
        ]
        first_image = os.path.join(test_images_dir, expected_files[0])
        assert app.current_image_path == first_image

    def test_set_current_directory(self, app, test_images_dir, qtbot):
        """Test setting current directory updates storage and sidebar."""
        app.set_current_directory(test_images_dir)

        qtbot.wait(100)

        assert app.current_directory == test_images_dir
        assert app.bounding_box_storage.directory == test_images_dir

        # Directory sidebar should be updated
        image_list = app.directory_list
        assert image_list.current_directory == test_images_dir
