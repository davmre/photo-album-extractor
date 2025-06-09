"""
Tests for image loading functionality in the photo extractor app.
"""

import os
import sys
import piexif
import pytest
import tempfile
import shutil
from PIL import Image
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtGui import QPixmap

import pytest_mock as mock


import logging

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from image_processing import image_processor
from gui.main_window import PhotoExtractorApp

class TestMainWindowWorkflow:
    """Test image loading through the main application window."""
    
    @pytest.fixture
    def app(self, qtbot, mocker):
        """Create application instance for testing."""
        test_app = PhotoExtractorApp()
        qtbot.addWidget(test_app)
        mocker.patch.object(QMessageBox, 'warning', return_value=None)
        mocker.patch.object(QMessageBox, 'information', return_value=None)
        return test_app

    @pytest.fixture
    def test_images_dir(self):
        """Path to test images directory."""
        return os.path.join(os.path.dirname(__file__), '..', 'test_data', 'album1')
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary directory for output testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_load_refine_and_extract(self, app, test_images_dir, temp_output_dir, qtbot):
        """Test loading image from path through main window."""
        jpeg_path = os.path.join(test_images_dir, 'album_page1.jpg')
        
        app.load_image_from_path(jpeg_path)
        
        # Wait for any async operations
        qtbot.wait(100)
        
        assert app.current_image_path == jpeg_path
        assert app.current_image is not None
        assert app.image_view.scene.items()  # Should have image item in scene
        
        # Should load stored boxes.
        assert len(app.image_view.bounding_boxes) == 3
        attributes = [box.get_attributes() for box in app.image_view.bounding_boxes]
        read_date = (attributes[0]["date_time"]) == "1985-06-20"
        read_comments = (attributes[0]["comments"]) == "Birthday party - Sarah blowing out candles"
        assert read_date
        assert read_comments

        # Refine the stored boxes. This shouldn't change them much.
        app.refine_all_boxes()
        assert len(app.image_view.bounding_boxes) == 3
        
        saved_files = app.extract_photos(output_directory=temp_output_dir)
        assert len(saved_files) == 3
        # TODO check the extracted photos have expected sizes

        # Verify extracted image
        extracted_img = Image.open(saved_files[0])
        assert extracted_img.width > 0
        assert extracted_img.height > 0

        # Check for EXIF data
        expected_descriptions = [b'Family group photo after party', 
                                 b'Birthday party - Sarah blowing out candles',
                                 b'Birthday cake - chocolate with strawberries']
        expected_datetimes =  ['1985:06:21 00:00:00', '1985:06:20 00:00:00', '1985:06:20 00:00:00']
        exif_dicts = [piexif.load(filename) for filename in saved_files]
        for exif_dict in exif_dicts:
            description = exif_dict['0th'][piexif.ImageIFD.ImageDescription]
            assert description in expected_descriptions
            expected_datetime = expected_datetimes[expected_descriptions.index(description)]
            assert exif_dict['0th'][piexif.ImageIFD.DateTime] == expected_datetime.encode()
            assert exif_dict['0th'][piexif.ImageIFD.Software] == b'Photo Album Extractor'


