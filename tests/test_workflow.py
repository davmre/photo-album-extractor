"""
Tests for image loading functionality in the photo extractor app.
"""

import os
import shutil
import sys
import tempfile

import piexif
import pytest
from PIL import Image
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QMessageBox

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gui.extract_dialog import ExtractDialog
from gui.main_window import PhotoExtractorApp


class TestMainWindowWorkflow:
    """Test image loading through the main application window."""

    @pytest.fixture
    def app(self, qtbot, mocker):
        """Create application instance for testing."""
        test_app = PhotoExtractorApp()
        qtbot.addWidget(test_app)
        mocker.patch.object(QMessageBox, "warning", return_value=None)
        mocker.patch.object(QMessageBox, "information", return_value=None)
        return test_app

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

    def test_load_refine_and_extract(
        self, app, test_images_dir, temp_output_dir, qtbot
    ):
        """Test loading image from path through main window."""
        jpeg_path = os.path.join(test_images_dir, "album_page1.jpg")

        app.load_image_from_path(jpeg_path)

        # Wait for any async operations
        qtbot.wait(100)

        assert app.current_image_path == jpeg_path
        assert app.current_image is not None

        # Should load stored boxes.
        assert len(app.image_view.bounding_boxes) == 3
        attributes = [box.get_attributes() for box in app.image_view.bounding_boxes]
        read_date = attributes[0].date_hint == "1985-06-20"
        read_comments = (
            attributes[0].comments == "Birthday party - Sarah blowing out candles"
        )
        assert read_date
        assert read_comments

        # Refine the stored boxes. This shouldn't change them much.
        app.refine_all_boxes()
        assert len(app.image_view.bounding_boxes) == 3

        # Handle the extract dialog
        saved_files = self._extract_photos_via_dialog(app, temp_output_dir, qtbot)
        assert len(saved_files) == 3
        # TODO check the extracted photos have expected sizes

        # Verify extracted image
        extracted_img = Image.open(saved_files[0])
        assert extracted_img.width > 0
        assert extracted_img.height > 0

        # Check for EXIF data - descriptions now include dates
        expected_descriptions = [
            b"Family group photo after party\nDate: 1985-06-21",
            b"Birthday party - Sarah blowing out candles\nDate: 1985-06-20",
            b"Birthday cake - chocolate with strawberries\nDate: 1985-06-20",
        ]
        expected_datetimes = [
            "1985:06:21 00:00:00",
            "1985:06:20 00:00:00",
            "1985:06:20 00:01:00",
        ]
        exif_dicts = [piexif.load(filename) for filename in saved_files]
        for exif_dict in exif_dicts:
            description = exif_dict["0th"][piexif.ImageIFD.ImageDescription]
            assert description in expected_descriptions
            expected_datetime = expected_datetimes[
                expected_descriptions.index(description)
            ]
            assert (
                exif_dict["0th"][piexif.ImageIFD.DateTime].decode() == expected_datetime
            )
            assert (
                exif_dict["0th"][piexif.ImageIFD.Software] == b"Photo Album Extractor"
            )

    def _extract_photos_via_dialog(self, app, temp_output_dir, qtbot):
        """Helper method to handle the extract dialog and return saved files."""
        saved_files = []
        dialog_opened = False
        extraction_completed = False

        def handle_dialog():
            nonlocal dialog_opened, saved_files, extraction_completed
            dialog_opened = True

            # Find the dialog window
            for widget in app.findChildren(ExtractDialog):
                dialog = widget
                break
            else:
                raise RuntimeError("ExtractDialog not found")

            # Set output directory
            dialog.output_dir_edit.setText(temp_output_dir)

            # Select "Current page only" radio button
            dialog.current_page_radio.setChecked(True)

            # Wait for any UI updates
            qtbot.wait(50)

            # Hook into the processing finished signal before starting extraction
            def on_processing_finished(processed, skipped, errors):
                nonlocal saved_files, extraction_completed
                # Find saved files in output directory
                for root, _, files in os.walk(temp_output_dir):
                    for file in files:
                        if file.lower().endswith((".jpg", ".jpeg", ".png", ".tiff")):
                            saved_files.append(os.path.join(root, file))
                extraction_completed = True
                # Close dialog after processing
                QTimer.singleShot(100, dialog.accept)

            # Override the dialog's start_extraction to hook our callback
            original_start_extraction = dialog.start_extraction

            def patched_start_extraction():
                original_start_extraction()
                # Connect to the processor after it's created
                if dialog.processor:
                    dialog.processor.finished_processing.connect(on_processing_finished)

            dialog.start_extraction = patched_start_extraction
            dialog.start_extraction()

        # Use a timer to handle the dialog after it opens
        QTimer.singleShot(100, handle_dialog)

        # Call extract_photos which will open the dialog
        app.extract_photos()

        # Wait for dialog to be handled
        qtbot.waitUntil(lambda: dialog_opened, timeout=2000)

        # Wait for processing to complete
        qtbot.waitUntil(lambda: extraction_completed, timeout=10000)

        return saved_files
