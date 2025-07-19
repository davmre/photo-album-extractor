"""
Batch preprocessing dialog for running detection and refinement on multiple images.
"""

from __future__ import annotations

import glob
import os

import PIL.Image
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
)

from core import geometry
from core.bounding_box_storage import BoundingBoxStorage
from core.detection_strategies import DETECTION_STRATEGIES, configure_detection_strategy
from core.refinement_strategies import (
    REFINEMENT_STRATEGIES,
    configure_refinement_strategy,
)
from core.settings import app_settings


def get_image_files_in_directory(directory: str) -> list[str]:
    """Get list of all image files in the directory."""
    supported_formats = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif"}
    image_files = []

    for ext in supported_formats:
        pattern = os.path.join(directory, f"*{ext}")
        image_files.extend(glob.glob(pattern, recursive=False))
        pattern = os.path.join(directory, f"*{ext.upper()}")
        image_files.extend(glob.glob(pattern, recursive=False))

    return sorted(image_files)


class BatchProcessor(QThread):
    """Background thread for batch processing operations."""

    progress_updated = pyqtSignal(int, int, str)  # current, total, status_message
    finished_processing = pyqtSignal(int, int, int)  # processed, skipped, errors
    error_occurred = pyqtSignal(str)  # error message
    log_message = pyqtSignal(str)  # detailed log message

    def __init__(
        self,
        image_files: list[str],
        storage: BoundingBoxStorage,
        detection_strategy_name: str | None,
        skip_existing: bool,
        refinement_strategy_name: str | None,
        refinement_tolerance: float,
        shrink_pixels: int,
    ):
        super().__init__()
        self.image_files = image_files
        self.storage = storage
        self.detection_strategy_name = detection_strategy_name
        self.skip_existing = skip_existing
        self.refinement_strategy_name = refinement_strategy_name
        self.refinement_tolerance = refinement_tolerance
        self.shrink_pixels = shrink_pixels
        self.cancelled = False

        # Initialize strategies
        self.detection_strategy = None
        self.refinement_strategy = None

        if detection_strategy_name:
            try:
                # Temporarily set strategy in settings for configuration
                original_strategy = app_settings.detection_strategy
                app_settings.detection_strategy = detection_strategy_name
                self.detection_strategy = configure_detection_strategy(app_settings)
                app_settings.detection_strategy = original_strategy
            except Exception as e:
                self.error_occurred.emit(f"Error configuring detection strategy: {e}")
                return

        if refinement_strategy_name:
            try:
                # Temporarily set strategy in settings for configuration
                original_strategy = app_settings.refinement_strategy
                app_settings.refinement_strategy = refinement_strategy_name
                self.refinement_strategy = configure_refinement_strategy(app_settings)
                app_settings.refinement_strategy = original_strategy
            except Exception as e:
                self.error_occurred.emit(f"Error configuring refinement strategy: {e}")
                return

    def cancel(self):
        """Cancel the batch processing operation."""
        self.cancelled = True

    def run(self):
        """Run the batch processing operation."""
        if not self.image_files:
            self.log_message.emit("No image files found to process.")
            self.finished_processing.emit(0, 0, 0)
            return

        self.log_message.emit(
            f"Starting batch processing of {len(self.image_files)} images..."
        )

        processed_count = 0
        skipped_count = 0
        error_count = 0

        for i, image_path in enumerate(self.image_files):
            if self.cancelled:
                self.log_message.emit("Processing cancelled by user.")
                break

            filename = os.path.basename(image_path)

            try:
                # Update progress
                self.progress_updated.emit(
                    i + 1, len(self.image_files), f"Processing {filename}"
                )

                self.log_message.emit(
                    f"--- Processing {filename} ({i + 1}/{len(self.image_files)}) ---"
                )

                # Process the image
                result = self._process_image(image_path)

                if result == "processed":
                    processed_count += 1
                elif result == "skipped":
                    skipped_count += 1

            except Exception as e:
                error_count += 1
                self.log_message.emit(f"ERROR processing {filename}: {e}")

        self.log_message.emit(
            f"Batch processing completed: {processed_count} processed, {skipped_count} skipped, {error_count} errors"
        )
        self.finished_processing.emit(processed_count, skipped_count, error_count)

    def _process_image(self, image_path: str) -> str:
        """Process a single image with detection and/or refinement."""
        filename = os.path.basename(image_path)

        # Check if bounding boxes already exist
        existing_boxes = self.storage.get_bounding_boxes(filename)

        if existing_boxes:
            self.log_message.emit(
                f"  Found {len(existing_boxes)} existing bounding boxes"
            )
        else:
            self.log_message.emit("  No existing bounding boxes found")

        if existing_boxes and self.skip_existing:
            self.log_message.emit("  Skipping - has existing boxes.")
            return "skipped"

        # Load image
        with PIL.Image.open(image_path) as image:
            self.log_message.emit(f"  Loaded image ({image.size[0]}x{image.size[1]})")
            current_boxes = existing_boxes.copy()

            # Run detection if requested
            if self.detection_strategy:
                # Run detection (either no existing boxes or we're replacing)
                self.log_message.emit(
                    f"  Running detection using {self.detection_strategy.name}"
                )
                detected_boxes = self.detection_strategy.detect_photos(image)
                current_boxes = detected_boxes
                self.storage.set_bounding_boxes(filename, current_boxes)
                self.log_message.emit(f"  Detection found {len(detected_boxes)} photos")

            # Run refinement if requested
            if self.refinement_strategy and current_boxes:
                self.log_message.emit(
                    f"  Running refinement using {self.refinement_strategy.name} (tolerance: {self.refinement_tolerance})"
                )
                refined_boxes = []
                for i, box in enumerate(current_boxes):
                    refined_corners = self.refinement_strategy.refine(
                        image,
                        corner_points=box.corners,
                        reltol=self.refinement_tolerance,
                    )
                    box.corners = refined_corners
                    refined_boxes.append(box)
                    self.log_message.emit(f"    Box {i + 1}: refined")

                current_boxes = refined_boxes
                self.storage.set_bounding_boxes(filename, refined_boxes)
                self.log_message.emit(
                    f"  Refinement completed for {len(refined_boxes)} boxes"
                )
            elif self.refinement_strategy and not current_boxes:
                self.log_message.emit("  Skipping refinement - no boxes to refine")

            # Apply shrinking if configured
            if self.shrink_pixels != 0 and current_boxes:
                shrunk_boxes = []
                for box in current_boxes:
                    shrunk_corners = geometry.shrink_rectangle(
                        box.corners, shrink_by=self.shrink_pixels
                    )
                    box.corners = shrunk_corners
                    shrunk_boxes.append(box)
                self.storage.set_bounding_boxes(filename, shrunk_boxes)
                self.log_message.emit(
                    f"    Shrunk {len(shrunk_boxes)} boxes by {self.shrink_pixels}px"
                )

        self.log_message.emit(f"  ✓ Completed processing {filename}")
        return "processed"


class BatchPreprocessDialog(QDialog):
    """Dialog for batch preprocessing operations."""

    def __init__(
        self,
        parent=None,
        directory: str = "",
        storage: BoundingBoxStorage | None = None,
    ):
        super().__init__(parent)
        self.directory = directory
        self.storage = storage
        self.processor: BatchProcessor | None = None
        self.selected_files: list[str] = []

        self.init_ui()

    def init_ui(self):
        """Initialize the dialog UI."""
        self.setWindowTitle("Batch Preprocess")
        self.setMinimumSize(600, 800)
        self.resize(600, 800)

        layout = QVBoxLayout(self)

        # File Selection group
        file_selection_group = QGroupBox("File Selection")
        file_selection_layout = QVBoxLayout(file_selection_group)

        # Radio buttons for selection mode
        self.all_files_radio = QRadioButton("Process all files in current directory")
        self.all_files_radio.setChecked(True)
        self.selected_files_radio = QRadioButton("Process selected files")

        file_selection_layout.addWidget(self.all_files_radio)
        file_selection_layout.addWidget(self.selected_files_radio)

        # Browse button and file info
        browse_layout = QHBoxLayout()
        self.browse_button = QPushButton("Browse...")
        self.browse_button.setEnabled(False)
        self.file_count_label = QLabel("")

        browse_layout.addWidget(self.browse_button)
        browse_layout.addWidget(self.file_count_label)
        browse_layout.addStretch()

        file_selection_layout.addLayout(browse_layout)

        # Connect signals
        self.selected_files_radio.toggled.connect(self.on_file_selection_mode_changed)
        self.browse_button.clicked.connect(self.browse_files)

        self.skip_existing_check = QCheckBox()
        self.skip_existing_check.setChecked(True)
        self.skip_existing_check.setText("Skip images with existing boxes")
        file_selection_layout.addWidget(self.skip_existing_check)

        layout.addWidget(file_selection_group)

        # Detection group
        detection_group = QGroupBox("Detection")
        detection_layout = QVBoxLayout(detection_group)
        run_detection_layout = QHBoxLayout()

        self.run_detection_check = QCheckBox()
        self.run_detection_check.setChecked(True)
        self.run_detection_check.setText("Run detection strategy: ")

        # Detection strategy combo
        self.detection_strategy_combo = QComboBox()
        for strategy in DETECTION_STRATEGIES.values():
            self.detection_strategy_combo.addItem(strategy.name, strategy.name)

        # Set default from settings
        if app_settings.detection_strategy:
            for i in range(self.detection_strategy_combo.count()):
                if (
                    self.detection_strategy_combo.itemData(i)
                    == app_settings.detection_strategy
                ):
                    self.detection_strategy_combo.setCurrentIndex(i)
                    break

        run_detection_layout.addWidget(self.run_detection_check)
        run_detection_layout.addWidget(self.detection_strategy_combo)
        detection_layout.addLayout(run_detection_layout)

        layout.addWidget(detection_group)

        # Refinement group
        refinement_group = QGroupBox("Refinement")
        refinement_layout = QVBoxLayout(refinement_group)

        run_refinement_layout = QHBoxLayout()

        self.run_refinement_check = QCheckBox()
        self.run_refinement_check.setChecked(True)
        self.run_refinement_check.setText("Run refinement strategy: ")

        # Refinement strategy combo
        self.refinement_strategy_combo = QComboBox()
        for strategy in REFINEMENT_STRATEGIES.values():
            self.refinement_strategy_combo.addItem(strategy.name, strategy.name)

        # Set default from settings
        if app_settings.refinement_strategy:
            for i in range(self.refinement_strategy_combo.count()):
                if (
                    self.refinement_strategy_combo.itemData(i)
                    == app_settings.refinement_strategy
                ):
                    self.refinement_strategy_combo.setCurrentIndex(i)
                    break
        run_refinement_layout.addWidget(self.run_refinement_check)
        run_refinement_layout.addWidget(self.refinement_strategy_combo)

        refinement_layout.addLayout(run_refinement_layout)

        # Refinement tolerance
        tolerance_row = QHBoxLayout()

        tolerance_label = QLabel()
        tolerance_label.setText("Refinement tolerance: ")
        tolerance_row.addWidget(tolerance_label)

        self.tolerance_slider = QSlider(Qt.Orientation.Horizontal)
        self.tolerance_slider.setMinimum(1)  # 0.01
        self.tolerance_slider.setMaximum(15)  # 0.15
        self.tolerance_slider.setValue(int(app_settings.refine_default_tolerance * 100))
        self.tolerance_slider.setFixedWidth(100)
        tolerance_row.addWidget(self.tolerance_slider)

        self.tolerance_value_label = QLabel(
            f"{app_settings.refine_default_tolerance:.2f}"
        )
        # Avoid a blowup where QT seems to think this label (and thus the toolbar) needs
        # a ton of vertical space.
        self.tolerance_value_label.setFixedHeight(20)
        tolerance_row.addWidget(self.tolerance_value_label)
        self.tolerance_slider.valueChanged.connect(
            lambda value: self.tolerance_value_label.setText(f"{value / 100.0:.2f}")
        )

        refinement_layout.addLayout(tolerance_row)

        layout.addWidget(refinement_group)

        shrink_group = QGroupBox("Shrink boxes")
        shrink_row = QHBoxLayout(shrink_group)

        # Shrink pixels
        self.shrink_checkbox = QCheckBox()
        self.shrink_checkbox.setText("Shrink boxes by")
        self.shrink_checkbox.setChecked(app_settings.shrink_after_refinement != 0)
        shrink_row.addWidget(self.shrink_checkbox)
        self.shrink_pixels_edit = QLineEdit()
        self.shrink_pixels_edit.setText(str(app_settings.shrink_after_refinement))
        self.shrink_pixels_edit.setMaximumWidth(40)
        shrink_row.addWidget(self.shrink_pixels_edit)

        shrink_pixels_label = QLabel()
        shrink_pixels_label.setText("pixels")
        shrink_row.addWidget(shrink_pixels_label)

        layout.addWidget(shrink_group)

        # Progress group
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(True)
        progress_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setVisible(True)
        progress_layout.addWidget(self.status_label)

        # Log widget
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setMaximumHeight(200)
        self.log_widget.setVisible(True)
        self.log_widget.setPlainText("")
        progress_layout.addWidget(self.log_widget)

        layout.addWidget(progress_group)

        # Buttons
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        run_button = self.button_box.button(QDialogButtonBox.StandardButton.Ok)
        if run_button:
            run_button.setText("Run")
        self.button_box.accepted.connect(self.start_processing)
        self.button_box.rejected.connect(self.cancel_or_close)

        layout.addWidget(self.button_box)

    def on_file_selection_mode_changed(self):
        """Handle changes to file selection mode radio buttons."""
        is_selected_files_mode = self.selected_files_radio.isChecked()
        self.browse_button.setEnabled(is_selected_files_mode)

        if not is_selected_files_mode:
            # Clear selected files when switching to "all files" mode
            self.selected_files = []
            self.file_count_label.setText("")

    def browse_files(self):
        """Open file dialog to select specific files."""
        if not self.directory:
            return

        # Define supported image formats
        image_filters = [
            "Image files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.gif)",
            "PNG files (*.png)",
            "JPEG files (*.jpg *.jpeg)",
            "All files (*)",
        ]

        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Images to Process", self.directory, ";;".join(image_filters)
        )

        if files:
            # Validate that all files are in the current directory
            valid_files = []
            invalid_files = []
            for file_path in files:
                file_dir = os.path.dirname(file_path)
                if file_dir == self.directory:
                    valid_files.append(file_path)
                else:
                    invalid_files.append(os.path.basename(file_path))

            # Show warning if some files were from different directories
            if invalid_files:
                invalid_count = len(invalid_files)
                if invalid_count == 1:
                    message = f"The file '{invalid_files[0]}' is not in the current directory and will be skipped."
                else:
                    message = f"{invalid_count} files are not in the current directory and will be skipped."
                QMessageBox.warning(self, "Files Skipped", message)

            self.selected_files = valid_files
            count = len(self.selected_files)
            if count > 0:
                self.file_count_label.setText(
                    f"{count} file{'s' if count != 1 else ''} selected"
                )
            else:
                self.file_count_label.setText("No valid files selected")

    def start_processing(self):
        """Start the batch processing operation."""
        if not self.directory or not self.storage:
            return

        # Get selected options
        if self.run_detection_check.isChecked():
            detection_strategy_name = self.detection_strategy_combo.currentData()
        else:
            detection_strategy_name = ""

        if self.run_refinement_check.isChecked():
            refinement_strategy_name = self.refinement_strategy_combo.currentData()
        else:
            refinement_strategy_name = ""
        refinement_tolerance = self.tolerance_slider.value() / 100.0

        if self.shrink_checkbox.isChecked():
            shrink_pixels = int(self.shrink_pixels_edit.text().strip())
        else:
            shrink_pixels = 0

        skip_existing = self.skip_existing_check.isChecked()

        # Validate that at least one operation is selected
        if (
            not detection_strategy_name
            and not refinement_strategy_name
            and not shrink_pixels
        ):
            return  # Nothing to do

        # Get list of files to process based on selection mode
        if self.all_files_radio.isChecked():
            image_files = get_image_files_in_directory(self.directory)
        else:
            image_files = self.selected_files

        # Validate we have files to process
        if not image_files:
            self.show_error("No image files to process.")
            return

        # Create and start processor
        self.processor = BatchProcessor(
            image_files,
            self.storage,
            detection_strategy_name,
            skip_existing,
            refinement_strategy_name,
            refinement_tolerance,
            shrink_pixels,
        )

        # Connect signals
        self.processor.progress_updated.connect(self.update_progress)
        self.processor.finished_processing.connect(self.processing_finished)
        self.processor.error_occurred.connect(self.show_error)
        self.processor.log_message.connect(self.append_log_message)

        # Update UI for processing state
        self.progress_bar.setVisible(True)
        self.status_label.setVisible(True)
        self.log_widget.setVisible(True)
        self.log_widget.clear()  # Clear previous log
        ok_button = self.button_box.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button:
            ok_button.setEnabled(False)
        cancel_button = self.button_box.button(QDialogButtonBox.StandardButton.Cancel)
        if cancel_button:
            cancel_button.setText("Cancel")

        # Start processing
        self.processor.start()

    def update_progress(self, current: int, total: int, status: str):
        """Update progress bar and status."""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.status_label.setText(status)

    def append_log_message(self, message: str):
        """Append a message to the log with timestamp and auto-scroll."""
        from datetime import datetime

        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"

        # Append to log widget
        self.log_widget.append(formatted_message)

        # Auto-scroll to bottom
        scrollbar = self.log_widget.verticalScrollBar()
        if scrollbar:
            scrollbar.setValue(scrollbar.maximum())

    def processing_finished(self, processed: int, skipped: int, errors: int):
        """Handle processing completion."""
        self.progress_bar.setVisible(False)

        # Show summary
        parts = []
        if processed > 0:
            parts.append(f"Processed: {processed}")
        if skipped > 0:
            parts.append(f"Skipped: {skipped}")
        if errors > 0:
            parts.append(f"Errors: {errors}")

        summary = "Completed. " + ", ".join(parts) if parts else "No images processed."
        self.status_label.setText(summary)

        # Reset UI
        ok_button = self.button_box.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button:
            ok_button.setEnabled(True)
        cancel_button = self.button_box.button(QDialogButtonBox.StandardButton.Cancel)
        if cancel_button:
            cancel_button.setText("Close")

        # Clean up processor
        if self.processor:
            self.processor.deleteLater()
            self.processor = None

    def show_error(self, error_message: str):
        """Show error message."""
        self.status_label.setText(f"Error: {error_message}")
        self.progress_bar.setVisible(False)

        # Reset UI
        ok_button = self.button_box.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button:
            ok_button.setEnabled(True)
        cancel_button = self.button_box.button(QDialogButtonBox.StandardButton.Cancel)
        if cancel_button:
            cancel_button.setText("Close")

    def cancel_or_close(self):
        """Cancel processing or close dialog."""
        if self.processor and self.processor.isRunning():
            self.processor.cancel()
            self.processor.wait()  # Wait for thread to finish

        self.reject()
