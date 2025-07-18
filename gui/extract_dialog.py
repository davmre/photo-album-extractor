"""
Batch extraction dialog for extracting photos from multiple images.
"""

from __future__ import annotations

import os
from pathlib import Path

import PIL.Image
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
)

from core.bounding_box_storage import BoundingBoxStorage
from core.extract import FileExistsBehavior, OutputFormat, save_cropped_images


class ExtractProcessor(QThread):
    """Background thread for batch extraction operations."""

    progress_updated = pyqtSignal(int, int, str)  # current, total, status_message
    finished_processing = pyqtSignal(int, int, int)  # processed, skipped, errors
    error_occurred = pyqtSignal(str)  # error message
    log_message = pyqtSignal(str)  # detailed log message

    def __init__(
        self,
        image_files: list[Path],
        output_dir: str,
        output_format: OutputFormat,
        jpeg_quality: int,
        conflict_mode: FileExistsBehavior,
        storage: BoundingBoxStorage,
    ):
        super().__init__()
        self.image_files = image_files
        self.output_dir = output_dir
        self.output_format = output_format
        self.jpeg_quality = jpeg_quality
        self.conflict_mode = conflict_mode
        self.storage = storage
        self.cancelled = False

    def cancel(self):
        """Cancel the extraction operation."""
        self.cancelled = True

    def run(self):
        """Run the batch extraction operation."""
        if not self.image_files:
            self.log_message.emit("No image files found to process.")
            self.finished_processing.emit(0, 0, 0)
            return

        # Validate and create output directory
        try:
            output_path = Path(self.output_dir).expanduser()
            if not output_path.exists():
                output_path.mkdir(parents=True)
                self.log_message.emit(f"Created output directory: {output_path}")
        except OSError as e:
            self.error_occurred.emit(f"Error creating output directory: {e}")
            return

        self.log_message.emit(
            f"Starting batch extraction of {len(self.image_files)} images to {output_path}"
        )

        processed_count = 0
        skipped_count = 0
        error_count = 0
        total_photos_extracted = 0

        directory = self.image_files[0].parent
        storage = BoundingBoxStorage(str(directory))

        for i, image_path in enumerate(self.image_files):
            if self.cancelled:
                self.log_message.emit("Extraction cancelled by user.")
                break

            filename = image_path.name

            try:
                # Update progress
                self.progress_updated.emit(
                    i + 1,
                    len(self.image_files),
                    f"Processing source image {filename} ({i + 1}/{len(self.image_files)})",
                )

                self.log_message.emit(
                    f"--- Processing source image {filename} ({i + 1}/{len(self.image_files)}) ---"
                )

                # Extract photos from this image
                photos_extracted = self._extract_from_image(storage, image_path)

                if photos_extracted > 0:
                    processed_count += 1
                    total_photos_extracted += photos_extracted
                    self.log_message.emit(f"  ✓ Extracted {photos_extracted} photos")
                elif photos_extracted == 0:
                    skipped_count += 1
                    self.log_message.emit("  Skipped")
                else:
                    error_count += 1

            except Exception as e:
                error_count += 1
                self.log_message.emit(f"ERROR processing {filename}: {e}")

        self.log_message.emit(
            f"Batch extraction completed: {processed_count} images processed, "
            f"{total_photos_extracted} photos extracted, {skipped_count} skipped, {error_count} errors"
        )
        self.finished_processing.emit(processed_count, skipped_count, error_count)

    def _extract_from_image(self, storage: BoundingBoxStorage, image_path: Path) -> int:
        """Extract photos from a single image. Returns number of photos extracted, or -1 on error."""
        filename = image_path.name

        # Load bounding box data
        bounding_boxes = storage.get_bounding_boxes(filename)

        if not bounding_boxes:
            return 0  # No bounding boxes found

        # Load and process image
        with PIL.Image.open(image_path) as image:
            self.log_message.emit(f"  Loaded image ({image.size[0]}x{image.size[1]})")

            # Use source filename as base name (without extension)
            base_name = image_path.stem

            # Track progress per photo
            extracted_count = 0

            for j, (status, val) in enumerate(
                save_cropped_images(
                    image=image,
                    bounding_box_data_list=bounding_boxes,
                    output_dir=self.output_dir,
                    base_name=base_name,
                    source_image_path=str(image_path),
                    file_exists_behavior=self.conflict_mode,
                    output_format=self.output_format,
                )
            ):
                if status == "saved":
                    extracted_count += 1
                    self.log_message.emit(f"    Photo {j + 1}: {val}")
                elif status == "error":
                    self.log_message.emit(f"    Photo {j + 1}: failed to extract")
                    self.log_message.emit(str(val))
                elif status == "skipped":
                    self.log_message.emit(
                        f"    Photo {j + 1}: {val} exists, skipping..."
                    )

                if self.cancelled:
                    break

            return extracted_count


class ExtractDialog(QDialog):
    """Dialog for batch extraction operations."""

    error_occurred = pyqtSignal(str)  # error message

    def __init__(
        self,
        parent=None,
        current_directory: str = "",
        current_image_path: str = "",
        storage: BoundingBoxStorage | None = None,
    ):
        super().__init__(parent)
        self.current_directory = current_directory
        self.current_image_path = current_image_path
        self.storage = storage
        self.processor: ExtractProcessor | None = None

        self.init_ui()

        self.error_occurred.connect(self.error_message)

    def init_ui(self):
        """Initialize the dialog UI."""
        self.setWindowTitle("Batch Extract Photos")
        self.setMinimumSize(600, 700)
        self.resize(600, 700)

        layout = QVBoxLayout(self)

        # Output directory group
        output_group = QGroupBox("Output")
        output_layout = QFormLayout(output_group)

        # Output directory
        dir_row = QHBoxLayout()
        self.output_dir_edit = QLineEdit()

        # Default to "extracted_photos" subdirectory
        if self.current_directory:
            default_output = os.path.join(self.current_directory, "extracted_photos")
            self.output_dir_edit.setText(default_output)

        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_output_directory)

        dir_row.addWidget(self.output_dir_edit)
        dir_row.addWidget(self.browse_button)
        output_layout.addRow("Output directory:", dir_row)

        layout.addWidget(output_group)

        # Scope group
        scope_group = QGroupBox("Scope")
        scope_layout = QVBoxLayout(scope_group)

        self.scope_button_group = QButtonGroup()

        self.all_pages_radio = QRadioButton("All pages in directory")
        self.all_pages_radio.setChecked(True)
        self.scope_button_group.addButton(self.all_pages_radio)
        scope_layout.addWidget(self.all_pages_radio)

        self.current_page_radio = QRadioButton("Current page only")
        self.scope_button_group.addButton(self.current_page_radio)
        scope_layout.addWidget(self.current_page_radio)

        layout.addWidget(scope_group)

        # Format group
        format_group = QGroupBox("Output Format")
        format_layout = QFormLayout(format_group)

        self.format_combo = QComboBox()
        self.format_combo.addItem("JPEG", OutputFormat.JPEG)
        self.format_combo.addItem("PNG", OutputFormat.PNG)
        self.format_combo.addItem("TIFF", OutputFormat.TIFF)
        self.format_combo.currentTextChanged.connect(self.on_format_changed)
        format_layout.addRow("Format:", self.format_combo)

        # JPEG quality
        quality_row = QHBoxLayout()
        self.quality_spinbox = QSpinBox()
        self.quality_spinbox.setRange(1, 100)
        self.quality_spinbox.setValue(95)
        self.quality_spinbox.setSuffix("%")
        quality_row.addWidget(self.quality_spinbox)
        self.quality_label = QLabel("JPEG Quality:")
        format_layout.addRow(self.quality_label, quality_row)

        layout.addWidget(format_group)

        # Conflict handling group
        conflict_group = QGroupBox("File Conflicts")
        conflict_layout = QFormLayout(conflict_group)

        self.conflict_combo = QComboBox()
        self.conflict_combo.addItem(
            "Overwrite existing files", FileExistsBehavior.OVERWRITE
        )
        self.conflict_combo.addItem("Skip existing files", FileExistsBehavior.SKIP)
        self.conflict_combo.addItem("Add number suffix", FileExistsBehavior.INCREMENT)
        conflict_layout.addRow("If file exists:", self.conflict_combo)

        layout.addWidget(conflict_group)

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
        extract_button = self.button_box.button(QDialogButtonBox.StandardButton.Ok)
        if extract_button:
            extract_button.setText("Extract")
        self.button_box.accepted.connect(self.start_extraction)
        self.button_box.rejected.connect(self.cancel_or_close)

        layout.addWidget(self.button_box)

    def error_message(self, error_str: str):
        QMessageBox.warning(self, "Error", error_str)

    def on_format_changed(self, format_text: str):
        """Handle format selection change."""
        del format_text
        format: OutputFormat = self.format_combo.currentData()
        is_jpeg = format == OutputFormat.JPEG
        self.quality_label.setVisible(is_jpeg)
        self.quality_spinbox.setVisible(is_jpeg)

    def browse_output_directory(self):
        """Browse for output directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", self.output_dir_edit.text()
        )
        if directory:
            self.output_dir_edit.setText(directory)

    def start_extraction(self):
        """Start the batch extraction operation."""
        if not self.current_directory or not self.storage:
            self.error_occurred.emit("No directory or storage available")
            return

        # Get output directory
        output_dir = self.output_dir_edit.text().strip()
        if not output_dir:
            self.error_occurred.emit("Please specify an output directory")
            return

        # Get image files based on scope
        if self.current_page_radio.isChecked():
            if not self.current_image_path:
                self.error_occurred.emit("No current image loaded")
                return
            image_files = [Path(self.current_image_path)]
        else:
            # All pages in directory
            image_files = self._get_image_files_in_directory(self.current_directory)

        if not image_files:
            self.error_occurred.emit("No image files found")
            return

        # Get format settings
        output_format: OutputFormat = self.format_combo.currentData()
        jpeg_quality = (
            self.quality_spinbox.value() if output_format == OutputFormat.JPEG else 95
        )
        conflict_mode: FileExistsBehavior = self.conflict_combo.currentData()

        # Create and start processor
        self.processor = ExtractProcessor(
            image_files=image_files,
            output_dir=output_dir,
            output_format=output_format,
            jpeg_quality=jpeg_quality,
            conflict_mode=conflict_mode,
            storage=self.storage,
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
        extract_button = self.button_box.button(QDialogButtonBox.StandardButton.Ok)
        if extract_button:
            extract_button.setEnabled(False)
        cancel_button = self.button_box.button(QDialogButtonBox.StandardButton.Cancel)
        if cancel_button:
            cancel_button.setText("Cancel")

        # Start processing
        self.processor.start()

    def _get_image_files_in_directory(self, directory: str) -> list[Path]:
        """Get all image files in the directory."""
        supported_formats = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
        image_files = []

        dir_path = Path(directory)
        for file_path in dir_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_formats:
                image_files.append(file_path)

        return sorted(image_files)

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

        summary = (
            "Extraction completed. " + ", ".join(parts)
            if parts
            else "No images processed."
        )
        self.status_label.setText(summary)

        # Reset UI
        extract_button = self.button_box.button(QDialogButtonBox.StandardButton.Ok)
        if extract_button:
            extract_button.setEnabled(True)
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
        extract_button = self.button_box.button(QDialogButtonBox.StandardButton.Ok)
        if extract_button:
            extract_button.setEnabled(True)
        cancel_button = self.button_box.button(QDialogButtonBox.StandardButton.Cancel)
        if cancel_button:
            cancel_button.setText("Close")

    def cancel_or_close(self):
        """Cancel processing or close dialog."""
        if self.processor and self.processor.isRunning():
            self.processor.cancel()
            self.processor.wait()  # Wait for thread to finish

        self.reject()
