"""
Directory sidebar widget for browsing images.
"""

import glob
import os

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.bounding_box_storage import BoundingBoxStorage
from core.validation_utils import (
    FileValidationSummary,
    get_validation_icon_text,
    validate_directory_files,
    validate_file_bounding_boxes,
)


class DirectoryImageList(QWidget):
    """Sidebar widget showing images in the current directory."""

    image_selected = pyqtSignal(str)  # Emits the full path of selected image
    directory_changed = pyqtSignal(str)  # Emits when user selects new directory
    batch_preprocess_requested = (
        pyqtSignal()
    )  # Emits when user requests batch preprocessing
    extract_photos_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.current_directory = None
        self.validation_cache: dict[str, FileValidationSummary] = {}
        self.supported_formats = {
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tiff",
            ".tif",
            ".gif",
        }

        # Set up the widget
        self.setMaximumWidth(250)
        self.setMinimumWidth(200)

        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Open directory button
        self.open_dir_btn = QPushButton("Open Directory")
        self.open_dir_btn.clicked.connect(self.open_directory)
        layout.addWidget(self.open_dir_btn)

        # Current directory label
        self.dir_label = QLabel("No directory selected")
        self.dir_label.setWordWrap(True)
        self.dir_label.setStyleSheet(
            "QLabel { font-size: 9pt; color: #666; padding: 5px; }"
        )
        layout.addWidget(self.dir_label)

        # Image list
        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.on_item_clicked)
        layout.addWidget(self.image_list)

        # Batch preprocess button
        self.batch_preprocess_btn = QPushButton("Batch preprocess...")
        self.batch_preprocess_btn.clicked.connect(self.batch_preprocess_requested.emit)
        self.batch_preprocess_btn.setEnabled(False)  # Disabled until directory is set
        layout.addWidget(self.batch_preprocess_btn)

        # Extract button
        self.extract_btn = QPushButton("Extract photos...")
        self.extract_btn.clicked.connect(self.extract_photos_requested.emit)
        self.extract_btn.setEnabled(False)  # Disabled until directory is set
        layout.addWidget(self.extract_btn)

    def set_directory(self, directory, storage: BoundingBoxStorage):
        """Set the directory to display images from."""
        if directory != self.current_directory:
            self.current_directory = directory
            self.validation_cache.clear()  # Clear cache for new directory
            self.update_directory_label()
            self.refresh_images()
            self._load_validation_cache(storage=storage)

        # Store storage reference for later use (e.g., in invalidate_validation_cache)
        self.storage = storage

        # Enable batch preprocess button when directory is set
        self.batch_preprocess_btn.setEnabled(bool(directory))

    def update_directory_label(self):
        """Update the directory label with current path."""
        if self.current_directory:
            # Show just the directory name and parent for space efficiency
            dir_name = os.path.basename(self.current_directory)
            parent_name = os.path.basename(os.path.dirname(self.current_directory))
            if parent_name and dir_name:
                display_text = f".../{parent_name}/{dir_name}"
            elif dir_name:
                display_text = dir_name
            else:
                display_text = self.current_directory
            self.dir_label.setText(display_text)
            self.dir_label.setToolTip(self.current_directory)  # Full path on hover
        else:
            self.dir_label.setText("No directory selected")
            self.dir_label.setToolTip("")

    def refresh_images(self):
        """Refresh the list of images in the current directory."""
        self.image_list.clear()

        if not self.current_directory or not os.path.isdir(self.current_directory):
            return

        # Find all image files in the directory
        image_files = []
        for ext in self.supported_formats:
            pattern = os.path.join(self.current_directory, f"*{ext}")
            image_files.extend(glob.glob(pattern, recursive=False))
            pattern = os.path.join(self.current_directory, f"*{ext.upper()}")
            image_files.extend(glob.glob(pattern, recursive=False))

        # Sort files alphabetically
        image_files.sort()

        # Add items to the list
        for image_path in image_files:
            filename = os.path.basename(image_path)
            self._add_image_item(filename, image_path)

    def on_item_clicked(self, item):
        """Handle image selection."""
        image_path = item.data(Qt.ItemDataRole.UserRole)
        if image_path:
            self.image_selected.emit(image_path)

    def select_image(self, image_path):
        """Select the specified image in the list."""
        filename = os.path.basename(image_path)
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            if item is not None and item.text() == filename:
                self.image_list.setCurrentItem(item)
                break

    def open_directory(self):
        """Open directory selection dialog."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory", self.current_directory or os.getcwd()
        )
        if directory:
            self.directory_changed.emit(directory)

    def get_first_image(self):
        """Get the path of the first image in the current directory."""
        if self.image_list.count() > 0:
            first_item = self.image_list.item(0)
            if first_item is not None:
                return first_item.data(Qt.ItemDataRole.UserRole)
        return None

    def currentRow(self):
        """Get the current row for navigation."""
        return self.image_list.currentRow()

    def count(self):
        """Get the count of images."""
        return self.image_list.count()

    def item(self, row):
        """Get item at specified row."""
        return self.image_list.item(row)

    def _load_validation_cache(self, storage):
        """Load validation data for all files in the current directory."""
        if not self.current_directory:
            return

        self.validation_cache = validate_directory_files(
            self.current_directory, storage
        )
        self._update_all_item_display()

    def _add_image_item(self, filename: str, image_path: str):
        """Add an image item to the list with validation icon if needed."""
        # Get validation icon for this file
        validation_icon = ""
        if filename in self.validation_cache:
            validation_icon = get_validation_icon_text(self.validation_cache[filename])

        # Create display text with icon (if any)
        display_text = f"{validation_icon} {filename}".strip()

        item = QListWidgetItem(display_text)
        item.setData(Qt.ItemDataRole.UserRole, image_path)  # Store full path
        item.setToolTip(self._get_validation_tooltip(filename))
        self.image_list.addItem(item)

    def _update_all_item_display(self):
        """Update display text for all items with current validation state."""
        items_updated = 0
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            if item is not None:
                image_path = item.data(Qt.ItemDataRole.UserRole)
                if image_path:
                    filename = os.path.basename(image_path)
                    old_text = item.text()

                    # Get validation icon for this file
                    validation_icon = ""
                    if filename in self.validation_cache:
                        validation_icon = get_validation_icon_text(
                            self.validation_cache[filename]
                        )

                    # Update display text with icon (if any)
                    display_text = f"{validation_icon} {filename}".strip()
                    if old_text != display_text:
                        items_updated += 1
                    item.setText(display_text)
                    item.setToolTip(self._get_validation_tooltip(filename))

        # Debug info - this will help see if the method is being called
        if items_updated > 0:
            print(
                f"DEBUG: Updated {items_updated} directory items with new validation icons"
            )

    def _get_validation_tooltip(self, filename: str) -> str:
        """Get tooltip text showing validation summary for a file."""
        if filename not in self.validation_cache:
            return ""

        summary = self.validation_cache[filename]
        if summary.error_count == 0 and summary.warning_count == 0:
            return "No validation issues"

        parts = []
        if summary.error_count > 0:
            parts.append(f"{summary.error_count} error(s)")
        if summary.warning_count > 0:
            parts.append(f"{summary.warning_count} warning(s)")

        return f"Validation issues: {', '.join(parts)}"

    def update_file_validation(self, filename: str, storage):
        """Update validation cache for a specific file and refresh its display."""
        if not self.current_directory:
            return

        # Use the improved validate_file_bounding_boxes with optional storage parameter
        self.validation_cache[filename] = validate_file_bounding_boxes(
            self.current_directory, filename, storage
        )

        self._update_all_item_display()

    def invalidate_validation_cache(self):
        """Invalidate and reload validation cache for current directory."""
        if (
            not self.current_directory
            or not hasattr(self, "storage")
            or not self.storage
        ):
            return

        # Preserve current selection
        current_selection = self.image_list.currentRow()

        # Clear cache and reload validation
        self.validation_cache.clear()
        self._load_validation_cache(self.storage)

        # Restore selection
        if current_selection >= 0:
            self.image_list.setCurrentRow(current_selection)
