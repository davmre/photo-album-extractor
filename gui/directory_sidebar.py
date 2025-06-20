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


class DirectoryImageList(QWidget):
    """Sidebar widget showing images in the current directory."""

    image_selected = pyqtSignal(str)  # Emits the full path of selected image
    directory_changed = pyqtSignal(str)  # Emits when user selects new directory

    def __init__(self):
        super().__init__()
        self.current_directory = None
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

    def set_directory(self, directory):
        """Set the directory to display images from."""
        if directory != self.current_directory:
            self.current_directory = directory
            self.update_directory_label()
            self.refresh_images()

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
            item = QListWidgetItem(filename)
            item.setData(Qt.ItemDataRole.UserRole, image_path)  # Store full path
            self.image_list.addItem(item)

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
