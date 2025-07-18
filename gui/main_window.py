"""
Main application window for the Photo Album Extractor.
"""

from __future__ import annotations

import os

import PIL.Image
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from core import images
from core.bounding_box import BoundingBox, PhotoAttributes
from core.bounding_box_storage import BoundingBoxStorage
from core.date_inference import infer_dates_for_current_directory
from core.detection_strategies import configure_detection_strategy
from core.errors import AppError
from core.settings import app_settings
from gui.attributes_sidebar import AttributesSidebar
from gui.batch_preprocess import BatchPreprocessDialog
from gui.directory_sidebar import DirectoryImageList
from gui.image_view import ImageView
from gui.settings_dialog import SettingsDialog


class PhotoExtractorApp(QMainWindow):
    """Main application window."""

    def __init__(
        self,
        initial_image: str | None = None,
        initial_directory: str | None = None,
        refine_debug_dir: str | None = None,
    ) -> None:
        super().__init__()

        self.refine_debug_dir = refine_debug_dir

        self.current_image_path: str | None = None
        self.current_image: PIL.Image.Image | None = None
        self.current_directory: str | None = None
        self.bounding_box_storage: BoundingBoxStorage | None = None

        # GUI components (will be initialized in init_ui)
        self.image_view: ImageView
        self.directory_list: DirectoryImageList
        self.attributes_sidebar: AttributesSidebar
        self.status_bar: QStatusBar
        self.settings_dialog: SettingsDialog

        self.init_ui()

        # Load initial content based on what was provided
        if initial_image:
            self.load_image_from_path(initial_image)
        elif initial_directory:
            self.load_directory(initial_directory)
        else:
            # Set current working directory as default
            self.set_current_directory(os.getcwd())

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Photo Album Extractor")
        self.setGeometry(100, 100, 1400, 800)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create main layout
        main_layout = QVBoxLayout(central_widget)

        # Create menu bar
        self.create_menu_bar()

        # Create toolbar
        toolbar_layout = QHBoxLayout()

        # File selection buttons
        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)
        toolbar_layout.addWidget(self.load_btn)

        # Extract button
        self.extract_btn = QPushButton("Extract Photos")
        self.extract_btn.clicked.connect(self.extract_photos)
        self.extract_btn.setEnabled(False)
        toolbar_layout.addWidget(self.extract_btn)

        # Detect photos button
        self.detect_btn = QPushButton("Detect Photos")
        self.detect_btn.clicked.connect(self.detect_photos)
        self.detect_btn.setEnabled(False)
        toolbar_layout.addWidget(self.detect_btn)

        # Refine all button
        self.refine_all_btn = QPushButton("Refine All")
        self.refine_all_btn.clicked.connect(self.refine_all_boxes)
        self.refine_all_btn.setEnabled(False)
        toolbar_layout.addWidget(self.refine_all_btn)

        # Clear all button
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_all_boxes)
        self.clear_btn.setEnabled(False)
        toolbar_layout.addWidget(self.clear_btn)

        # Add tolerance slider
        tolerance_label = QLabel("Refinement tolerance:")
        toolbar_layout.addWidget(tolerance_label)

        self.tolerance_slider = QSlider(Qt.Orientation.Horizontal)
        self.tolerance_slider.setMinimum(1)  # 0.01
        self.tolerance_slider.setMaximum(15)  # 0.15
        self.tolerance_slider.setValue(int(app_settings.refine_default_tolerance * 100))
        self.tolerance_slider.setFixedWidth(100)
        self.tolerance_slider.valueChanged.connect(self.on_tolerance_changed)
        toolbar_layout.addWidget(self.tolerance_slider)

        self.tolerance_value_label = QLabel(
            f"{app_settings.refine_default_tolerance:.2f}"
        )
        # Avoid a blowup where QT seems to think this label (and thus the toolbar) needs
        # a ton of vertical space.
        self.tolerance_value_label.setFixedHeight(20)

        self.tolerance_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        toolbar_layout.addWidget(self.tolerance_value_label)

        toolbar_layout.addStretch()
        main_layout.addLayout(toolbar_layout)

        # Create main horizontal splitter for three-panel layout
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Create directory image list (left sidebar)
        self.directory_list = DirectoryImageList()
        self.directory_list.image_selected.connect(self.on_image_selected)
        self.directory_list.directory_changed.connect(self.on_directory_changed)
        self.directory_list.batch_preprocess_requested.connect(
            self.open_batch_preprocess_dialog
        )
        # self.directory_list.batch_detect_requested.connect(self.batch_detect_photos)
        # self.directory_list.batch_extract_requested.connect(self.batch_extract_photos)
        main_splitter.addWidget(self.directory_list)

        # Create image view (center panel)
        self.image_view = ImageView()
        self.image_view.boxes_changed.connect(self.on_box_changed)
        self.image_view.box_selected.connect(self.on_box_selected)
        self.image_view.box_deselected.connect(self.on_box_deselected)
        main_splitter.addWidget(self.image_view)

        # Create attributes sidebar (right panel)
        self.attributes_sidebar = AttributesSidebar()
        self.attributes_sidebar.attributes_changed.connect(self.on_attributes_changed)
        main_splitter.addWidget(self.attributes_sidebar)

        # Connect magnifier signals
        self.image_view.mouse_moved.connect(
            self.attributes_sidebar.magnifier.set_cursor_position
        )
        self.image_view.image_updated.connect(self.on_image_updated)

        self.image_view.mouse_entered_viewport.connect(
            self.attributes_sidebar.magnifier.resume_cursor_tracking
        )

        # Set splitter proportions: [left sidebar, main view, right sidebar]
        main_splitter.setSizes([250, 800, 300])

        main_layout.addWidget(main_splitter)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Load an image to begin")

        # Create settings dialog and connect validation settings change signal
        self.settings_dialog = SettingsDialog(self)
        self.settings_dialog.validation_settings_changed.connect(
            self.directory_list.invalidate_validation_cache
        )
        self.settings_dialog.validation_settings_changed.connect(
            self.attributes_sidebar.refresh_validation_display
        )
        self.settings_dialog.validation_settings_changed.connect(
            self.image_view.refresh_all_validation
        )

    def create_menu_bar(self):
        """Create the application menu bar."""
        menubar = self.menuBar()
        if not menubar:
            # Silly error case to convince the type checker that `menubar` is not None
            # below this point. (pattern also used for individual menus below).
            raise ValueError("Error creating menu bar")

        # File menu
        file_menu = menubar.addMenu("File")
        if not file_menu:
            raise ValueError("Error creating file menu")

        load_action = QAction("Load Image", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self.load_image)
        file_menu.addAction(load_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit menu
        edit_menu = menubar.addMenu("Edit")
        if not edit_menu:
            raise ValueError("Error creating edit menu")

        clear_action = QAction("Clear All Boxes", self)
        clear_action.triggered.connect(self.clear_all_boxes)
        edit_menu.addAction(clear_action)

        edit_menu.addSeparator()

        settings_action = QAction("Settings...", self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.triggered.connect(self.open_settings)
        edit_menu.addAction(settings_action)

        # View menu
        view_menu = menubar.addMenu("View")
        if not view_menu:
            raise ValueError("Error creating view menu")

        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.setShortcut("Ctrl+=")
        zoom_in_action.triggered.connect(self.zoom_in)
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.setShortcut("Ctrl+-")
        zoom_out_action.triggered.connect(self.zoom_out)
        view_menu.addAction(zoom_out_action)

        view_menu.addSeparator()

        fit_action = QAction("Fit Image", self)
        fit_action.setShortcut("Ctrl+0")
        fit_action.triggered.connect(self.fit_image)
        view_menu.addAction(fit_action)

        view_menu.addSeparator()

        prev_image_action = QAction("Previous Image", self)
        prev_image_action.setShortcut("Ctrl+Left")
        prev_image_action.triggered.connect(self.previous_image)
        view_menu.addAction(prev_image_action)

        next_image_action = QAction("Next Image", self)
        next_image_action.setShortcut("Ctrl+Right")
        next_image_action.triggered.connect(self.next_image)
        view_menu.addAction(next_image_action)

    def load_image(self):
        """Load an image file using file dialog."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.gif)",
        )

        if file_path:
            self.load_image_from_path(file_path)

    def load_image_from_path(self, file_path):
        """Load an image from the specified file path."""
        # Save current bounding boxes before switching images
        if self.current_image_path and self.bounding_box_storage:
            self.save_current_bounding_boxes()

        try:
            image = images.load_image(file_path)
            self.image_view.set_image(image)
            if self.refine_debug_dir:
                self.image_view.refine_debug_dir = os.path.join(
                    self.refine_debug_dir,
                    os.path.splitext(os.path.basename(file_path))[0],
                )
                print("IMAGE RDD", self.image_view.refine_debug_dir)
            self.current_image = image
            self.current_image_path = file_path

            # Update directory context
            new_directory = os.path.dirname(file_path)
            self.set_current_directory(new_directory)

            # Load saved bounding boxes for this image
            self.load_saved_bounding_boxes()

            # Update sidebar selection
            self.directory_list.select_image(file_path)

            self.status_bar.showMessage(f"Loaded: {os.path.basename(file_path)}")
            self.update_extract_button_state()

            return True  # Load succeeded.

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load image: {e}")
            # Revert to previous selection in directory listing.
            self.directory_list.select_image(self.current_image_path)

        return False

    def on_image_selected(self, image_path):
        """Handle image selection from the directory list."""
        self.load_image_from_path(image_path)

    def on_directory_changed(self, directory):
        """Handle directory change from the sidebar."""
        self.load_directory(directory)

    def set_current_directory(self, directory):
        """Set the current directory and update the sidebar."""
        if directory != self.current_directory:
            self.current_directory = directory
            self.bounding_box_storage = BoundingBoxStorage(directory)
            self.directory_list.set_directory(
                directory, storage=self.bounding_box_storage
            )

    def load_directory(self, directory):
        """Load a directory and its first image."""
        if not os.path.isdir(directory):
            QMessageBox.warning(self, "Error", f"Directory not found: {directory}")
            return False

        # Set the directory first
        self.set_current_directory(directory)

        # Load the first image in the directory
        first_image = self.directory_list.get_first_image()
        if first_image:
            self.load_image_from_path(first_image)
            return True
        else:
            # No images in directory, just clear the current image
            if self.current_image_path:
                # Save current boxes before clearing
                self.save_current_bounding_boxes()
            self.current_image_path = None
            self.image_view.set_image()  # Clear image
            self.status_bar.showMessage(
                f"No images found in directory: {os.path.basename(directory)}"
            )
            self.update_extract_button_state()
            return True

    def clear_all_boxes(self):
        """Clear all bounding boxes."""
        self.image_view.clear_boxes()

    def on_tolerance_changed(self, value: int):
        """Handle tolerance slider value changes."""
        tolerance = value / 100.0
        app_settings.refine_current_tolerance = tolerance
        self.tolerance_value_label.setText(f"{tolerance:.2f}")

    def on_box_selected(self, bounding_box_data: BoundingBox):
        """Handle box selection from ImageView."""
        # Extract coordinates for the sidebar coordinate display
        self.attributes_sidebar.set_box_data(bounding_box_data)

    def on_box_deselected(self):
        """Handle box deselection from ImageView."""
        self.attributes_sidebar.show_no_selection()

    def on_attributes_changed(
        self, box_id: str, key_changed: str, attributes: PhotoAttributes
    ):
        """Handle attribute changes from AttributesSidebar."""
        # Get current box to extract its corners
        selected_box = self.image_view.get_selected_box()
        if selected_box and selected_box.box_id == box_id:
            # Create updated BoundingBoxData with new attributes but existing corners
            current_data = selected_box.get_bounding_box_data()
            updated_data = BoundingBox(
                corners=current_data.corners, box_id=box_id, attributes=attributes
            )

            # Update the box and save
            self.image_view.update_box_data(updated_data)

            if self.current_image_path and self.bounding_box_storage:
                filename = os.path.basename(self.current_image_path)
                self.bounding_box_storage.update_box_data(filename, updated_data)

                # Update directory sidebar validation for this file
                self.directory_list.update_file_validation(
                    filename, storage=self.bounding_box_storage
                )

            # If date hint changed, trigger date inference for the whole directory
            if key_changed == "date_hint":
                self._trigger_date_inference()

    def _trigger_date_inference(self):
        """
        Run date inference for the entire directory and update UI accordingly.

        This fills in exif_date fields and sets date_inconsistent flags based on
        the date hints across all photos in the directory.
        """
        if not self.bounding_box_storage:
            return

        try:
            # Run date inference on the entire directory
            result = infer_dates_for_current_directory(self.bounding_box_storage)
            if result.updated_files:
                # Update validation cache for all affected files using the current storage object
                for filename in result.updated_files:
                    self.directory_list.update_file_validation(
                        filename, self.bounding_box_storage
                    )

                # If the current image was updated, reload and refresh the UI
                if self.current_image_path:
                    current_filename = os.path.basename(self.current_image_path)
                    if current_filename in result.updated_files:
                        # Reload bounding boxes from storage (they've been updated).
                        self.load_saved_bounding_boxes()

                # Show status message about what was updated
                if result.total_boxes_updated > 0:
                    msg = f"Updated {result.total_boxes_updated} photos"
                    if result.inconsistent_boxes_updated > 0:
                        msg += f" ({result.inconsistent_boxes_updated} marked as inconsistent)"
                    self.status_bar.showMessage(msg, 3000)  # Show for 3 seconds

        except Exception as e:
            QMessageBox.warning(
                self, "Date Inference Error", f"Failed to infer dates: {e}"
            )

    def detect_photos(self):
        """Run the configured detection strategy to automatically detect photos."""
        if (
            not self.current_image_path
            or not self.current_image
            or not self.bounding_box_storage
        ):
            return

        try:
            selected_strategy = configure_detection_strategy(app_settings)
        except AppError as err:
            err.show_warning(parent=self)
            return

        # Clear existing boxes
        self.image_view.clear_boxes()

        # Run detection strategy
        try:
            detected_boxes = selected_strategy.detect_photos(self.current_image)

            # Create bounding boxes
            for box_data in detected_boxes:
                self.image_view.add_bounding_box(box_data)

            # Save detected bounding boxes.
            filename = os.path.basename(self.current_image_path)
            self.bounding_box_storage.set_bounding_boxes(filename, detected_boxes)
            # Update directory sidebar validation for this file
            self.directory_list.update_file_validation(
                filename, storage=self.bounding_box_storage
            )

            self.status_bar.showMessage(
                f"Detected {len(detected_boxes)} photos using {selected_strategy.name}"
            )

            # Auto-refine if enabled in settings
            if app_settings.auto_refine_detection and detected_boxes:
                self.refine_all_boxes()
                self.status_bar.showMessage(
                    f"Detected and refined {len(detected_boxes)} photos using {selected_strategy.name}"
                )

        except Exception as e:
            QMessageBox.warning(
                self, "Detection Error", f"Failed to detect photos: {str(e)}"
            )

    def update_extract_button_state(self):
        """Enable/disable extract, clear, and detect buttons based on current state."""
        has_image = self.current_image_path is not None
        has_boxes = len(self.image_view.bounding_boxes) > 0

        self.extract_btn.setEnabled(has_image and has_boxes)
        self.clear_btn.setEnabled(has_boxes)
        self.detect_btn.setEnabled(has_image)
        self.refine_all_btn.setEnabled(has_image and has_boxes)

    def extract_photos(self, output_directory=None):
        """Extract all photos based on bounding boxes."""
        if not self.current_image_path or not self.current_image:
            QMessageBox.warning(self, "Error", "Please load an image first")
            return

        bounding_box_data_list = self.image_view.get_bounding_box_data_list()
        if not bounding_box_data_list:
            QMessageBox.warning(self, "Error", "No bounding boxes found")
            return

        if not output_directory:
            # Prompt for output directory
            output_directory = QFileDialog.getExistingDirectory(
                self, "Select Output Folder for Extracted Photos"
            )

        if not output_directory:
            return []  # User cancelled

        # Extract and save photos with unified bounding box data
        base_name = "photo"
        saved_files = images.save_cropped_images(
            self.current_image,
            bounding_box_data_list,
            output_directory,
            base_name,
            source_image_path=self.current_image_path,
        )

        if saved_files:
            QMessageBox.information(
                self,
                "Success",
                f"Extracted {len(saved_files)} photos to:\n{output_directory}",
            )
            self.status_bar.showMessage(f"Extracted {len(saved_files)} photos")
            return saved_files
        else:
            QMessageBox.warning(self, "Error", "Failed to extract photos")

    def refine_all_boxes(self):
        """Refine all bounding boxes using edge detection."""
        self.image_view.refine_all_bounding_boxes()
        self.status_bar.showMessage("Refined all bounding boxes")

    def zoom_in(self):
        """Zoom in the image view."""
        zoom_factor = 1.15
        self.image_view.scale(zoom_factor, zoom_factor)

    def zoom_out(self):
        """Zoom out the image view."""
        zoom_factor = 1.0 / 1.15
        self.image_view.scale(zoom_factor, zoom_factor)

    def fit_image(self):
        """Fit the image to the view."""
        if self.image_view.image_item:
            self.image_view.fitInView(
                self.image_view.image_item, Qt.AspectRatioMode.KeepAspectRatio
            )

    def save_current_bounding_boxes(self):
        """Save the current bounding boxes to storage."""
        # Bounding box storage should already have been updated on
        # any changes, so just ensure it's synced to disk.
        if self.bounding_box_storage:
            self.bounding_box_storage.save_data()

    def load_saved_bounding_boxes(self):
        """Load saved bounding boxes for the current image."""
        if self.current_image_path and self.bounding_box_storage:
            filename = os.path.basename(self.current_image_path)
            saved_data = self.bounding_box_storage.get_bounding_boxes(filename)

            # Remember the currently selected box id
            selected_box = self.image_view.selected_box
            selected_box_id = selected_box.box_id if selected_box else None

            # Clear existing boxes
            self.image_view.clear_boxes(emit_signals=False)

            # Load saved boxes using BoundingBoxData
            for bbox_data in saved_data:
                # Create box with BoundingBoxData
                self.image_view.add_bounding_box(bbox_data)

            # Restore the selected box (if any).
            if selected_box_id:
                self.image_view.on_box_selection_changed(selected_box_id)

    def closeEvent(self, event):
        """Save bounding boxes before closing the application."""
        if self.current_image_path and self.bounding_box_storage:
            self.save_current_bounding_boxes()
        event.accept()

    def previous_image(self):
        """Navigate to the previous image in the directory."""
        if not self.current_image_path:
            return

        current_row = self.directory_list.currentRow()
        if current_row > 0:
            previous_item = self.directory_list.item(current_row - 1)
            if previous_item:
                image_path = previous_item.data(Qt.ItemDataRole.UserRole)
                self.load_image_from_path(image_path)

    def next_image(self):
        """Navigate to the next image in the directory."""
        if not self.current_image_path:
            return

        current_row = self.directory_list.currentRow()
        if current_row < self.directory_list.count() - 1:
            next_item = self.directory_list.item(current_row + 1)
            if next_item:
                image_path = next_item.data(Qt.ItemDataRole.UserRole)
                self.load_image_from_path(image_path)

    def on_image_updated(self):
        if not self.image_view.image_item:
            return
        # Set magnifier source image
        pixmap = self.image_view.image_item.pixmap()
        self.attributes_sidebar.magnifier.set_source_image(pixmap)

    def on_box_changed(self):
        """Handle changes to box properties (eg corners) from the image view."""

        if not self.bounding_box_storage or not self.current_image_path:
            return

        filename = os.path.basename(self.current_image_path)
        bounding_box_data_list = self.image_view.get_bounding_box_data_list()
        self.bounding_box_storage.set_bounding_boxes(
            filename, bounding_box_data_list, save_data=False
        )
        # for box in bounding_box_data_list:
        #    self.bounding_box_storage.update_box_data(filename, box, save_data=False)
        self.directory_list.update_file_validation(
            filename, storage=self.bounding_box_storage
        )

        # Set magnifier bounding boxes
        self.attributes_sidebar.magnifier.set_bounding_boxes(bounding_box_data_list)

        # Update the attributes sidebar so it can display any validation warnings.
        selected_box = self.image_view.get_selected_box()
        if selected_box and self.attributes_sidebar.current_box:
            self.attributes_sidebar.set_box_data(selected_box.get_bounding_box_data())

        self.update_extract_button_state()

    def open_settings(self):
        """Open the settings dialog."""
        self.settings_dialog.exec()

    def open_batch_preprocess_dialog(self):
        """Open the batch preprocess dialog."""
        if not self.current_directory or not self.bounding_box_storage:
            QMessageBox.warning(
                self, "No Directory", "Please select a directory first."
            )
            return

        # Create and show the batch preprocess dialog
        dialog = BatchPreprocessDialog(
            parent=self,
            directory=self.current_directory,
            storage=self.bounding_box_storage,
        )

        # Connect to handle completion
        dialog.accepted.connect(self.on_batch_preprocess_completed)
        dialog.exec()

    def on_batch_preprocess_completed(self):
        """Handle batch preprocessing completion by refreshing the UI."""
        # Refresh the directory sidebar validation cache
        self.directory_list.invalidate_validation_cache()

        # Reload the current image's bounding boxes if an image is loaded
        if self.current_image_path:
            self.load_saved_bounding_boxes()

        # Update the status bar
        self.status_bar.showMessage("Batch preprocessing completed", 3000)
