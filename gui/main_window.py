"""
Main application window for the Photo Album Extractor.
"""

import os
from PyQt6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, 
                             QPushButton, QFileDialog, QLabel, QMessageBox,
                             QStatusBar, QLineEdit, QComboBox, QSplitter,
                             QProgressDialog)
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QAction, QPixmap

from image_processing.image_processor import ImageProcessor
from gui.quad_bounding_box import QuadBoundingBox
from image_processing.detection_strategies import DETECTION_STRATEGIES
from storage.bounding_box_storage import BoundingBoxStorage
from gui.settings_dialog import Settings, SettingsDialog
from gui.directory_sidebar import DirectoryImageList
from gui.image_view import ImageView


class PhotoExtractorApp(QMainWindow):
    """Main application window."""
    
    def __init__(self, initial_image=None, initial_directory=None):
        super().__init__()
        
        self.image_processor = ImageProcessor()
        self.current_image_path = None
        self.current_directory = None
        self.bounding_box_storage = None
        self.settings = Settings()
        
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
        
        
        # Base name input
        toolbar_layout.addWidget(QLabel("Base Name:"))
        self.base_name_edit = QLineEdit("photo")
        self.base_name_edit.setMaximumWidth(100)
        toolbar_layout.addWidget(self.base_name_edit)
        
        # Extract button
        self.extract_btn = QPushButton("Extract Photos")
        self.extract_btn.clicked.connect(self.extract_photos)
        self.extract_btn.setEnabled(False)
        toolbar_layout.addWidget(self.extract_btn)
        
        # Clear all button
        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self.clear_all_boxes)
        self.clear_btn.setEnabled(False)
        toolbar_layout.addWidget(self.clear_btn)
        
        # Detection strategy dropdown
        toolbar_layout.addWidget(QLabel("Detection:"))
        self.strategy_combo = QComboBox()
        for strategy in DETECTION_STRATEGIES:
            self.strategy_combo.addItem(strategy.name, strategy)
        self.strategy_combo.setMinimumWidth(150)
        toolbar_layout.addWidget(self.strategy_combo)
        
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
        
        
        toolbar_layout.addStretch()
        main_layout.addLayout(toolbar_layout)
        
        # Create splitter for sidebar and main content
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Create directory image list (sidebar)
        self.directory_list = DirectoryImageList()
        self.directory_list.image_selected.connect(self.on_image_selected)
        self.directory_list.directory_changed.connect(self.on_directory_changed)
        self.directory_list.batch_detect_requested.connect(self.batch_detect_photos)
        self.directory_list.batch_extract_requested.connect(self.batch_extract_photos)
        splitter.addWidget(self.directory_list)
        
        # Create image view
        self.image_view = ImageView()
        self.image_view.add_box_requested.connect(self.add_bounding_box)
        self.image_view.remove_box_requested.connect(self.remove_bounding_box)
        self.image_view.boxes_changed.connect(self.update_extract_button_state)
        splitter.addWidget(self.image_view)
        
        # Set splitter proportions (sidebar smaller than main view)
        splitter.setSizes([250, 1000])
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Load an image to begin")
        
    def create_menu_bar(self):
        """Create the application menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        load_action = QAction('Load Image', self)
        load_action.setShortcut('Ctrl+O')
        load_action.triggered.connect(self.load_image)
        file_menu.addAction(load_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu('Edit')
        
        clear_action = QAction('Clear All Boxes', self)
        clear_action.triggered.connect(self.clear_all_boxes)
        edit_menu.addAction(clear_action)
        
        edit_menu.addSeparator()
        
        settings_action = QAction('Settings...', self)
        settings_action.setShortcut('Ctrl+,')
        settings_action.triggered.connect(self.open_settings)
        edit_menu.addAction(settings_action)
        
        # View menu
        view_menu = menubar.addMenu('View')
        
        zoom_in_action = QAction('Zoom In', self)
        zoom_in_action.setShortcut('Ctrl+=')
        zoom_in_action.triggered.connect(self.zoom_in)
        view_menu.addAction(zoom_in_action)
        
        zoom_out_action = QAction('Zoom Out', self)
        zoom_out_action.setShortcut('Ctrl+-')
        zoom_out_action.triggered.connect(self.zoom_out)
        view_menu.addAction(zoom_out_action)
        
        view_menu.addSeparator()
        
        fit_action = QAction('Fit Image', self)
        fit_action.setShortcut('Ctrl+0')
        fit_action.triggered.connect(self.fit_image)
        view_menu.addAction(fit_action)
        
        view_menu.addSeparator()
        
        prev_image_action = QAction('Previous Image', self)
        prev_image_action.setShortcut('Left')
        prev_image_action.triggered.connect(self.previous_image)
        view_menu.addAction(prev_image_action)
        
        next_image_action = QAction('Next Image', self)
        next_image_action.setShortcut('Right')
        next_image_action.triggered.connect(self.next_image)
        view_menu.addAction(next_image_action)
        
    def load_image(self):
        """Load an image file using file dialog."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Load Image", 
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.gif)"
        )
        
        if file_path:
            self.load_image_from_path(file_path)
            
    def load_image_from_path(self, file_path):
        """Load an image from the specified file path."""
        # Save current bounding boxes before switching images
        if self.current_image_path and self.bounding_box_storage:
            self.save_current_bounding_boxes()
            
        if self.image_processor.load_image(file_path):
            pixmap = self.image_processor.get_pixmap()
            if pixmap:
                self.image_view.set_image(pixmap)
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
            else:
                QMessageBox.warning(self, "Error", "Failed to display image")
        else:
            QMessageBox.warning(self, "Error", "Failed to load image")
            
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
            self.directory_list.set_directory(directory)
            
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
            self.image_view.set_image(QPixmap())  # Clear image
            self.status_bar.showMessage(f"No images found in directory: {os.path.basename(directory)}")
            self.update_extract_button_state()
            return True
                
    def add_bounding_box(self, x, y):
        """Add a new bounding box at the specified position."""
        self.image_view.add_bounding_box(x, y)
        
    def remove_bounding_box(self, box):
        """Remove the specified bounding box."""
        self.image_view.remove_bounding_box(box)
        
    def clear_all_boxes(self):
        """Clear all bounding boxes."""
        self.image_view.clear_boxes()
        
    def detect_photos(self):
        """Run the selected detection strategy to automatically detect photos."""
        if not self.current_image_path:
            return
            
        # Get the selected strategy
        selected_strategy = self.strategy_combo.currentData()
        if not selected_strategy:
            return
            
        # Get image dimensions
        if not self.image_view.image_item:
            return
            
        pixmap = self.image_view.image_item.pixmap()
        image_width = pixmap.width()
        image_height = pixmap.height()
        
        # Clear existing boxes first
        self.image_view.clear_boxes()
        
        # Configure API key for strategies that need it
        if hasattr(selected_strategy, 'set_api_key'):
            api_key = self.settings.get('gemini_api_key', '')
            if api_key:
                selected_strategy.set_api_key(api_key)
            else:
                QMessageBox.warning(self, "API Key Required", 
                                  f"The {selected_strategy.name} strategy requires an API key. "
                                  "Please configure it in Edit > Settings.")
                return
        
        # Run detection strategy
        try:
            detected_quads = selected_strategy.detect_photos(image_width, image_height, self.current_image_path)
            
            # Convert relative coordinates to scene coordinates and create boxes
            for quad_corners in detected_quads:
                # Convert relative coordinates (0-1) to scene coordinates
                scene_corners = []
                for corner in quad_corners:
                    scene_x = corner.x() * image_width
                    scene_y = corner.y() * image_height
                    scene_corners.append(QPointF(scene_x, scene_y))
                
                # Create the bounding box
                box = QuadBoundingBox(scene_corners)
                self.image_view.add_bounding_box_object(box)
                
            self.status_bar.showMessage(f"Detected {len(detected_quads)} photos using {selected_strategy.name}")
            
        except Exception as e:
            QMessageBox.warning(self, "Detection Error", f"Failed to detect photos: {str(e)}")
        
    def update_extract_button_state(self):
        """Enable/disable extract, clear, and detect buttons based on current state."""
        has_image = self.current_image_path is not None
        has_boxes = len(self.image_view.bounding_boxes) > 0
        
        self.extract_btn.setEnabled(has_image and has_boxes)
        self.clear_btn.setEnabled(has_boxes)
        self.detect_btn.setEnabled(has_image)
        self.refine_all_btn.setEnabled(has_image and has_boxes)
        
    def extract_photos(self):
        """Extract all photos based on bounding boxes."""
        if not self.current_image_path:
            QMessageBox.warning(self, "Error", "Please load an image first")
            return
            
        crop_data = self.image_view.get_crop_rects()
        if not crop_data:
            QMessageBox.warning(self, "Error", "No bounding boxes found")
            return
            
        # Prompt for output directory
        output_directory = QFileDialog.getExistingDirectory(
            self, "Select Output Folder for Extracted Photos"
        )
        
        if not output_directory:
            return  # User cancelled
            
        # Extract and save photos
        base_name = self.base_name_edit.text() or "photo"
        saved_files = self.image_processor.save_cropped_images(
            crop_data, output_directory, base_name
        )
        
        if saved_files:
            QMessageBox.information(
                self, "Success", 
                f"Extracted {len(saved_files)} photos to:\n{output_directory}"
            )
            self.status_bar.showMessage(f"Extracted {len(saved_files)} photos")
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
            self.image_view.fitInView(self.image_view.image_item, Qt.AspectRatioMode.KeepAspectRatio)
            
    def save_current_bounding_boxes(self):
        """Save the current bounding boxes to storage."""
        if self.current_image_path and self.bounding_box_storage:
            filename = os.path.basename(self.current_image_path)
            self.bounding_box_storage.save_bounding_boxes(filename, self.image_view.bounding_boxes)
            
    def load_saved_bounding_boxes(self):
        """Load saved bounding boxes for the current image."""
        if self.current_image_path and self.bounding_box_storage:
            filename = os.path.basename(self.current_image_path)
            saved_boxes = self.bounding_box_storage.load_bounding_boxes(filename)
            
            # Clear existing boxes
            self.image_view.clear_boxes()
            
            # Load saved boxes
            for box_data in saved_boxes:
                if box_data.get('type') == 'quad' and 'corners' in box_data:
                    corners = [QPointF(corner[0], corner[1]) for corner in box_data['corners']]
                    box = QuadBoundingBox(corners)
                    self.image_view.add_bounding_box_object(box)
                    
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
                
    def batch_detect_photos(self):
        """Run detection strategy on all images in the current directory."""
        if not self.current_directory:
            return
            
        # Get the selected strategy
        selected_strategy = self.strategy_combo.currentData()
        if not selected_strategy:
            QMessageBox.warning(self, "Error", "No detection strategy selected")
            return
            
        # Get all images in the directory
        all_image_paths = []
        for i in range(self.directory_list.count()):
            item = self.directory_list.item(i)
            image_path = item.data(Qt.ItemDataRole.UserRole)
            all_image_paths.append(image_path)
            
        if not all_image_paths:
            QMessageBox.information(self, "No Images", "No images found in current directory")
            return
            
        # Configure API key for strategies that need it
        if hasattr(selected_strategy, 'set_api_key'):
            api_key = self.settings.get('gemini_api_key', '')
            if api_key:
                selected_strategy.set_api_key(api_key)
            else:
                QMessageBox.warning(self, "API Key Required", 
                                  f"The {selected_strategy.name} strategy requires an API key. "
                                  "Please configure it in Edit > Settings.")
                return
            
        # Create progress dialog
        progress = QProgressDialog("Detecting photos...", "Cancel", 0, len(all_image_paths), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        
        successful_detections = 0
        failed_detections = 0
        
        for i, image_path in enumerate(all_image_paths):
            if progress.wasCanceled():
                break
                
            progress.setValue(i)
            progress.setLabelText(f"Processing {os.path.basename(image_path)}...")
            
            try:
                # Load image temporarily to get dimensions
                temp_processor = ImageProcessor()
                if temp_processor.load_image(image_path):
                    pixmap = temp_processor.get_pixmap()
                    if pixmap:
                        image_width = pixmap.width()
                        image_height = pixmap.height()
                        
                        # Run detection strategy
                        detected_quads = selected_strategy.detect_photos(image_width, image_height, image_path)
                        
                        # Convert to storage format
                        box_data = []
                        for quad_corners in detected_quads:
                            # Convert relative coordinates to absolute coordinates
                            absolute_corners = []
                            for corner in quad_corners:
                                abs_x = corner.x() * image_width
                                abs_y = corner.y() * image_height
                                absolute_corners.append([abs_x, abs_y])
                            box_data.append({'type': 'quad', 'corners': absolute_corners})
                        
                        # Save to storage
                        filename = os.path.basename(image_path)
                        self.bounding_box_storage.save_bounding_boxes(filename, [])  # Clear existing
                        if box_data:
                            self.bounding_box_storage.data[filename] = box_data
                            self.bounding_box_storage.save_data()
                        
                        successful_detections += 1
                    else:
                        failed_detections += 1
                else:
                    failed_detections += 1
                    
            except Exception as e:
                print(f"Failed to detect photos in {image_path}: {e}")
                failed_detections += 1
                
        progress.setValue(len(all_image_paths))
        progress.close()
        
        # Refresh current image if it has saved boxes
        if self.current_image_path:
            self.load_saved_bounding_boxes()
            
        # Show results
        total_processed = successful_detections + failed_detections
        message = f"Batch detection completed:\n\n"
        message += f"Successfully processed: {successful_detections}/{total_processed} images\n"
        message += f"Using strategy: {selected_strategy.name}"
        
        if failed_detections > 0:
            message += f"\n\nFailed to process: {failed_detections} images"
            
        QMessageBox.information(self, "Batch Detection Complete", message)
        self.status_bar.showMessage(f"Batch detection: {successful_detections}/{total_processed} images processed")
        
    def batch_extract_photos(self):
        """Extract photos from all images in the current directory using stored bounding boxes."""
        if not self.current_directory:
            return
            
        # Get all images in the directory
        all_image_paths = []
        for i in range(self.directory_list.count()):
            item = self.directory_list.item(i)
            image_path = item.data(Qt.ItemDataRole.UserRole)
            all_image_paths.append(image_path)
            
        if not all_image_paths:
            QMessageBox.information(self, "No Images", "No images found in current directory")
            return
            
        # Count images that have stored bounding boxes
        images_with_boxes = 0
        for image_path in all_image_paths:
            filename = os.path.basename(image_path)
            saved_boxes = self.bounding_box_storage.load_bounding_boxes(filename)
            if saved_boxes:
                images_with_boxes += 1
                
        if images_with_boxes == 0:
            QMessageBox.information(self, "No Bounding Boxes", 
                                  "No stored bounding boxes found for images in this directory. "
                                  "Run 'Batch Detect' or manually add bounding boxes first.")
            return
            
        # Prompt for output directory
        output_directory = QFileDialog.getExistingDirectory(
            self, "Select Output Folder for Batch Extraction"
        )
        
        if not output_directory:
            return  # User cancelled
            
        # Create progress dialog
        progress = QProgressDialog("Extracting photos...", "Cancel", 0, images_with_boxes, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        
        successful_extractions = 0
        total_photos_extracted = 0
        processed_images = 0
        
        for image_path in all_image_paths:
            filename = os.path.basename(image_path)
            saved_boxes = self.bounding_box_storage.load_bounding_boxes(filename)
            
            if not saved_boxes:
                continue  # Skip images without bounding boxes
                
            if progress.wasCanceled():
                break
                
            progress.setValue(processed_images)
            progress.setLabelText(f"Extracting from {filename}...")
            processed_images += 1
            
            try:
                # Load image temporarily
                temp_processor = ImageProcessor()
                if temp_processor.load_image(image_path):
                    # Convert saved boxes to crop data format
                    crop_data = []
                    for box_data in saved_boxes:
                        if box_data.get('type') == 'quad' and 'corners' in box_data:
                            # Convert absolute coordinates back to relative coordinates
                            pixmap = temp_processor.get_pixmap()
                            if pixmap:
                                image_width = pixmap.width()
                                image_height = pixmap.height()
                                
                                rel_corners = []
                                for corner in box_data['corners']:
                                    rel_x = corner[0] / image_width
                                    rel_y = corner[1] / image_height
                                    rel_corners.append((rel_x, rel_y))
                                crop_data.append(('quad', rel_corners))
                    
                    if crop_data:
                        # Use filename without extension as base name
                        base_name = os.path.splitext(filename)[0]
                        
                        # Extract and save photos
                        saved_files = temp_processor.save_cropped_images(
                            crop_data, output_directory, base_name
                        )
                        
                        if saved_files:
                            successful_extractions += 1
                            total_photos_extracted += len(saved_files)
                        else:
                            print(f"Failed to extract photos from {filename}")
                    
            except Exception as e:
                print(f"Failed to extract photos from {image_path}: {e}")
                
        progress.setValue(images_with_boxes)
        progress.close()
        
        # Show results
        message = f"Batch extraction completed:\n\n"
        message += f"Successfully processed: {successful_extractions}/{images_with_boxes} images\n"
        message += f"Total photos extracted: {total_photos_extracted}\n"
        message += f"Output directory: {output_directory}"
        
        QMessageBox.information(self, "Batch Extraction Complete", message)
        self.status_bar.showMessage(f"Batch extraction: {total_photos_extracted} photos from {successful_extractions} images")
        
    def open_settings(self):
        """Open the settings dialog."""
        dialog = SettingsDialog(self.settings, self)
        dialog.exec()