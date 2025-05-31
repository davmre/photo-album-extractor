"""
Main GUI application for the Photo Album Extractor.
"""

import os
from PyQt6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, 
                             QPushButton, QFileDialog, QLabel, QMessageBox,
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                             QMenuBar, QMenu, QStatusBar, QLineEdit, QComboBox)
from PyQt6.QtCore import Qt, pyqtSignal, QRectF, QPointF
from PyQt6.QtGui import QAction, QPixmap, QPainter

from image_processor import ImageProcessor
from quad_bounding_box import QuadBoundingBox, QuadEdgeLine
from detection_strategies import DETECTION_STRATEGIES
import refine_bounds

class ImageView(QGraphicsView):
    """Custom graphics view for displaying images with bounding box interaction."""
    
    # Signal emitted when user right-clicks to add a new box
    add_box_requested = pyqtSignal(float, float)
    # Signal emitted when user right-clicks on a box to remove it
    remove_box_requested = pyqtSignal(object)
    # Signal emitted when boxes are added or removed
    boxes_changed = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        
        # Set up the graphics scene
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        
        # Image item
        self.image_item = None
        self.bounding_boxes = []
        
        # View settings
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Enable context menu
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        
        # Drag state for creating boxes
        self.is_dragging = False
        self.drag_start_pos = None
        self.temp_box = None
        
    def set_image(self, pixmap):
        """Set the image to display."""
        # Clear existing image
        if self.image_item:
            self.scene.removeItem(self.image_item)
            
        # Add new image
        self.image_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.image_item)
        
        # Clear existing bounding boxes
        self.clear_boxes()
        
        # Fit image in view
        self.fitInView(self.image_item, Qt.AspectRatioMode.KeepAspectRatio)
        
    def add_bounding_box(self, x, y, width=100, height=100, quad=True):
        """Add a new bounding box at the specified position."""
        if self.image_item is None:
            return None
            
        # Create quadrilateral bounding box by default
        # Create initial rectangle corners
        corners = [
            QPointF(x - width/2, y - height/2),  # top-left
            QPointF(x + width/2, y - height/2),  # top-right
            QPointF(x + width/2, y + height/2),  # bottom-right
            QPointF(x - width/2, y + height/2)   # bottom-left
        ]
        box = QuadBoundingBox(corners)
            
        self.scene.addItem(box)
        
        # Add handles to scene
        if hasattr(box, 'handles'):
            for handle in box.handles:
                self.scene.addItem(handle)
                
        # Add edge lines for quadrilateral boxes
        if isinstance(box, QuadBoundingBox):
            for i in range(4):
                edge_line = QuadEdgeLine(box, i)
                box.edge_lines = getattr(box, 'edge_lines', [])
                box.edge_lines.append(edge_line)
                self.scene.addItem(edge_line)
                edge_line.update_edge_geometry()
                
        self.bounding_boxes.append(box)
        
        # Connect signals
        box.changed.connect(self.box_changed)
        if isinstance(box, QuadBoundingBox):
            box.changed.connect(self.update_edge_lines)
        
        # Emit signal that boxes changed
        self.boxes_changed.emit()
        
        return box
        
    def add_bounding_box_object(self, box):
        """Add a pre-created bounding box object to the scene."""
        if self.image_item is None:
            return None
            
        self.scene.addItem(box)
        
        # Add handles to scene
        if hasattr(box, 'handles'):
            for handle in box.handles:
                self.scene.addItem(handle)
                
        # Add edge lines for quadrilateral boxes
        if isinstance(box, QuadBoundingBox):
            for i in range(4):
                edge_line = QuadEdgeLine(box, i)
                box.edge_lines = getattr(box, 'edge_lines', [])
                box.edge_lines.append(edge_line)
                self.scene.addItem(edge_line)
                edge_line.update_edge_geometry()
                
        self.bounding_boxes.append(box)
        
        # Connect signals
        box.changed.connect(self.box_changed)
        if isinstance(box, QuadBoundingBox):
            box.changed.connect(self.update_edge_lines)
        
        # Emit signal that boxes changed
        self.boxes_changed.emit()
        
        return box
        
    def remove_bounding_box(self, box):
        """Remove a bounding box from the scene."""
        if box in self.bounding_boxes:
            # Remove handles
            if hasattr(box, 'handles'):
                for handle in box.handles:
                    self.scene.removeItem(handle)
                    
            # Remove edge lines for quadrilateral boxes
            if hasattr(box, 'edge_lines'):
                for edge_line in box.edge_lines:
                    self.scene.removeItem(edge_line)
                
            # Remove box
            self.scene.removeItem(box)
            self.bounding_boxes.remove(box)
            
            # Emit signal that boxes changed
            self.boxes_changed.emit()
            
    def clear_boxes(self):
        """Remove all bounding boxes."""
        if self.bounding_boxes:  # Only emit signal if there were boxes to remove
            for box in self.bounding_boxes[:]:  # Copy list to avoid modification during iteration
                self.remove_bounding_box(box)
            # Note: remove_bounding_box already emits boxes_changed for each removal
            
    def get_crop_rects(self):
        """Get all bounding box rectangles/polygons in scene coordinates."""
        crop_data = []
        for box in self.bounding_boxes:
            # Get corner points for quadrilateral box in proper extraction order
            corners = box.get_corner_points_for_extraction()
            if self.image_item:
                img_rect = self.image_item.boundingRect()
                # Convert to relative coordinates within image
                rel_corners = []
                for corner in corners:
                    rel_x = (corner.x() - img_rect.x()) / img_rect.width()
                    rel_y = (corner.y() - img_rect.y()) / img_rect.height()
                    rel_corners.append((rel_x, rel_y))
                crop_data.append(('quad', rel_corners))
                
        return crop_data
        
    def show_context_menu(self, position):
        """Show context menu for adding/removing boxes."""
        # Convert position to scene coordinates
        scene_pos = self.mapToScene(position)
        
        # Check if we're clicking on a bounding box
        clicked_item = self.scene.itemAt(scene_pos, self.transform())
        clicked_box = None
        
        # Find the bounding box if we clicked on one or its handle
        for box in self.bounding_boxes:
            if clicked_item == box or (hasattr(box, 'handles') and clicked_item in box.handles):
                clicked_box = box
                break
                
        # Create context menu
        menu = QMenu(self)
        
        if clicked_box:
            # Menu for existing box
            refine_action = menu.addAction("Refine Bounding Box")
            refine_parallel_action = menu.addAction("Refine Bounding Box (enforce parallel)")
            refine_multiscale_action = menu.addAction("Refine Bounding Box multiscale (enforce parallel)")
            remove_action = menu.addAction("Remove Bounding Box")
            action = menu.exec(self.mapToGlobal(position))
            
            if action == refine_action:
                self.refine_bounding_box(clicked_box)
            if action == refine_parallel_action:
                self.refine_bounding_box(clicked_box,
                                         enforce_parallel_sides=True)
            if action == refine_multiscale_action:
                self.refine_bounding_box(clicked_box,
                                         enforce_parallel_sides=True, 
                                         multiscale=True)
            elif action == remove_action:
                self.remove_box_requested.emit(clicked_box)
        else:
            # Menu for adding new box
            add_action = menu.addAction("Add Bounding Box")
            action = menu.exec(self.mapToGlobal(position))
            
            if action == add_action:
                self.add_box_requested.emit(scene_pos.x(), scene_pos.y())
                
    def box_changed(self):
        """Handle bounding box changes (for future features like live preview)."""
        pass
        
    def update_edge_lines(self):
        """Update edge line geometries when quadrilateral changes."""
        for box in self.bounding_boxes:
            if isinstance(box, QuadBoundingBox) and hasattr(box, 'edge_lines'):
                for edge_line in box.edge_lines:
                    edge_line.update_edge_geometry()
                    
    def refine_bounding_box(self, box, multiscale=False, enforce_parallel_sides=False):
        """Refine a single bounding box using edge detection."""
        if not self.image_item or not isinstance(box, QuadBoundingBox):
            return
            
        # Get the current image as numpy array
        pixmap = self.image_item.pixmap()
        # Convert QPixmap to numpy array
        image = pixmap.toImage()
        width = image.width()
        height = image.height()
        
        # Convert QImage to numpy array
        import numpy as np
        ptr = image.constBits()
        ptr.setsize(height * width * 4)  # 4 bytes per pixel (RGBA)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        # Convert RGBA to BGR for OpenCV
        image_bgr = arr[:, :, [2, 1, 0]]  # BGR format
        
        # Get current box corners in image coordinates
        corners = box.get_corner_points_for_extraction()
        corner_coords = []
        for corner in corners:
            corner_coords.append((corner.x(), corner.y()))
            
        try:
            # Run refinement
            if multiscale:
                refined_corners = refine_bounds.refine_bounding_box_multiscale(
                image_bgr, corner_coords,
                enforce_parallel_sides=enforce_parallel_sides)
            else:
                refined_corners = refine_bounds.refine_bounding_box(
                    image_bgr, corner_coords,
                    enforce_parallel_sides=enforce_parallel_sides)
            
            # Update box with refined corners
            refined_qpoints = []
            for corner in refined_corners:
                refined_qpoints.append(QPointF(float(corner[0]), float(corner[1])))
                
            box.set_corners(refined_qpoints)
            
            # Update edge lines
            if hasattr(box, 'edge_lines'):
                for edge_line in box.edge_lines:
                    edge_line.update_edge_geometry()
                    
        except Exception as e:
            print(f"Error refining bounding box: {e}")
            
    def refine_all_bounding_boxes(self):
        """Refine all bounding boxes using edge detection."""
        for box in self.bounding_boxes:
            if isinstance(box, QuadBoundingBox):
                self.refine_bounding_box(box)
        
    def mousePressEvent(self, event):
        """Handle mouse press for starting box creation."""
        if event.button() == Qt.MouseButton.LeftButton:
            # Check if we're clicking on an existing item
            scene_pos = self.mapToScene(event.position().toPoint())
            clicked_item = self.scene.itemAt(scene_pos, self.transform())
            
            # If we didn't click on an existing item, start creating a new box
            if clicked_item is None or clicked_item == self.image_item:
                self.is_dragging = True
                self.drag_start_pos = scene_pos
                self.setDragMode(QGraphicsView.DragMode.NoDrag)  # Disable default drag
                return
                
        super().mousePressEvent(event)
        
    def mouseMoveEvent(self, event):
        """Handle mouse move for box creation."""
        if self.is_dragging and self.drag_start_pos:
            scene_pos = self.mapToScene(event.position().toPoint())
            
            # Remove temporary box if it exists
            if self.temp_box:
                self.scene.removeItem(self.temp_box)
                self.temp_box = None
                
            # Create temporary quadrilateral box for preview
            corners = [
                self.drag_start_pos,
                QPointF(scene_pos.x(), self.drag_start_pos.y()),
                scene_pos,
                QPointF(self.drag_start_pos.x(), scene_pos.y())
            ]
            self.temp_box = QuadBoundingBox(corners)
            self.scene.addItem(self.temp_box)
            
        super().mouseMoveEvent(event)
        
    def mouseReleaseEvent(self, event):
        """Handle mouse release for finishing box creation."""
        if event.button() == Qt.MouseButton.LeftButton and self.is_dragging:
            scene_pos = self.mapToScene(event.position().toPoint())
            
            # Remove temporary box
            if self.temp_box:
                self.scene.removeItem(self.temp_box)
                self.temp_box = None
                
            # Create final box if drag was significant
            if self.drag_start_pos:
                distance = ((scene_pos.x() - self.drag_start_pos.x())**2 + 
                           (scene_pos.y() - self.drag_start_pos.y())**2)**0.5
                if distance > 10:  # Minimum drag distance
                    # Create the actual quadrilateral bounding box
                    corners = [
                        self.drag_start_pos,
                        QPointF(scene_pos.x(), self.drag_start_pos.y()),
                        scene_pos,
                        QPointF(self.drag_start_pos.x(), scene_pos.y())
                    ]
                    box = QuadBoundingBox(corners)
                    self.scene.addItem(box)
                    
                    # Add handles to scene
                    if hasattr(box, 'handles'):
                        for handle in box.handles:
                            self.scene.addItem(handle)
                            
                    # Add edge lines
                    for i in range(4):
                        edge_line = QuadEdgeLine(box, i)
                        box.edge_lines = getattr(box, 'edge_lines', [])
                        box.edge_lines.append(edge_line)
                        self.scene.addItem(edge_line)
                        edge_line.update_edge_geometry()
                        
                    self.bounding_boxes.append(box)
                    box.changed.connect(self.box_changed)
                    box.changed.connect(self.update_edge_lines)
                    
                    # Emit signal that boxes changed
                    self.boxes_changed.emit()
                    
            # Reset drag state
            self.is_dragging = False
            self.drag_start_pos = None
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)  # Re-enable default drag
            
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        """Handle zoom with mouse wheel or trackpad."""
        # On macOS, distinguish between scroll and zoom gestures
        # If Ctrl/Cmd is held down, zoom; otherwise, scroll normally
        if event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier):
            # Zoom when Ctrl/Cmd is held
            zoom_factor = 1.15
            if event.angleDelta().y() >= 0:
                zoom_factor = 1.0 / zoom_factor
            self.scale(zoom_factor, zoom_factor)
        else:
            # Normal scrolling behavior
            super().wheelEvent(event)

class PhotoExtractorApp(QMainWindow):
    """Main application window."""
    
    def __init__(self, initial_image=None):
        super().__init__()
        
        self.image_processor = ImageProcessor()
        self.current_image_path = None
        self.output_directory = None
        
        self.init_ui()
        
        # Load initial image if provided
        if initial_image:
            self.load_image_from_path(initial_image)
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Photo Album Extractor")
        self.setGeometry(100, 100, 1200, 800)
        
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
        
        self.output_btn = QPushButton("Set Output Folder")
        self.output_btn.clicked.connect(self.set_output_folder)
        toolbar_layout.addWidget(self.output_btn)
        
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
        
        # Create image view
        self.image_view = ImageView()
        self.image_view.add_box_requested.connect(self.add_bounding_box)
        self.image_view.remove_box_requested.connect(self.remove_bounding_box)
        self.image_view.boxes_changed.connect(self.update_extract_button_state)
        main_layout.addWidget(self.image_view)
        
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
        
    def load_image(self):
        """Load an image file using file dialog."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Load Image", 
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.gif)"
        )
        
        if file_path:
            self.load_image_from_path(file_path)
            
    def load_image_from_path(self, file_path):
        """Load an image from the specified file path."""
        if self.image_processor.load_image(file_path):
            pixmap = self.image_processor.get_pixmap()
            if pixmap:
                self.image_view.set_image(pixmap)
                self.current_image_path = file_path
                self.status_bar.showMessage(f"Loaded: {os.path.basename(file_path)}")
                self.update_extract_button_state()
            else:
                QMessageBox.warning(self, "Error", "Failed to display image")
        else:
            QMessageBox.warning(self, "Error", "Failed to load image")
                
    def set_output_folder(self):
        """Set the output directory for extracted photos."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Output Folder"
        )
        
        if directory:
            self.output_directory = directory
            self.status_bar.showMessage(f"Output folder: {directory}")
            self.update_extract_button_state()
            
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
        has_output = self.output_directory is not None
        has_boxes = len(self.image_view.bounding_boxes) > 0
        
        self.extract_btn.setEnabled(has_image and has_output and has_boxes)
        self.clear_btn.setEnabled(has_boxes)
        self.detect_btn.setEnabled(has_image)
        self.refine_all_btn.setEnabled(has_image and has_boxes)
        
    def extract_photos(self):
        """Extract all photos based on bounding boxes."""
        if not self.current_image_path or not self.output_directory:
            QMessageBox.warning(self, "Error", "Please load an image and set output folder")
            return
            
        crop_data = self.image_view.get_crop_rects()
        if not crop_data:
            QMessageBox.warning(self, "Error", "No bounding boxes found")
            return
            
        # Extract and save photos
        base_name = self.base_name_edit.text() or "photo"
        saved_files = self.image_processor.save_cropped_images(
            crop_data, self.output_directory, base_name
        )
        
        if saved_files:
            QMessageBox.information(
                self, "Success", 
                f"Extracted {len(saved_files)} photos to:\n{self.output_directory}"
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