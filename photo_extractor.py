"""
Main GUI application for the Photo Album Extractor.
"""

import os
from PyQt6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, 
                             QPushButton, QFileDialog, QLabel, QMessageBox,
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                             QMenuBar, QMenu, QStatusBar, QLineEdit)
from PyQt6.QtCore import Qt, pyqtSignal, QRectF, QPointF
from PyQt6.QtGui import QAction, QPixmap, QPainter

from image_processor import ImageProcessor
from bounding_box import BoundingBox
from rotated_bounding_box import RotatedBoundingBox
from quad_bounding_box import QuadBoundingBox, QuadEdgeLine

class ImageView(QGraphicsView):
    """Custom graphics view for displaying images with bounding box interaction."""
    
    # Signal emitted when user right-clicks to add a new box
    add_box_requested = pyqtSignal(float, float)
    # Signal emitted when user right-clicks on a box to remove it
    remove_box_requested = pyqtSignal(object)
    
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
        if quad:
            # Create initial rectangle corners
            corners = [
                QPointF(x - width/2, y - height/2),  # top-left
                QPointF(x + width/2, y - height/2),  # top-right
                QPointF(x + width/2, y + height/2),  # bottom-right
                QPointF(x - width/2, y + height/2)   # bottom-left
            ]
            box = QuadBoundingBox(corners)
        else:
            # Fallback to old system for compatibility
            box = RotatedBoundingBox(x, y, width, height)
            
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
            
    def clear_boxes(self):
        """Remove all bounding boxes."""
        for box in self.bounding_boxes[:]:  # Copy list to avoid modification during iteration
            self.remove_bounding_box(box)
            
    def get_crop_rects(self):
        """Get all bounding box rectangles/polygons in scene coordinates."""
        crop_data = []
        for box in self.bounding_boxes:
            if isinstance(box, QuadBoundingBox):
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
            elif hasattr(box, 'get_corner_points'):  # Rotated box (legacy)
                # Get corner points for rotated box
                corners = box.get_corner_points()
                if self.image_item:
                    img_rect = self.image_item.boundingRect()
                    # Convert to relative coordinates within image
                    rel_corners = []
                    for corner in corners:
                        rel_x = (corner.x() - img_rect.x()) / img_rect.width()
                        rel_y = (corner.y() - img_rect.y()) / img_rect.height()
                        rel_corners.append((rel_x, rel_y))
                    crop_data.append(('rotated', rel_corners))
            else:  # Regular axis-aligned box
                scene_rect = box.sceneBoundingRect()
                if self.image_item:
                    img_rect = self.image_item.boundingRect()
                    rel_x = (scene_rect.x() - img_rect.x()) / img_rect.width()
                    rel_y = (scene_rect.y() - img_rect.y()) / img_rect.height()
                    rel_w = scene_rect.width() / img_rect.width()
                    rel_h = scene_rect.height() / img_rect.height()
                    crop_data.append(('rect', QRectF(rel_x, rel_y, rel_w, rel_h)))
                
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
            remove_action = menu.addAction("Remove Bounding Box")
            action = menu.exec(self.mapToGlobal(position))
            
            if action == remove_action:
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
                    
            # Reset drag state
            self.is_dragging = False
            self.drag_start_pos = None
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)  # Re-enable default drag
            
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        """Handle zoom with mouse wheel."""
        # Zoom in/out
        zoom_factor = 1.15
        if event.angleDelta().y() < 0:
            zoom_factor = 1.0 / zoom_factor
            
        self.scale(zoom_factor, zoom_factor)

class PhotoExtractorApp(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        self.image_processor = ImageProcessor()
        self.current_image_path = None
        self.output_directory = None
        
        self.init_ui()
        
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
        
        toolbar_layout.addStretch()
        main_layout.addLayout(toolbar_layout)
        
        # Create image view
        self.image_view = ImageView()
        self.image_view.add_box_requested.connect(self.add_bounding_box)
        self.image_view.remove_box_requested.connect(self.remove_bounding_box)
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
        
    def load_image(self):
        """Load an image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Load Image", 
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.gif)"
        )
        
        if file_path:
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
        self.update_extract_button_state()
        
    def remove_bounding_box(self, box):
        """Remove the specified bounding box."""
        self.image_view.remove_bounding_box(box)
        self.update_extract_button_state()
        
    def clear_all_boxes(self):
        """Clear all bounding boxes."""
        self.image_view.clear_boxes()
        self.update_extract_button_state()
        
    def update_extract_button_state(self):
        """Enable/disable extract button based on current state."""
        has_image = self.current_image_path is not None
        has_output = self.output_directory is not None
        has_boxes = len(self.image_view.bounding_boxes) > 0
        
        self.extract_btn.setEnabled(has_image and has_output and has_boxes)
        
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