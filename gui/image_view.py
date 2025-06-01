"""
Custom graphics view for displaying images with bounding box interaction.
"""

import numpy as np
from PyQt6.QtWidgets import (QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, 
                             QMenu, QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal, QPointF
from PyQt6.QtGui import QPainter

from quad_bounding_box import QuadBoundingBox, QuadEdgeLine
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