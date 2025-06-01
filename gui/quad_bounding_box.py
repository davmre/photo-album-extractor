"""
Quadrilateral bounding box widget for arbitrary four-sided photo selection.
"""

import math
from PyQt6.QtWidgets import QGraphicsObject, QGraphicsRectItem, QGraphicsItem
from PyQt6.QtCore import Qt, QRectF, pyqtSignal, QPointF
from PyQt6.QtGui import QPen, QBrush, QColor, QPainter, QPolygonF

class QuadBoundingBox(QGraphicsObject):
    """A quadrilateral bounding box with four draggable corner points."""
    
    changed = pyqtSignal()
    
    def __init__(self, corners, parent=None):
        super().__init__(parent)
        
        # Store the four corner points
        self.corners = [QPointF(corner) for corner in corners]
        
        # Visual styling
        self.pen = QPen(QColor(255, 0, 0), 2)
        self.brush = QBrush(QColor(255, 0, 0, 30))
        
        # Interaction flags
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        
        # Corner handles
        self.handles = []
        self.create_handles()
        
    def create_handles(self):
        """Create corner handles."""
        self.handles = []
        for i in range(4):
            handle = CornerHandle(self, i)
            self.handles.append(handle)
            
        # Initial handle positioning
        self.update_handles()
        
    def update_handles(self):
        """Update handle positions to match corner points."""
        for i, handle in enumerate(self.handles):
            # Position handles relative to this item's coordinate system
            world_pos = self.pos() + self.corners[i]
            handle.setPos(world_pos)
            
    def boundingRect(self):
        """Return the bounding rectangle that encompasses all corners."""
        # Find min/max coordinates
        min_x = min(corner.x() for corner in self.corners)
        max_x = max(corner.x() for corner in self.corners)
        min_y = min(corner.y() for corner in self.corners)
        max_y = max(corner.y() for corner in self.corners)
        
        # Add padding for handles
        padding = 30
        return QRectF(min_x - padding, min_y - padding, 
                     max_x - min_x + 2*padding, max_y - min_y + 2*padding)
        
    def paint(self, painter, option, widget):
        """Paint the quadrilateral bounding box."""
        painter.setPen(self.pen)
        painter.setBrush(self.brush)
        
        # Create polygon from corners and draw
        polygon = QPolygonF(self.corners)
        painter.drawPolygon(polygon)
        
    def itemChange(self, change, value):
        """Handle item changes (movement, etc.)."""
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            # When the box moves, update corners to maintain world coordinates
            # and update handle positions
            new_pos = value
            old_pos = self.pos()
            delta = new_pos - old_pos
            
            # Update corners to new world positions
            for i in range(len(self.corners)):
                self.corners[i] += delta
                
            # Update handles
            self.update_handles()
            self.changed.emit()
        return super().itemChange(change, value)
        
    def get_corner_points(self):
        """Get the four corner points in world coordinates."""
        # Always return world coordinates regardless of item position
        world_corners = []
        for corner in self.corners:
            world_corners.append(self.pos() + corner)
        return world_corners
        
    def get_corner_points_for_extraction(self):
        """Alias for get_ordered_corners_for_extraction for backward compatibility."""
        return self.get_ordered_corners_for_extraction()
        
    def set_from_drag(self, start_point, end_point):
        """Set initial rectangle from drag operation."""
        # Create a rectangle from drag points
        min_x = min(start_point.x(), end_point.x())
        max_x = max(start_point.x(), end_point.x())
        min_y = min(start_point.y(), end_point.y())
        max_y = max(start_point.y(), end_point.y())
        
        # Set corners in clockwise order: top-left, top-right, bottom-right, bottom-left
        self.corners = [
            QPointF(min_x, min_y),  # top-left
            QPointF(max_x, min_y),  # top-right
            QPointF(max_x, max_y),  # bottom-right
            QPointF(min_x, max_y)   # bottom-left
        ]
        
        # Set position to origin since corners are in absolute coordinates
        self.setPos(0, 0)
        self.update_handles()
        self.update()
        
    def move_corner(self, corner_id, new_position):
        """Move a specific corner to a new position (in world coordinates)."""
        if 0 <= corner_id < 4:
            # Convert world coordinates to local coordinates
            local_pos = new_position - self.pos()
            self.corners[corner_id] = local_pos
            self.update_handles()
            self.update()
            self.changed.emit()
            
    def get_ordered_corners_for_extraction(self):
        """Get corners in proper order for perspective extraction (prevents flipping)."""
        # Use world coordinates for extraction
        world_corners = self.get_corner_points()
        
        # Calculate centroid
        centroid_x = sum(corner.x() for corner in world_corners) / 4
        centroid_y = sum(corner.y() for corner in world_corners) / 4
        centroid = QPointF(centroid_x, centroid_y)
        
        # Create list of (angle, corner) tuples
        corner_data = []
        for corner in world_corners:
            # Calculate angle from centroid to corner
            dx = corner.x() - centroid.x()
            dy = corner.y() - centroid.y()
            angle = math.atan2(dy, dx)
            corner_data.append((angle, corner))
        
        # Sort by angle to get proper clockwise ordering
        corner_data.sort(key=lambda x: x[0])
        
        # Return just the corners in correct order
        return [data[1] for data in corner_data]
        
    def set_corners(self, new_corners):
        """Set the corners to new world positions."""
        # Reset to origin and store corners as local coordinates
        self.setPos(0, 0)
        self.corners = [QPointF(corner) for corner in new_corners]
        self.update_handles()
        self.update()
        self.changed.emit()

class CornerHandle(QGraphicsRectItem):
    """Draggable handle for corner points of quadrilateral bounding box."""
    
    def __init__(self, parent_box, corner_id):
        super().__init__()
        self.parent_box = parent_box
        self.corner_id = corner_id
        
        # Make handles ignore transformations so they stay the same screen size
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        
        # Visual styling
        self.setPen(QPen(QColor(255, 0, 0), 2))
        self.setBrush(QBrush(QColor(255, 255, 255)))
        
        # Set size in screen coordinates
        screen_size = 12
        self.setRect(-screen_size/2, -screen_size/2, screen_size, screen_size)
        
        # Set high z-value to ensure handles are always on top
        self.setZValue(1000)
        
        # Set cursor
        self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        
        # Dragging state
        self.is_dragging = False
        
    def mousePressEvent(self, event):
        """Start corner drag operation."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = True
            # Don't call super() to prevent default item movement
            
    def mouseMoveEvent(self, event):
        """Handle corner dragging."""
        if self.is_dragging:
            # Move the corner to the new position
            new_pos = event.scenePos()
            self.parent_box.move_corner(self.corner_id, new_pos)
            # Update our position to follow the corner
            self.setPos(new_pos)
            
    def mouseReleaseEvent(self, event):
        """End corner drag operation."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = False
            
class QuadEdgeLine(QGraphicsRectItem):
    """Invisible draggable line along an edge of the quadrilateral for moving two corners together."""
    
    def __init__(self, parent_box, edge_id):
        super().__init__()
        self.parent_box = parent_box
        self.edge_id = edge_id  # 0=top, 1=right, 2=bottom, 3=left
        
        # Make invisible but still clickable
        self.setPen(QPen(QColor(0, 0, 0, 0)))  # Transparent
        self.setBrush(QBrush(QColor(0, 0, 0, 0)))  # Transparent
        
        # Set z-value below handles but above box
        self.setZValue(999)
        
        # Set cursor based on edge orientation
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        
        # Dragging state
        self.is_dragging = False
        self.drag_start_pos = None
        self.start_corners = []
        
    def update_edge_geometry(self):
        """Update the edge geometry to match the quadrilateral edge."""
        corners = self.parent_box.get_corner_points()  # Get world coordinates
        
        # Get the two corners that define this edge
        corner1 = corners[self.edge_id]
        corner2 = corners[(self.edge_id + 1) % 4]
        
        # Create a thick invisible rectangle along the edge
        thickness = 15  # 15 pixel wide invisible hit area
        
        # Calculate the edge direction and perpendicular
        edge_vector = QPointF(corner2.x() - corner1.x(), corner2.y() - corner1.y())
        edge_length = (edge_vector.x()**2 + edge_vector.y()**2)**0.5
        
        if edge_length < 1:  # Avoid division by zero
            return
            
        # Normalize edge vector
        edge_unit = QPointF(edge_vector.x() / edge_length, edge_vector.y() / edge_length)
        
        # Perpendicular vector
        perp_unit = QPointF(-edge_unit.y(), edge_unit.x())
        
        # Create rectangle along the edge
        half_thickness = thickness / 2
        
        # Calculate the four corners of the hit area rectangle
        p1 = QPointF(corner1.x() - perp_unit.x() * half_thickness,
                     corner1.y() - perp_unit.y() * half_thickness)
        p2 = QPointF(corner1.x() + perp_unit.x() * half_thickness,
                     corner1.y() + perp_unit.y() * half_thickness)
        p3 = QPointF(corner2.x() + perp_unit.x() * half_thickness,
                     corner2.y() + perp_unit.y() * half_thickness)
        p4 = QPointF(corner2.x() - perp_unit.x() * half_thickness,
                     corner2.y() - perp_unit.y() * half_thickness)
        
        # Find bounding rectangle
        min_x = min(p1.x(), p2.x(), p3.x(), p4.x())
        max_x = max(p1.x(), p2.x(), p3.x(), p4.x())
        min_y = min(p1.y(), p2.y(), p3.y(), p4.y())
        max_y = max(p1.y(), p2.y(), p3.y(), p4.y())
        
        self.setRect(min_x, min_y, max_x - min_x, max_y - min_y)
        
    def mousePressEvent(self, event):
        """Start edge drag operation."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = True
            self.drag_start_pos = event.scenePos()
            # Store the starting positions of the two corners that define this edge
            corners = self.parent_box.get_corner_points()
            self.start_corners = [QPointF(corner) for corner in corners]
            
    def mouseMoveEvent(self, event):
        """Handle edge dragging by moving both edge corners together."""
        if self.is_dragging and self.drag_start_pos:
            delta = event.scenePos() - self.drag_start_pos
            
            # Move the two corners that define this edge
            corner1_id = self.edge_id
            corner2_id = (self.edge_id + 1) % 4
            
            # start_corners contains world coordinates
            new_corner1 = self.start_corners[corner1_id] + delta
            new_corner2 = self.start_corners[corner2_id] + delta
            
            self.parent_box.move_corner(corner1_id, new_corner1)
            self.parent_box.move_corner(corner2_id, new_corner2)
            
            # Update our geometry
            self.update_edge_geometry()
            
    def mouseReleaseEvent(self, event):
        """End edge drag operation."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = False
            self.drag_start_pos = None
            self.start_corners = []