"""
Rotated bounding box widget for non-axis-aligned photo selection.
"""

import math
from PyQt6.QtWidgets import QGraphicsObject, QGraphicsRectItem, QGraphicsItem
from PyQt6.QtCore import Qt, QRectF, pyqtSignal, QPointF
from PyQt6.QtGui import QPen, QBrush, QColor, QPainter, QTransform, QPolygonF

class RotatedBoundingBox(QGraphicsObject):
    """A rotatable bounding box for selecting non-axis-aligned photos."""
    
    changed = pyqtSignal()
    
    def __init__(self, center_x, center_y, width, height, rotation=0, parent=None):
        super().__init__(parent)
        
        # Box properties (use item position as reference, not separate center)
        self.setPos(center_x, center_y)
        self.box_width = width
        self.box_height = height
        self.rotation_angle = rotation  # in degrees
        
        # Visual styling
        self.pen = QPen(QColor(255, 0, 0), 2)
        self.brush = QBrush(QColor(255, 0, 0, 30))
        
        # Interaction flags
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        
        # Resize/rotation handles
        self.handles = []
        self.handle_size = 8
        self.create_handles()
        
        # Interaction state
        self.resize_handle = None
        self.rotation_handle = None
        self.is_resizing = False
        self.is_rotating = False
        
    def create_handles(self):
        """Create resize and rotation handles."""
        self.handles = []
        # Corner resize handles (4)
        for i in range(4):
            handle = RotatedHandle(self, i, 'corner')
            self.handles.append(handle)
            
        # Edge handles (4) - small visible handles
        for i in range(4, 8):
            handle = RotatedHandle(self, i, 'edge')
            self.handles.append(handle)
            
        # Edge lines (4) - invisible draggable lines along each edge
        for i in range(8, 12):
            edge_line = RotatedEdgeLine(self, i - 8)
            self.handles.append(edge_line)
            
        # Rotation handle (1)
        rotation_handle = RotatedHandle(self, 12, 'rotate')
        self.handles.append(rotation_handle)
        
        # Initial handle positioning
        self.update_handles()
        
    def update_handles(self):
        """Update handle positions based on current box state."""
        # Calculate corner and edge positions relative to item position
        half_w = self.box_width / 2
        half_h = self.box_height / 2
        
        # Local positions (before rotation)
        corners = [
            QPointF(-half_w, -half_h),  # 0: top-left
            QPointF(half_w, -half_h),   # 1: top-right
            QPointF(half_w, half_h),    # 2: bottom-right
            QPointF(-half_w, half_h)    # 3: bottom-left
        ]
        
        edges = [
            QPointF(0, -half_h),        # 4: top edge
            QPointF(half_w, 0),         # 5: right edge
            QPointF(0, half_h),         # 6: bottom edge
            QPointF(-half_w, 0)         # 7: left edge
        ]
        
        # Rotate positions around center
        angle_rad = math.radians(self.rotation_angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        center = self.pos()  # Use item position as center
        
        # Update corner handles
        for i, handle in enumerate(self.handles[:4]):
            corner = corners[i]
            x = corner.x() * cos_a - corner.y() * sin_a + center.x()
            y = corner.x() * sin_a + corner.y() * cos_a + center.y()
            handle.setPos(QPointF(x, y))
            
        # Update edge handles
        for i, handle in enumerate(self.handles[4:8]):
            edge = edges[i]
            x = edge.x() * cos_a - edge.y() * sin_a + center.x()
            y = edge.x() * sin_a + edge.y() * cos_a + center.y()
            handle.setPos(QPointF(x, y))
            
        # Update edge lines (invisible draggable lines)
        for i, edge_line in enumerate(self.handles[8:12]):
            edge_line.update_line_geometry()
            
        # Update rotation handle (positioned above the box)
        rotation_offset = QPointF(0, -half_h - 25)
        rot_x = rotation_offset.x() * cos_a - rotation_offset.y() * sin_a + center.x()
        rot_y = rotation_offset.x() * sin_a + rotation_offset.y() * cos_a + center.y()
        self.handles[12].setPos(QPointF(rot_x, rot_y))
        
    def boundingRect(self):
        """Return the bounding rectangle in local coordinates."""
        # Calculate the actual bounds of the rotated rectangle in local space
        half_w = self.box_width / 2
        half_h = self.box_height / 2
        
        corners = [
            QPointF(-half_w, -half_h),
            QPointF(half_w, -half_h),
            QPointF(half_w, half_h),
            QPointF(-half_w, half_h)
        ]
        
        angle_rad = math.radians(self.rotation_angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        
        for corner in corners:
            x = corner.x() * cos_a - corner.y() * sin_a
            y = corner.x() * sin_a + corner.y() * cos_a
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            
        # Add some padding for handles
        padding = 30
        return QRectF(min_x - padding, min_y - padding, 
                     max_x - min_x + 2*padding, max_y - min_y + 2*padding)
        
    def paint(self, painter, option, widget):
        """Paint the rotated bounding box."""
        painter.setPen(self.pen)
        painter.setBrush(self.brush)
        
        # Create polygon for the rotated rectangle in local coordinates
        half_w = self.box_width / 2
        half_h = self.box_height / 2
        
        corners = [
            QPointF(-half_w, -half_h),
            QPointF(half_w, -half_h),
            QPointF(half_w, half_h),
            QPointF(-half_w, half_h)
        ]
        
        angle_rad = math.radians(self.rotation_angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        polygon_points = []
        for corner in corners:
            x = corner.x() * cos_a - corner.y() * sin_a
            y = corner.x() * sin_a + corner.y() * cos_a
            polygon_points.append(QPointF(x, y))
            
        polygon = QPolygonF(polygon_points)
        painter.drawPolygon(polygon)
        
    def itemChange(self, change, value):
        """Handle item changes (movement, etc.)."""
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            # Update handles when position changes
            self.update_handles()
            self.changed.emit()
        return super().itemChange(change, value)
        
    def get_corner_points(self):
        """Get the four corner points of the rotated rectangle in world coordinates."""
        half_w = self.box_width / 2
        half_h = self.box_height / 2
        
        corners = [
            QPointF(-half_w, -half_h),
            QPointF(half_w, -half_h),
            QPointF(half_w, half_h),
            QPointF(-half_w, half_h)
        ]
        
        angle_rad = math.radians(self.rotation_angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        center = self.pos()  # Use item position as center
        world_corners = []
        for corner in corners:
            x = corner.x() * cos_a - corner.y() * sin_a + center.x()
            y = corner.x() * sin_a + corner.y() * cos_a + center.y()
            world_corners.append(QPointF(x, y))
            
        return world_corners
        
    def set_from_drag(self, start_point, end_point):
        """Set box properties from drag operation."""
        # Calculate center, width, height from drag
        center_x = (start_point.x() + end_point.x()) / 2
        center_y = (start_point.y() + end_point.y()) / 2
        self.setPos(center_x, center_y)
        
        self.box_width = abs(end_point.x() - start_point.x())
        self.box_height = abs(end_point.y() - start_point.y())
        self.rotation_angle = 0  # Start with no rotation
        self.update_handles()
        self.update()

class RotatedHandle(QGraphicsRectItem):
    """Handle for resizing and rotating the rotated bounding box."""
    
    def __init__(self, parent_box, handle_id, handle_type):
        super().__init__()
        self.parent_box = parent_box
        self.handle_id = handle_id
        self.handle_type = handle_type  # 'resize' or 'rotate'
        
        # Make handles ignore transformations so they stay the same screen size
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        
        # Visual styling based on handle type
        if handle_type == 'rotate':
            self.setPen(QPen(QColor(0, 255, 0), 2))
            self.setBrush(QBrush(QColor(0, 255, 0, 200)))
            screen_size = 12
        elif handle_type == 'edge':
            self.setPen(QPen(QColor(0, 0, 255), 2))
            self.setBrush(QBrush(QColor(100, 100, 255, 150)))
            screen_size = 8  # Smaller edge handles
        else:  # corner
            self.setPen(QPen(QColor(255, 0, 0), 2))
            self.setBrush(QBrush(QColor(255, 255, 255)))
            screen_size = 12
            
        # Set size in screen coordinates (will stay constant regardless of zoom)
        self.setRect(-screen_size/2, -screen_size/2, screen_size, screen_size)
        
        # Set high z-value to ensure handles are always on top
        self.setZValue(1000)
        
        # Set cursor based on handle type
        if handle_type == 'rotate':
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        elif handle_type == 'edge':
            # Set cursor based on edge direction
            if handle_id in [4, 6]:  # top, bottom edges
                self.setCursor(Qt.CursorShape.SizeVerCursor)
            else:  # left, right edges
                self.setCursor(Qt.CursorShape.SizeHorCursor)
        else:  # corner
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
            
    def mousePressEvent(self, event):
        """Start resize or rotation operation."""
        if event.button() == Qt.MouseButton.LeftButton:
            if self.handle_type == 'resize':
                self.parent_box.is_resizing = True
                self.parent_box.resize_handle = self.handle_id
            else:
                self.parent_box.is_rotating = True
                self.parent_box.rotation_handle = self.handle_id
                
            self.start_pos = event.scenePos()
            self.start_center = QPointF(self.parent_box.pos())
            self.start_width = self.parent_box.box_width
            self.start_height = self.parent_box.box_height
            self.start_rotation = self.parent_box.rotation_angle
            
    def mouseMoveEvent(self, event):
        """Handle resize/rotation dragging."""
        if self.parent_box.is_resizing:
            self.handle_resize(event.scenePos())
        elif self.parent_box.is_rotating:
            self.handle_rotation(event.scenePos())
            
    def mouseReleaseEvent(self, event):
        """End resize/rotation operation."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.parent_box.is_resizing = False
            self.parent_box.is_rotating = False
            self.parent_box.resize_handle = None
            self.parent_box.rotation_handle = None
            self.parent_box.changed.emit()
            
    def handle_resize(self, current_pos):
        """Handle resize operation by moving only the edges connected to this handle."""
        delta = current_pos - self.start_pos
        
        # Transform delta to local coordinates considering rotation
        angle_rad = math.radians(self.parent_box.rotation_angle)
        cos_a = math.cos(-angle_rad)  # Negative to reverse the rotation
        sin_a = math.sin(-angle_rad)
        
        # Rotate delta vector to local box coordinates
        local_dx = delta.x() * cos_a - delta.y() * sin_a
        local_dy = delta.x() * sin_a + delta.y() * cos_a
        
        # Get starting values
        center = self.start_center
        new_width = self.start_width
        new_height = self.start_height
        new_center_x = center.x()
        new_center_y = center.y()
        
        # Handle corner resizing - each corner affects two edges
        if self.handle_id == 0:  # top-left corner
            new_height = max(20, self.start_height - local_dy)
            new_width = max(20, self.start_width - local_dx)
            # Center shift in world coordinates
            center_shift_x = (-local_dx / 2) * cos_a - (-local_dy / 2) * sin_a
            center_shift_y = (-local_dx / 2) * sin_a + (-local_dy / 2) * cos_a
            new_center_x = center.x() + center_shift_x
            new_center_y = center.y() + center_shift_y
            
        elif self.handle_id == 1:  # top-right corner
            new_height = max(20, self.start_height - local_dy)
            new_width = max(20, self.start_width + local_dx)
            center_shift_x = (local_dx / 2) * cos_a - (-local_dy / 2) * sin_a
            center_shift_y = (local_dx / 2) * sin_a + (-local_dy / 2) * cos_a
            new_center_x = center.x() + center_shift_x
            new_center_y = center.y() + center_shift_y
            
        elif self.handle_id == 2:  # bottom-right corner
            new_height = max(20, self.start_height + local_dy)
            new_width = max(20, self.start_width + local_dx)
            center_shift_x = (local_dx / 2) * cos_a - (local_dy / 2) * sin_a
            center_shift_y = (local_dx / 2) * sin_a + (local_dy / 2) * cos_a
            new_center_x = center.x() + center_shift_x
            new_center_y = center.y() + center_shift_y
            
        elif self.handle_id == 3:  # bottom-left corner
            new_height = max(20, self.start_height + local_dy)
            new_width = max(20, self.start_width - local_dx)
            center_shift_x = (-local_dx / 2) * cos_a - (local_dy / 2) * sin_a
            center_shift_y = (-local_dx / 2) * sin_a + (local_dy / 2) * cos_a
            new_center_x = center.x() + center_shift_x
            new_center_y = center.y() + center_shift_y
            
        # Handle edge resizing - each edge affects only one dimension
        elif self.handle_id == 4:  # top edge (handle index 4)
            new_height = max(20, self.start_height - local_dy)
            height_change = new_height - self.start_height
            # Center shifts in local Y direction
            center_shift_local_x = 0
            center_shift_local_y = height_change / 2
            
        elif self.handle_id == 5:  # right edge (handle index 5)
            new_width = max(20, self.start_width + local_dx)
            width_change = new_width - self.start_width
            # Center shifts in local X direction
            center_shift_local_x = width_change / 2
            center_shift_local_y = 0
            
        elif self.handle_id == 6:  # bottom edge (handle index 6)
            new_height = max(20, self.start_height + local_dy)
            height_change = new_height - self.start_height
            # Center shifts in local Y direction
            center_shift_local_x = 0
            center_shift_local_y = height_change / 2
            
        elif self.handle_id == 7:  # left edge (handle index 7)
            new_width = max(20, self.start_width - local_dx)
            width_change = new_width - self.start_width
            # Center shifts in local X direction  
            center_shift_local_x = width_change / 2
            center_shift_local_y = 0
            
        # Transform center shift back to world coordinates for edge handles
        if self.handle_id >= 4:  # Edge handles
            rev_cos_a = math.cos(angle_rad)
            rev_sin_a = math.sin(angle_rad)
            center_shift_x = center_shift_local_x * rev_cos_a - center_shift_local_y * rev_sin_a
            center_shift_y = center_shift_local_x * rev_sin_a + center_shift_local_y * rev_cos_a
            new_center_x = center.x() + center_shift_x
            new_center_y = center.y() + center_shift_y
        
        # Apply the changes (keep original rotation)
        self.parent_box.box_width = new_width
        self.parent_box.box_height = new_height
        self.parent_box.setPos(new_center_x, new_center_y)
        # Don't change rotation_angle - it should stay the same
        self.parent_box.update_handles()
        self.parent_box.update()
        
    def handle_rotation(self, current_pos):
        """Handle rotation operation."""
        # Calculate angle from center to current mouse position
        center = self.parent_box.pos()
        angle = math.degrees(math.atan2(current_pos.y() - center.y(), 
                                      current_pos.x() - center.x()))
        
        # Adjust angle (rotation handle is above the box)
        angle = (angle + 90) % 360
        
        self.parent_box.rotation_angle = angle
        self.parent_box.update_handles()
        self.parent_box.update()

class RotatedEdgeLine(QGraphicsRectItem):
    """Invisible draggable line along the edge of a rotated bounding box."""
    
    def __init__(self, parent_box, edge_id):
        super().__init__()
        self.parent_box = parent_box
        self.edge_id = edge_id  # 0=top, 1=right, 2=bottom, 3=left
        
        # Make invisible but still clickable
        self.setPen(QPen(QColor(0, 0, 0, 0)))  # Transparent
        self.setBrush(QBrush(QColor(0, 0, 0, 0)))  # Transparent
        
        # Set high z-value but lower than handles
        self.setZValue(999)
        
        # Set cursor based on edge direction
        if edge_id in [0, 2]:  # top, bottom edges
            self.setCursor(Qt.CursorShape.SizeVerCursor)
        else:  # left, right edges
            self.setCursor(Qt.CursorShape.SizeHorCursor)
            
    def update_line_geometry(self):
        """Update the line geometry to match the edge."""
        half_w = self.parent_box.box_width / 2
        half_h = self.parent_box.box_height / 2
        
        # Define edge endpoints in local coordinates
        if self.edge_id == 0:  # top edge
            start = QPointF(-half_w, -half_h)
            end = QPointF(half_w, -half_h)
        elif self.edge_id == 1:  # right edge
            start = QPointF(half_w, -half_h)
            end = QPointF(half_w, half_h)
        elif self.edge_id == 2:  # bottom edge
            start = QPointF(half_w, half_h)
            end = QPointF(-half_w, half_h)
        else:  # left edge
            start = QPointF(-half_w, half_h)
            end = QPointF(-half_w, -half_h)
            
        # Transform to world coordinates
        angle_rad = math.radians(self.parent_box.rotation_angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        center = self.parent_box.pos()
        
        world_start_x = start.x() * cos_a - start.y() * sin_a + center.x()
        world_start_y = start.x() * sin_a + start.y() * cos_a + center.y()
        world_end_x = end.x() * cos_a - end.y() * sin_a + center.x()
        world_end_y = end.x() * sin_a + end.y() * cos_a + center.y()
        
        # Create a thick invisible rectangle along the edge
        thickness = 10  # 10 pixel wide invisible hit area
        
        # Calculate rectangle that encompasses the edge with thickness
        min_x = min(world_start_x, world_end_x) - thickness/2
        max_x = max(world_start_x, world_end_x) + thickness/2
        min_y = min(world_start_y, world_end_y) - thickness/2
        max_y = max(world_start_y, world_end_y) + thickness/2
        
        # Set the rectangle
        self.setRect(min_x, min_y, max_x - min_x, max_y - min_y)
        
    def mousePressEvent(self, event):
        """Start edge resize operation."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.parent_box.is_resizing = True
            self.parent_box.resize_handle = self.edge_id + 4  # Map to edge handle IDs
            
            self.start_pos = event.scenePos()
            self.start_center = QPointF(self.parent_box.pos())
            self.start_width = self.parent_box.box_width
            self.start_height = self.parent_box.box_height
            self.start_rotation = self.parent_box.rotation_angle
            
    def mouseMoveEvent(self, event):
        """Handle edge resize dragging."""
        if self.parent_box.is_resizing:
            self.handle_resize(event.scenePos())
            
    def mouseReleaseEvent(self, event):
        """End edge resize operation."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.parent_box.is_resizing = False
            self.parent_box.resize_handle = None
            self.parent_box.changed.emit()
            
    def handle_resize(self, current_pos):
        """Handle resize using the same logic as edge handles."""
        delta = current_pos - self.start_pos
        
        # Transform delta to local coordinates considering rotation
        angle_rad = math.radians(self.parent_box.rotation_angle)
        cos_a = math.cos(-angle_rad)  # Negative to reverse the rotation
        sin_a = math.sin(-angle_rad)
        
        # Rotate delta vector to local box coordinates
        local_dx = delta.x() * cos_a - delta.y() * sin_a
        local_dy = delta.x() * sin_a + delta.y() * cos_a
        
        # Get starting values
        center = self.start_center
        new_width = self.start_width
        new_height = self.start_height
        
        # Calculate new dimensions based on which edge is being dragged
        # For edge dragging, we change only one dimension and move the center accordingly
        
        if self.edge_id == 0:  # top edge - moving top edge up/down
            # Moving top edge up decreases height, moving down increases height  
            # In local coordinates: negative local_dy = moving edge away from center (up) = smaller height
            new_height = max(20, self.start_height - local_dy)
            height_change = new_height - self.start_height
            
        elif self.edge_id == 1:  # right edge - moving right edge left/right
            # Moving right edge right increases width, moving left decreases width
            # In local coordinates: positive local_dx = moving edge away from center (right) = larger width
            new_width = max(20, self.start_width + local_dx)
            width_change = new_width - self.start_width
            
        elif self.edge_id == 2:  # bottom edge - moving bottom edge up/down
            # Moving bottom edge down increases height, moving up decreases height
            # In local coordinates: positive local_dy = moving edge away from center (down) = larger height  
            new_height = max(20, self.start_height + local_dy)
            height_change = new_height - self.start_height
            
        elif self.edge_id == 3:  # left edge - moving left edge left/right
            # Moving left edge left increases width, moving right decreases width
            # In local coordinates: negative local_dx = moving edge away from center (left) = larger width
            new_width = max(20, self.start_width - local_dx)
            width_change = new_width - self.start_width
        
        # Calculate center shift in world coordinates
        # The center moves by half the dimension change in the direction of the moved edge
        rev_cos_a = math.cos(angle_rad)
        rev_sin_a = math.sin(angle_rad)
        
        if self.edge_id == 0:  # top edge
            # Center shifts in the direction of the top edge normal (negative Y in local space)
            center_shift_local_x = 0
            center_shift_local_y = height_change / 2
        elif self.edge_id == 1:  # right edge  
            # Center shifts in the direction of the right edge normal (positive X in local space)
            center_shift_local_x = width_change / 2
            center_shift_local_y = 0
        elif self.edge_id == 2:  # bottom edge
            # Center shifts in the direction of the bottom edge normal (positive Y in local space)
            center_shift_local_x = 0
            center_shift_local_y = height_change / 2
        elif self.edge_id == 3:  # left edge
            # Center shifts in the direction of the left edge normal (negative X in local space)
            center_shift_local_x = width_change / 2
            center_shift_local_y = 0
            
        # Transform center shift back to world coordinates
        center_shift_x = center_shift_local_x * rev_cos_a - center_shift_local_y * rev_sin_a
        center_shift_y = center_shift_local_x * rev_sin_a + center_shift_local_y * rev_cos_a
        
        new_center_x = center.x() + center_shift_x
        new_center_y = center.y() + center_shift_y
        
        # Apply the changes (keep original rotation)
        self.parent_box.box_width = new_width
        self.parent_box.box_height = new_height
        self.parent_box.setPos(new_center_x, new_center_y)
        self.parent_box.update_handles()
        self.parent_box.update()