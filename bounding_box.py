"""
Draggable and resizable bounding box widget.
"""

from PyQt6.QtWidgets import QWidget, QGraphicsRectItem, QGraphicsItem, QGraphicsObject
from PyQt6.QtCore import Qt, QRectF, pyqtSignal, QObject
from PyQt6.QtGui import QPen, QBrush, QColor, QPainter

class BoundingBox(QGraphicsObject):
    """A draggable and resizable rectangle for selecting photo regions."""
    
    # Signal emitted when box is being moved or resized
    changed = pyqtSignal()
    
    def __init__(self, x, y, width, height, parent=None):
        super().__init__(parent)
        
        # Store rectangle
        self.rect_item = QRectF(x, y, width, height)
        
        # Make the box movable and selectable
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        
        # Visual styling
        self.pen = QPen(QColor(255, 0, 0), 2)
        self.brush = QBrush(QColor(255, 0, 0, 30))
        
        # Resize handles
        self.handles = []
        self.handle_size = 8
        self.create_handles()
        
        # State tracking
        self.resize_handle = None
        self.is_resizing = False
        
    def create_handles(self):
        """Create resize handles at corners and edges."""
        self.handles = []
        # Corner handles: top-left, top-right, bottom-left, bottom-right
        # Edge handles: top, right, bottom, left
        positions = [
            (0, 0), (1, 0), (0, 1), (1, 1),  # corners
            (0.5, 0), (1, 0.5), (0.5, 1), (0, 0.5)  # edges
        ]
        
        for i, (rx, ry) in enumerate(positions):
            handle = ResizeHandle(self, i, rx, ry)
            self.handles.append(handle)
            
    def update_handles(self):
        """Update handle positions based on current rectangle."""
        rect = self.rect_item
        for handle in self.handles:
            handle.update_position(rect)
            
    def itemChange(self, change, value):
        """Handle item changes (movement, etc.)."""
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.update_handles()
            self.changed.emit()
        return super().itemChange(change, value)
        
    def setRect(self, rect):
        """Set rectangle and update handles."""
        self.rect_item = rect
        self.update_handles()
        self.update()
        
    def get_crop_rect(self, image_size):
        """Get the crop rectangle in image coordinates."""
        scene_rect = self.sceneBoundingRect()
        # Convert scene coordinates to image coordinates
        # This will be implemented when we integrate with the image viewer
        return scene_rect
        
    def boundingRect(self):
        """Return the bounding rectangle."""
        return self.rect_item
        
    def rect(self):
        """Return the rectangle."""
        return self.rect_item
        
    def paint(self, painter, option, widget):
        """Paint the bounding box."""
        painter.setPen(self.pen)
        painter.setBrush(self.brush)
        painter.drawRect(self.rect_item)

class ResizeHandle(QGraphicsRectItem):
    """Small squares at corners/edges for resizing bounding boxes."""
    
    def __init__(self, parent_box, handle_id, rel_x, rel_y):
        super().__init__()
        self.parent_box = parent_box
        self.handle_id = handle_id
        self.rel_x = rel_x
        self.rel_y = rel_y
        
        # Make handles ignore transformations so they stay the same screen size
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        
        # Visual styling
        self.setPen(QPen(QColor(255, 0, 0), 2))
        self.setBrush(QBrush(QColor(255, 255, 255)))
        
        # Set size in screen coordinates (will stay constant regardless of zoom)
        screen_size = 12  # 12 pixels on screen
        self.setRect(-screen_size/2, -screen_size/2, screen_size, screen_size)
        
        # Set high z-value to ensure handles are always on top
        self.setZValue(1000)
        
        # Make it movable
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        
        # Set cursor based on handle position
        self.set_cursor()
        
    def set_cursor(self):
        """Set appropriate cursor for resize direction."""
        if self.handle_id in [0, 3]:  # top-left, bottom-right
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif self.handle_id in [1, 2]:  # top-right, bottom-left
            self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        elif self.handle_id in [4, 6]:  # top, bottom
            self.setCursor(Qt.CursorShape.SizeVerCursor)
        elif self.handle_id in [5, 7]:  # right, left
            self.setCursor(Qt.CursorShape.SizeHorCursor)
            
    def update_position(self, rect):
        """Update handle position based on parent rectangle."""
        x = rect.x() + self.rel_x * rect.width()
        y = rect.y() + self.rel_y * rect.height()
        self.setPos(x, y)
        
    def mousePressEvent(self, event):
        """Start resize operation."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.parent_box.is_resizing = True
            self.parent_box.resize_handle = self.handle_id
            self.start_pos = event.scenePos()
            self.start_rect = self.parent_box.rect()
            
    def mouseMoveEvent(self, event):
        """Handle resize dragging."""
        if self.parent_box.is_resizing:
            delta = event.scenePos() - self.start_pos
            new_rect = self.calculate_new_rect(delta)
            self.parent_box.setRect(new_rect)
            
    def mouseReleaseEvent(self, event):
        """End resize operation."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.parent_box.is_resizing = False
            self.parent_box.resize_handle = None
            self.parent_box.changed.emit()
            
    def calculate_new_rect(self, delta):
        """Calculate new rectangle based on handle movement."""
        rect = self.start_rect
        dx, dy = delta.x(), delta.y()
        
        # Handle different resize directions
        if self.handle_id == 0:  # top-left
            new_rect = QRectF(rect.x() + dx, rect.y() + dy, 
                             rect.width() - dx, rect.height() - dy)
        elif self.handle_id == 1:  # top-right
            new_rect = QRectF(rect.x(), rect.y() + dy, 
                             rect.width() + dx, rect.height() - dy)
        elif self.handle_id == 2:  # bottom-left
            new_rect = QRectF(rect.x() + dx, rect.y(), 
                             rect.width() - dx, rect.height() + dy)
        elif self.handle_id == 3:  # bottom-right
            new_rect = QRectF(rect.x(), rect.y(), 
                             rect.width() + dx, rect.height() + dy)
        elif self.handle_id == 4:  # top
            new_rect = QRectF(rect.x(), rect.y() + dy, 
                             rect.width(), rect.height() - dy)
        elif self.handle_id == 5:  # right
            new_rect = QRectF(rect.x(), rect.y(), 
                             rect.width() + dx, rect.height())
        elif self.handle_id == 6:  # bottom
            new_rect = QRectF(rect.x(), rect.y(), 
                             rect.width(), rect.height() + dy)
        elif self.handle_id == 7:  # left
            new_rect = QRectF(rect.x() + dx, rect.y(), 
                             rect.width() - dx, rect.height())
        else:
            new_rect = rect
            
        # Ensure minimum size
        min_size = 10
        if new_rect.width() < min_size:
            new_rect.setWidth(min_size)
        if new_rect.height() < min_size:
            new_rect.setHeight(min_size)
            
        return new_rect