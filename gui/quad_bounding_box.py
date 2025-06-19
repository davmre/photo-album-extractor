"""
Quadrilateral bounding box widget for arbitrary four-sided photo selection.
"""

import uuid

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPen, QPolygonF
from PyQt6.QtWidgets import QGraphicsItem, QGraphicsObject, QGraphicsRectItem

import core.photo_types as photo_types
from core import geometry


class QuadBoundingBox(QGraphicsObject):
    """A quadrilateral bounding box with four draggable corner points."""

    changed = pyqtSignal()
    selected_changed = pyqtSignal(str)  # Emits box_id when selection changes

    def __init__(
        self,
        corners: photo_types.BoundingBoxAny,
        parent=None,
        box_id=None,
        attributes=None,
    ):
        super().__init__(parent)

        # Store the four corner points
        self.corners = photo_types.bounding_box_as_list_of_qpointfs(corners)

        # Unique identifier and attributes
        self.box_id = box_id or str(uuid.uuid4())
        self.attributes = attributes or {}
        self._is_selected = False

        # Visual styling
        self.pen = QPen(QColor(255, 0, 0), 2)
        self.brush = QBrush(QColor(255, 0, 0, 30))
        self.selected_pen = QPen(QColor(0, 150, 255), 3)
        self.selected_brush = QBrush(QColor(0, 150, 255, 50))

        # Interaction flags
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)

        # Corner handles
        self.handles = []
        self.create_handles()

        # Enable mouse interaction for selection
        self.setAcceptHoverEvents(True)

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
        return QRectF(
            min_x - padding,
            min_y - padding,
            max_x - min_x + 2 * padding,
            max_y - min_y + 2 * padding,
        )

    def paint(self, painter, option, widget):
        """Paint the quadrilateral bounding box."""
        # Use selection-specific styling if selected
        if self._is_selected:
            painter.setPen(self.selected_pen)
            painter.setBrush(self.selected_brush)
        else:
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

    def get_corner_points(self) -> photo_types.BoundingBoxQPointF:
        """Get the four corner points in world coordinates."""
        # Always return world coordinates regardless of item position
        world_corners = []
        for corner in self.corners:
            world_corners.append(self.pos() + corner)
        return world_corners

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
            QPointF(min_x, max_y),  # bottom-left
        ]

        # Set position to origin since corners are in absolute coordinates
        self.setPos(0, 0)
        self.update_handles()
        self.update()

    def move_corner(self, corner_id: int, new_position: QPointF):
        """Move a specific corner to a new position (in world coordinates)."""
        if 0 <= corner_id < 4:
            # Convert world coordinates to local coordinates
            local_pos = new_position - self.pos()
            self.corners[corner_id] = local_pos
            self.update_handles()
            self.update()
            self.changed.emit()

    def get_ordered_corners_for_extraction(self) -> photo_types.QuadArray:
        """Get corners in proper order for perspective extraction (prevents flipping)."""
        # Use world coordinates for extraction
        return geometry.sort_clockwise(self.get_corner_points())

    def set_corners(self, new_corners: photo_types.BoundingBoxAny):
        """Set the corners to new world positions."""
        # Reset to origin and store corners as local coordinates
        self.setPos(0, 0)
        self.corners = photo_types.bounding_box_as_list_of_qpointfs(new_corners)
        self.update_handles()
        self.update()
        self.changed.emit()

    def set_selected(self, selected):
        """Set the selection state of this bounding box."""
        if self._is_selected != selected:
            self._is_selected = selected
            self.update()  # Trigger repaint
            if selected:
                self.selected_changed.emit(self.box_id)

    def is_selected(self):
        """Return whether this bounding box is selected."""
        return self._is_selected

    def get_attributes(self):
        """Get the attributes dictionary."""
        return self.attributes.copy()

    def set_attributes(self, attributes):
        """Set the attributes dictionary."""
        self.attributes = attributes.copy()

    def get_attribute(self, key, default=None):
        """Get a specific attribute value."""
        return self.attributes.get(key, default)

    def set_attribute(self, key, value):
        """Set a specific attribute value."""
        self.attributes[key] = value

    def mousePressEvent(self, event):
        """Handle mouse press for selection."""
        if event.button() == Qt.MouseButton.LeftButton:
            # Emit selection signal
            self.selected_changed.emit(self.box_id)
        super().mousePressEvent(event)


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
        self.setRect(-screen_size / 2, -screen_size / 2, screen_size, screen_size)

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
