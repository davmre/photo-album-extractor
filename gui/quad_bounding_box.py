"""
Quadrilateral bounding box widget for arbitrary four-sided photo selection.
"""

from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QKeyEvent, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import (
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsRectItem,
    QGraphicsSceneMouseEvent,
)

import core.photo_types as photo_types
from core import geometry
from core.bounding_box_data import (
    BoundingBoxData,
    PhotoAttributes,
    Severity,
    ValidationIssue,
)


class QuadBoundingBox(QGraphicsObject):
    """A quadrilateral bounding box with four draggable corner points."""

    changed = pyqtSignal()
    selected_changed = pyqtSignal(str)  # Emits box_id when selection changes

    def __init__(self, box_data: BoundingBoxData, parent=None):
        super().__init__(parent)

        # Store the four corner points, unique id, and attributes
        self.corners = photo_types.bounding_box_as_list_of_qpointfs(box_data.corners)
        self.box_id = box_data.box_id
        self.attributes = box_data.attributes
        self.marked_as_good = box_data.marked_as_good

        # Initialize keep_rectangular based on whether the box is currently a rectangle
        self.keep_rectangular = box_data.is_rectangle()

        self._is_selected = False

        # Validation state
        self.validation_issues: list[ValidationIssue] = []
        self._update_validation()

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

    def paint(self, painter: QPainter, option, widget):
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

        # Draw validation icons if there are issues or if marked as good
        if self.validation_issues or self.marked_as_good:
            self._draw_validation_icons(painter)

    def _draw_validation_icons(self, painter: QPainter):
        """Draw validation issue icons and/or marked as good checkmark on the bounding box."""
        # Find the center of the bounding box for overlay positioning
        center_x = sum(corner.x() for corner in self.corners) / 4
        center_y = sum(corner.y() for corner in self.corners) / 4

        # Collect all icons to draw
        icons_to_draw = []

        # Add validation issue icons
        if self.validation_issues:
            # Sort issues by severity (errors first)
            sorted_issues = sorted(
                self.validation_issues,
                key=lambda x: x.severity == Severity.ERROR,
                reverse=True,
            )

            for issue in sorted_issues:
                if issue.severity == Severity.ERROR:
                    icons_to_draw.append("🚨")
                else:
                    icons_to_draw.append("⚠️")

        # Add green checkmark if marked as good
        if self.marked_as_good:
            icons_to_draw.append("✅")

        if not icons_to_draw:
            return

        # Icon settings - much larger for better visibility
        width, height = geometry.dimension_bounds(self.corners)
        icon_scale = min(width, height)
        icon_size = int(icon_scale / 5.0)
        icon_spacing = icon_scale / 20.0

        # Set up font for emoji icons
        font = QFont()
        font.setPixelSize(icon_size)
        painter.setFont(font)

        # Calculate total width needed for all icons
        total_width = (
            len(icons_to_draw) * icon_size + (len(icons_to_draw) - 1) * icon_spacing
        )
        start_x = center_x - total_width // 2

        # Draw icons centered horizontally on the bounding box
        x_offset = 0
        for icon in icons_to_draw:
            # Calculate icon position centered on the bounding box
            icon_x = int(start_x + x_offset)
            icon_y = int(center_y - icon_size // 2)

            # Draw the icon
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            text_x = icon_x + icon_size // 4  # Adjust for emoji positioning
            text_y = icon_y + icon_size * 3 // 4  # Adjust for baseline
            painter.drawText(QPointF(text_x, text_y), icon)

            x_offset += icon_size + icon_spacing

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

    def is_axis_aligned_rect(self):
        unique_x_values = {c.x() for c in self.corners}
        unique_y_values = {c.y() for c in self.corners}
        return len(unique_x_values) == 2 and len(unique_y_values) == 2

    def corner_dragged(
        self, corner_id: int, new_position: QPointF, override_rectangular: bool = False
    ):
        local_pos = new_position - self.pos()

        # Check if we should override rectangular mode
        if override_rectangular and self.keep_rectangular:
            self.keep_rectangular = False

        if self.keep_rectangular:
            # Rectangle-preserving drag: work like axis-aligned dragging but in rotated frame
            opposite_id = (corner_id + 2) % 4
            adj1_id = (corner_id + 1) % 4
            adj2_id = (corner_id + 3) % 4

            opposite_corner = self.corners[opposite_id]

            # Get the two edge vectors from the dragged corner to its adjacent corners
            edge1 = (
                self.corners[adj1_id] - self.corners[corner_id]
            )  # edge to next corner
            edge2 = (
                self.corners[adj2_id] - self.corners[corner_id]
            )  # edge to prev corner

            # Normalize edge vectors to get local coordinate system
            edge1_len = (edge1.x() ** 2 + edge1.y() ** 2) ** 0.5
            edge2_len = (edge2.x() ** 2 + edge2.y() ** 2) ** 0.5

            if edge1_len > 1e-10 and edge2_len > 1e-10:  # Avoid division by zero
                edge1_unit = QPointF(edge1.x() / edge1_len, edge1.y() / edge1_len)
                edge2_unit = QPointF(edge2.x() / edge2_len, edge2.y() / edge2_len)

                # Vector from opposite corner to new drag position
                drag_vector = local_pos - opposite_corner

                # Project drag vector onto the two edge directions
                proj1 = (
                    drag_vector.x() * edge1_unit.x() + drag_vector.y() * edge1_unit.y()
                )
                proj2 = (
                    drag_vector.x() * edge2_unit.x() + drag_vector.y() * edge2_unit.y()
                )

                # Reconstruct rectangle using projected lengths
                self.corners[corner_id] = opposite_corner + QPointF(
                    proj1 * edge1_unit.x() + proj2 * edge2_unit.x(),
                    proj1 * edge1_unit.y() + proj2 * edge2_unit.y(),
                )
                self.corners[adj1_id] = opposite_corner + QPointF(
                    proj1 * edge1_unit.x(), proj1 * edge1_unit.y()
                )
                self.corners[adj2_id] = opposite_corner + QPointF(
                    proj2 * edge2_unit.x(), proj2 * edge2_unit.y()
                )
                # opposite_corner stays unchanged
            else:
                # Fallback to free-form if edges are degenerate
                self.corners[corner_id] = local_pos
        else:
            # Free-form quadrilateral: only update the dragged corner
            self.corners[corner_id] = local_pos
            self._update_validation()

        self.update_handles()
        self.update()
        self._update_validation()
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
        self._update_validation()
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

    def get_attributes(self) -> PhotoAttributes:
        """Get the photo attributes."""
        return self.attributes

    def set_attributes(self, attributes: PhotoAttributes):
        """Set the photo attributes."""
        self.attributes = attributes

    def get_bounding_box_data(self) -> BoundingBoxData:
        """Get the complete bounding box data (corners + attributes)."""
        corners_array = self.get_ordered_corners_for_extraction()
        return BoundingBoxData(
            box_id=self.box_id,
            corners=corners_array,
            attributes=self.attributes,
            marked_as_good=self.marked_as_good,
        )

    def set_bounding_box_data(self, data: BoundingBoxData):
        """Set the complete bounding box data (corners + attributes)."""
        if self.box_id != data.box_id:
            raise ValueError("box id doesn't match!")
        self.set_corners(data.corners)
        self.attributes = data.attributes
        self.marked_as_good = data.marked_as_good
        self._update_validation()

    def set_marked_as_good(self, marked_as_good: bool):
        """Set the marked as good status."""
        self.marked_as_good = marked_as_good
        self._update_validation()
        self.update()  # Trigger repaint to update validation icons
        self.changed.emit()

    def _update_validation(self):
        """Update validation state and trigger repaint if needed."""
        old_issues = self.validation_issues
        box_data = self.get_bounding_box_data()
        self.validation_issues = box_data.validate()

        # Trigger repaint if validation state changed
        if old_issues != self.validation_issues:
            self.update()

    def update_validation(self):
        """Public method to update validation state."""
        self._update_validation()

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        """Handle mouse press for selection."""
        if event.button() == Qt.MouseButton.LeftButton:
            # Emit selection signal
            self.selected_changed.emit(self.box_id)
        super().mousePressEvent(event)

    def keyPressEvent(self, event: QKeyEvent | None) -> None:
        print("BBox keypress")
        return super().keyPressEvent(event)


class CornerHandle(QGraphicsRectItem):
    """Draggable handle for corner points of quadrilateral bounding box."""

    def __init__(self, parent_box: QuadBoundingBox, corner_id: int):
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

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        """Start corner drag operation."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = True
            # Don't call super() to prevent default item movement

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
        """Handle corner dragging."""
        if self.is_dragging:
            # Check if Command key is held (Meta on macOS, Ctrl on Windows/Linux)
            modifiers = event.modifiers()
            override_rectangular = bool(
                modifiers
                & (
                    Qt.KeyboardModifier.MetaModifier
                    | Qt.KeyboardModifier.ControlModifier
                )
            )

            # Move the corner to the new position
            new_pos = event.scenePos()
            self.parent_box.corner_dragged(
                self.corner_id, new_pos, override_rectangular
            )
            # Update our position to follow the corner
            self.setPos(new_pos)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        """End corner drag operation."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = False
