"""
Magnifier widget for precise cursor positioning in image editing.
"""

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget


class MagnifierWidget(QWidget):
    """Widget that shows a magnified view around the cursor position."""

    def __init__(self, zoom_factor=5, size=150):
        super().__init__()
        self.zoom_factor = zoom_factor
        self.widget_size = size
        self.source_size = size // zoom_factor  # Size of source region to magnify

        # Current state
        self.source_pixmap = None
        self.cursor_pos = QPointF(0, 0)  # Cursor position in image coordinates
        self.image_rect = QRectF()  # Image bounds
        self.bounding_boxes = []  # List of bounding box corners in image coordinates

        # Tracking mode
        self.tracking_mode = "cursor"  # "cursor" or "corner"
        self.focused_corner_pos = QPointF(0, 0)  # Position when focusing on a corner

        # Set up widget
        self.setFixedSize(size, size)
        self.setStyleSheet("""
            QWidget {
                border: 2px solid #ccc;
                background-color: #f0f0f0;
            }
        """)

        # Create layout with title
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Title label
        title_label = QLabel("Magnifier")
        title_label.setStyleSheet("""
            QLabel {
                background-color: #ddd;
                padding: 2px;
                font-size: 10px;
                font-weight: bold;
                text-align: center;
                border: none;
                border-bottom: 1px solid #ccc;
            }
        """)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        # Magnifier display area
        self.magnifier_label = QLabel()
        self.magnifier_label.setMinimumSize(size, size - 20)  # Account for title
        self.magnifier_label.setStyleSheet("border: none;")
        layout.addWidget(self.magnifier_label)

    def set_source_image(self, pixmap):
        """Set the source image to magnify from."""
        self.source_pixmap = pixmap
        if pixmap:
            self.image_rect = QRectF(0, 0, pixmap.width(), pixmap.height())
        self.update_magnifier()

    def set_cursor_position(self, pos):
        """Set the cursor position in image coordinates."""
        self.cursor_pos = QPointF(pos)
        # Switch back to cursor tracking mode when cursor moves
        if self.tracking_mode != "cursor":
            self.tracking_mode = "cursor"
        self.update_magnifier()

    def set_bounding_boxes(self, boxes):
        """Set the list of bounding boxes to overlay."""
        self.bounding_boxes = boxes
        self.update_magnifier()

    def focus_on_corner(self, corner_pos):
        """Focus the magnifier on a specific corner position."""
        self.tracking_mode = "corner"
        # Handle both list [x, y] and QPointF inputs
        if isinstance(corner_pos, list) and len(corner_pos) >= 2:
            self.focused_corner_pos = QPointF(
                float(corner_pos[0]), float(corner_pos[1])
            )
        elif isinstance(corner_pos, QPointF):
            self.focused_corner_pos = corner_pos
        else:
            # If it's something else, create a default point
            self.focused_corner_pos = QPointF(0.0, 0.0)
        self.update_magnifier()

    def resume_cursor_tracking(self):
        """Resume following the cursor."""
        self.tracking_mode = "cursor"
        self.update_magnifier()

    def update_magnifier(self):
        """Update the magnified view."""
        if not self.source_pixmap:
            # Show empty state
            empty_pixmap = QPixmap(self.widget_size, self.widget_size - 20)
            empty_pixmap.fill(QColor(240, 240, 240))
            painter = QPainter(empty_pixmap)
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(
                empty_pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "No Image"
            )
            painter.end()
            self.magnifier_label.setPixmap(empty_pixmap)
            return

        # Calculate source region around the focus point
        focus_pos = (
            self.focused_corner_pos
            if self.tracking_mode == "corner"
            else self.cursor_pos
        )
        half_source = self.source_size / 2
        source_rect = QRectF(
            focus_pos.x() - half_source,
            focus_pos.y() - half_source,
            self.source_size,
            self.source_size,
        )

        # Clamp to image bounds
        source_rect = source_rect.intersected(self.image_rect)

        # Extract and scale the region
        magnified_pixmap = QPixmap(self.widget_size, self.widget_size - 20)
        magnified_pixmap.fill(QColor(255, 255, 255))

        painter = QPainter(magnified_pixmap)
        painter.setRenderHint(
            QPainter.RenderHint.SmoothPixmapTransform, False
        )  # Pixelated zoom

        if not source_rect.isEmpty():
            # Calculate destination rect (centered)
            dest_width = source_rect.width() * self.zoom_factor
            dest_height = source_rect.height() * self.zoom_factor
            dest_x = (self.widget_size - dest_width) / 2
            dest_y = (self.widget_size - 20 - dest_height) / 2
            dest_rect = QRectF(dest_x, dest_y, dest_width, dest_height)

            # Draw magnified image
            painter.drawPixmap(dest_rect, self.source_pixmap, source_rect)

            # Draw bounding box overlays
            self._draw_bounding_box_overlays(painter, source_rect, dest_rect)

            # Draw crosshair at cursor position
            self._draw_crosshair(painter, source_rect, dest_rect)

        painter.end()
        self.magnifier_label.setPixmap(magnified_pixmap)

    def _draw_bounding_box_overlays(self, painter, source_rect, dest_rect):
        """Draw bounding box edges that intersect the magnified region."""
        if not self.bounding_boxes:
            return

        painter.save()

        # Set up pen for bounding box edges
        pen = QPen(QColor(255, 0, 0), 1)
        pen.setCosmetic(True)  # Don't scale with zoom
        painter.setPen(pen)

        # Transform coordinates from source to destination
        scale_x = dest_rect.width() / source_rect.width()
        scale_y = dest_rect.height() / source_rect.height()

        for box_corners in self.bounding_boxes:
            if len(box_corners) >= 4:
                # Draw edges of the quadrilateral
                for i in range(4):
                    p1 = box_corners[i]
                    p2 = box_corners[(i + 1) % 4]

                    # Check if this edge intersects our source region
                    if self._line_intersects_rect(p1, p2, source_rect):
                        # Transform to destination coordinates
                        dest_p1 = QPointF(
                            dest_rect.x() + (p1.x() - source_rect.x()) * scale_x,
                            dest_rect.y() + (p1.y() - source_rect.y()) * scale_y,
                        )
                        dest_p2 = QPointF(
                            dest_rect.x() + (p2.x() - source_rect.x()) * scale_x,
                            dest_rect.y() + (p2.y() - source_rect.y()) * scale_y,
                        )

                        painter.drawLine(dest_p1, dest_p2)

        painter.restore()

    def _draw_crosshair(self, painter, source_rect, dest_rect):
        """Draw crosshair at the focus position."""
        painter.save()

        # Set up pen for crosshair - different colors for different modes
        if self.tracking_mode == "corner":
            pen = QPen(QColor(255, 150, 0), 2)  # Orange for corner focus
        else:
            pen = QPen(QColor(0, 150, 255), 1)  # Blue for cursor tracking
        pen.setCosmetic(True)
        painter.setPen(pen)

        # Calculate focus position in destination coordinates
        focus_pos = (
            self.focused_corner_pos
            if self.tracking_mode == "corner"
            else self.cursor_pos
        )
        if source_rect.contains(focus_pos):
            scale_x = dest_rect.width() / source_rect.width()
            scale_y = dest_rect.height() / source_rect.height()

            focus_dest_x = dest_rect.x() + (focus_pos.x() - source_rect.x()) * scale_x
            focus_dest_y = dest_rect.y() + (focus_pos.y() - source_rect.y()) * scale_y

            # Draw crosshair
            crosshair_size = (
                10 if self.tracking_mode == "cursor" else 15
            )  # Larger for corner focus
            painter.drawLine(
                focus_dest_x - crosshair_size,
                focus_dest_y,
                focus_dest_x + crosshair_size,
                focus_dest_y,
            )
            painter.drawLine(
                focus_dest_x,
                focus_dest_y - crosshair_size,
                focus_dest_x,
                focus_dest_y + crosshair_size,
            )

        painter.restore()

    def _line_intersects_rect(self, p1, p2, rect):
        """Check if a line segment intersects with a rectangle."""
        # Simple bounding box check first
        line_rect = QRectF(
            min(p1.x(), p2.x()),
            min(p1.y(), p2.y()),
            abs(p2.x() - p1.x()),
            abs(p2.y() - p1.y()),
        )
        return rect.intersects(line_rect)

    def set_zoom_factor(self, factor):
        """Change the zoom factor."""
        self.zoom_factor = max(2, min(20, factor))  # Clamp between 2x and 20x
        self.source_size = self.widget_size // self.zoom_factor
        self.update_magnifier()
