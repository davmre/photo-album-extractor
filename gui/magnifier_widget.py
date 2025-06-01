"""
Magnifier widget for precise cursor positioning in image editing.
"""

import math
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QColor, QPixmap, QBrush


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
        self.update_magnifier()
        
    def set_bounding_boxes(self, boxes):
        """Set the list of bounding boxes to overlay."""
        self.bounding_boxes = boxes
        self.update_magnifier()
        
    def update_magnifier(self):
        """Update the magnified view."""
        if not self.source_pixmap:
            # Show empty state
            empty_pixmap = QPixmap(self.widget_size, self.widget_size - 20)
            empty_pixmap.fill(QColor(240, 240, 240))
            painter = QPainter(empty_pixmap)
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(empty_pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "No Image")
            painter.end()
            self.magnifier_label.setPixmap(empty_pixmap)
            return
            
        # Calculate source region around cursor
        half_source = self.source_size / 2
        source_rect = QRectF(
            self.cursor_pos.x() - half_source,
            self.cursor_pos.y() - half_source,
            self.source_size,
            self.source_size
        )
        
        # Clamp to image bounds
        source_rect = source_rect.intersected(self.image_rect)
        
        # Extract and scale the region
        magnified_pixmap = QPixmap(self.widget_size, self.widget_size - 20)
        magnified_pixmap.fill(QColor(255, 255, 255))
        
        painter = QPainter(magnified_pixmap)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)  # Pixelated zoom
        
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
                            dest_rect.y() + (p1.y() - source_rect.y()) * scale_y
                        )
                        dest_p2 = QPointF(
                            dest_rect.x() + (p2.x() - source_rect.x()) * scale_x,
                            dest_rect.y() + (p2.y() - source_rect.y()) * scale_y
                        )
                        
                        painter.drawLine(dest_p1, dest_p2)
        
        painter.restore()
        
    def _draw_crosshair(self, painter, source_rect, dest_rect):
        """Draw crosshair at the cursor position."""
        painter.save()
        
        # Set up pen for crosshair
        pen = QPen(QColor(0, 150, 255), 1)
        pen.setCosmetic(True)
        painter.setPen(pen)
        
        # Calculate cursor position in destination coordinates
        if source_rect.contains(self.cursor_pos):
            scale_x = dest_rect.width() / source_rect.width()
            scale_y = dest_rect.height() / source_rect.height()
            
            cursor_dest_x = dest_rect.x() + (self.cursor_pos.x() - source_rect.x()) * scale_x
            cursor_dest_y = dest_rect.y() + (self.cursor_pos.y() - source_rect.y()) * scale_y
            
            # Draw crosshair
            crosshair_size = 10
            painter.drawLine(
                cursor_dest_x - crosshair_size, cursor_dest_y,
                cursor_dest_x + crosshair_size, cursor_dest_y
            )
            painter.drawLine(
                cursor_dest_x, cursor_dest_y - crosshair_size,
                cursor_dest_x, cursor_dest_y + crosshair_size
            )
        
        painter.restore()
        
    def _line_intersects_rect(self, p1, p2, rect):
        """Check if a line segment intersects with a rectangle."""
        # Simple bounding box check first
        line_rect = QRectF(
            min(p1.x(), p2.x()), min(p1.y(), p2.y()),
            abs(p2.x() - p1.x()), abs(p2.y() - p1.y())
        )
        return rect.intersects(line_rect)
        
    def set_zoom_factor(self, factor):
        """Change the zoom factor."""
        self.zoom_factor = max(2, min(20, factor))  # Clamp between 2x and 20x
        self.source_size = self.widget_size // self.zoom_factor
        self.update_magnifier()