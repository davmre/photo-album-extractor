"""
Custom graphics view for displaying images with bounding box interaction.
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageQt
from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QAction,
    QEnterEvent,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPixmap,
    QWheelEvent,
)
from PyQt6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QMenu

import core.geometry as geometry
import core.inscribed_rectangle as inscribed_rectangle
from core import refinement_strategies
from core.bounding_box import BoundingBox
from core.settings import app_settings
from gui.quad_bounding_box import QuadBoundingBox


class ImageView(QGraphicsView):
    """Custom graphics view for displaying images with bounding box interaction."""

    # Signal emitted when boxes are added or removed
    boxes_changed = pyqtSignal()
    # Signal emitted when a box is selected
    box_selected = pyqtSignal(BoundingBox)  # Emits (box_id, BoundingBoxData)
    # Signal emitted when no box is selected
    box_deselected = pyqtSignal()
    # Signal emitted when mouse moves over image
    mouse_moved = pyqtSignal(object)  # Emits QPointF in scene coordinates
    # Signal emitted when image changes
    image_updated = pyqtSignal()
    # Signal emitted when mouse enters viewport
    mouse_entered_viewport = pyqtSignal()

    def __init__(self, settings=None) -> None:
        super().__init__()
        self.selected_box: QuadBoundingBox | None = None

        self.refine_debug_dir: str | None = None

        # Set up the graphics scene
        self._scene = QGraphicsScene()
        self.setScene(self._scene)

        # Image item
        self.image_item: QGraphicsPixmapItem | None = None
        self.bounding_boxes: list[QuadBoundingBox] = []

        # Drag state
        self.is_dragging: bool = False
        self.drag_start_pos: QPointF | None = None
        self.temp_box: QuadBoundingBox | None = None

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

        # Enable mouse tracking for magnifier
        self.setMouseTracking(True)

        # Track enter/leave events for magnifier mode switching
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.mouse_entered = False

        # Enable keyboard focus for key events
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_image(self, image: Image.Image | None = None):
        """Set the image to display."""
        # QPixmap uses implicit sharing semantics, so it seems we need to
        # keep a reference to the QImage so it doesn't get GC'd.
        if image is None:
            self._image_qt = None
            pixmap = QPixmap()
        else:
            self._image_qt = ImageQt.ImageQt(image)
            pixmap = QPixmap.fromImage(self._image_qt)

        # Clear existing image
        if self.image_item:
            self._scene.removeItem(self.image_item)

        # Add new image
        self.image_item = QGraphicsPixmapItem(pixmap)
        self._scene.addItem(self.image_item)

        # Fit image in view
        self.fitInView(self.image_item, Qt.AspectRatioMode.KeepAspectRatio)

        # Emit signal for magnifier
        self.image_updated.emit()

    def add_bounding_box(self, box_data: BoundingBox, emit_signals=True):
        """Add a pre-created bounding box object to the scene."""
        if self.image_item is None:
            return None

        box = QuadBoundingBox(box_data=box_data)
        self._scene.addItem(box)

        # Add handles to scene
        if hasattr(box, "handles"):
            for handle in box.handles:
                self._scene.addItem(handle)

        self.bounding_boxes.append(box)

        # Connect signals
        box.changed.connect(self.box_changed)
        box.selected_changed.connect(self.on_box_selection_changed)

        # Emit signal that boxes changed
        if emit_signals:
            self.boxes_changed.emit()

        return box

    def remove_bounding_box(self, box, emit_signals=True):
        """Remove a bounding box from the scene."""
        if box in self.bounding_boxes:
            # Remove handles
            if hasattr(box, "handles"):
                for handle in box.handles:
                    self._scene.removeItem(handle)

            # Remove box
            self._scene.removeItem(box)
            self.bounding_boxes.remove(box)

            # Emit signal that boxes changed
            if emit_signals:
                self.boxes_changed.emit()

    def clear_boxes(self, emit_signals=True):
        """Remove all bounding boxes."""
        if self.bounding_boxes:  # Only emit signal if there were boxes to remove
            for box in self.bounding_boxes[
                :
            ]:  # Copy list to avoid modification during iteration
                self.remove_bounding_box(box, emit_signals=emit_signals)
            # Note: remove_bounding_box already emits boxes_changed for each removal

    def get_bounding_box_data_list(self) -> list[BoundingBox]:
        """Get all bounding box data for extraction."""
        return [box.get_bounding_box_data() for box in self.bounding_boxes]

    def show_context_menu(self, position):
        """Show context menu for adding/removing boxes."""
        # Convert position to scene coordinates
        scene_pos = self.mapToScene(position)

        # Check if we're clicking on a bounding box
        clicked_item = self._scene.itemAt(scene_pos, self.transform())
        clicked_box = None

        # Find the bounding box if we clicked on one or its handle
        for box in self.bounding_boxes:
            if clicked_item == box or (
                hasattr(box, "handles") and clicked_item in box.handles
            ):
                clicked_box = box
                break

        # Create context menu
        menu = QMenu(self)

        if clicked_box:
            # Menu for existing box
            refine_action = menu.addAction("Refine")
            rectangle_inner_action = menu.addAction("Rectangle-ify (inner)")
            rectangle_outer_action = menu.addAction("Rectangle-ify (outer)")

            # Add keep rectangular toggle
            keep_rectangular_action: QAction = menu.addAction("Keep Rectangular")  # type: ignore
            keep_rectangular_action.setCheckable(True)
            keep_rectangular_action.setChecked(clicked_box.keep_rectangular)
            # Only enable for rectangles, gray out for non-rectangles
            keep_rectangular_action.setEnabled(
                clicked_box.get_bounding_box_data().is_rectangle()
            )

            # Add mark as good toggle
            mark_as_good_action: QAction = menu.addAction("Mark as Good")  # type: ignore
            mark_as_good_action.setCheckable(True)
            mark_as_good_action.setChecked(
                clicked_box.get_bounding_box_data().marked_as_good
            )

            remove_action = menu.addAction("Remove")
            action = menu.exec(self.mapToGlobal(position))

            if action == refine_action:
                self.refine_bounding_box(clicked_box, multiscale=True)
            elif action == rectangle_inner_action:
                self.rectangleify_bounding_box(clicked_box, inner=True)
            elif action == rectangle_outer_action:
                self.rectangleify_bounding_box(clicked_box, inner=False)
            elif action == keep_rectangular_action:
                clicked_box.keep_rectangular = keep_rectangular_action.isChecked()
            elif action == mark_as_good_action:
                clicked_box.set_marked_as_good(mark_as_good_action.isChecked())
            elif action == remove_action:
                self.remove_bounding_box(clicked_box)

    def box_changed(self):
        """Handle bounding box changes and update magnifier."""
        # Emit signal to update magnifier with new bounding box positions
        self.boxes_changed.emit()

    def image_as_numpy(self, format="rgb"):
        image = self._image_qt
        if image is None:
            raise ValueError("no image loaded!")
        width = image.width()
        height = image.height()

        # Convert QImage to numpy array
        ptr = image.constBits()
        ptr.setsize(height * width * 4)  # 4 bytes per pixel (RGBA)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        # Convert RGBA to BGR for OpenCV
        if format == "rgba":
            image = arr
        if format == "rgb":
            image = arr[:, :, [0, 1, 2]]  # RGB format
        elif format == "bgr":
            image = arr[:, :, [2, 1, 0]]  # BGR format
        else:
            raise ValueError("Unrecognized image format", format)
        return image

    def rectangleify_bounding_box(self, box: QuadBoundingBox, inner=True):
        if not self.image_item or not isinstance(box, QuadBoundingBox):
            return
        corner_coords = box.get_ordered_corners_for_extraction()
        if inner:
            rect_array, _ = inscribed_rectangle.largest_inscribed_rectangle(
                corner_coords
            )
        else:
            rect_array, _ = geometry.minimum_bounding_rectangle(corner_coords)
        box.set_corners(rect_array)
        box.keep_rectangular = True

    def refine_bounding_box(
        self, box: QuadBoundingBox, multiscale=False, enforce_parallel_sides=None
    ):
        """Refine a single bounding box using edge detection."""
        if not self.image_item or not isinstance(box, QuadBoundingBox):
            return

        strategy = refinement_strategies.configure_refinement_strategy(app_settings)

        # Get the current image as numpy array
        image_bgr = self.image_as_numpy(format="bgr")

        # Get current box corners in image coordinates
        corner_coords = box.get_ordered_corners_for_extraction()

        debug_dir = getattr(self, "refine_debug_dir", None)
        if debug_dir is not None:
            debug_dir = os.path.join(debug_dir, str(box.box_id))

        try:
            refined_corners = strategy.refine(
                image_bgr, corner_coords, reltol=app_settings.refine_current_tolerance, debug_dir=debug_dir
            )
            box.set_corners(refined_corners)
            box.set_marked_as_good(False)
            box.keep_rectangular = geometry.is_rectangle(refined_corners)

        except Exception as e:
            print(f"Error refining bounding box: {e}")

    def refine_all_bounding_boxes(self):
        """Refine all bounding boxes using edge detection."""
        for box in self.bounding_boxes:
            if isinstance(box, QuadBoundingBox):
                self.refine_bounding_box(box)

    def refresh_all_validation(self):
        """Refresh validation for all bounding boxes using current settings."""
        for box in self.bounding_boxes:
            if isinstance(box, QuadBoundingBox):
                box.update_validation()  # This triggers _update_validation() and repaint

    def on_box_selection_changed(self, box_id: str):
        """Handle box selection changes."""
        # Find the box with this ID
        selected_box = None
        for box in self.bounding_boxes:
            if isinstance(box, QuadBoundingBox):
                if box.box_id == box_id:
                    selected_box = box
                    box.set_selected(True)
                else:
                    box.set_selected(False)

        # Update selected box reference
        self.selected_box = selected_box

        # Emit appropriate signal
        if selected_box:
            # Emit the complete bounding box data
            bounding_box_data = selected_box.get_bounding_box_data()
            self.box_selected.emit(bounding_box_data)
        else:
            self.box_deselected.emit()

    def clear_selection(self):
        """Clear the current selection."""
        if self.selected_box:
            self.selected_box.set_selected(False)
            self.selected_box = None
            self.box_deselected.emit()

    def get_selected_box(self):
        """Get the currently selected box."""
        return self.selected_box

    def update_box_data(self, bounding_box_data: BoundingBox):
        """Update complete bounding box data for a specific box."""
        for box in self.bounding_boxes:
            if (
                isinstance(box, QuadBoundingBox)
                and box.box_id == bounding_box_data.box_id
            ):
                box.set_bounding_box_data(bounding_box_data)
                # Update magnifier
                self.boxes_changed.emit()
                break

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press for starting box creation."""
        if event.button() == Qt.MouseButton.LeftButton:
            # Ensure ImageView has focus for keyboard events
            self.setFocus()

            # Check if we're clicking on an existing item
            scene_pos = self.mapToScene(event.position().toPoint())
            clicked_item = self._scene.itemAt(scene_pos, self.transform())

            # If we didn't click on an existing item, clear selection and start creating a new box
            if clicked_item is None or clicked_item == self.image_item:
                self.clear_selection()
                self.is_dragging = True
                self.drag_start_pos = scene_pos
                self.setDragMode(QGraphicsView.DragMode.NoDrag)  # Disable default drag
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move for box creation and magnifier updates."""
        scene_pos = self.mapToScene(event.position().toPoint())

        # Emit mouse position for magnifier
        self.mouse_moved.emit(scene_pos)

        if self.is_dragging and self.drag_start_pos is not None:
            corners = [
                self.drag_start_pos,
                QPointF(scene_pos.x(), self.drag_start_pos.y()),
                scene_pos,
                QPointF(self.drag_start_pos.x(), scene_pos.y()),
            ]

            # Create or update temporary box for preview.
            if self.temp_box:
                # self._scene.removeItem(self.temp_box)
                # self.temp_box = None
                self.temp_box.set_corners(corners)
            else:
                self.temp_box = QuadBoundingBox(BoundingBox.new(corners=corners))
                self._scene.addItem(self.temp_box)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release for finishing box creation."""
        if event.button() == Qt.MouseButton.LeftButton and self.is_dragging:
            scene_pos = self.mapToScene(event.position().toPoint())

            # Remove temporary box if it exists
            if self.temp_box:
                self._scene.removeItem(self.temp_box)
                self.temp_box = None

            # Create final box if drag was significant
            if self.drag_start_pos is not None:
                distance = (
                    (scene_pos.x() - self.drag_start_pos.x()) ** 2
                    + (scene_pos.y() - self.drag_start_pos.y()) ** 2
                ) ** 0.5
                if distance > 10:  # Minimum drag distance
                    # Create the actual quadrilateral bounding box
                    corners = [
                        self.drag_start_pos,
                        QPointF(scene_pos.x(), self.drag_start_pos.y()),
                        scene_pos,
                        QPointF(self.drag_start_pos.x(), scene_pos.y()),
                    ]
                    # Create new box with unique ID
                    box_data = BoundingBox.new(corners=corners)
                    self.add_bounding_box(box_data)

            # Reset drag state
            self.is_dragging = False
            self.drag_start_pos = None
            self.setDragMode(
                QGraphicsView.DragMode.RubberBandDrag
            )  # Re-enable default drag

        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        """Handle zoom with mouse wheel or trackpad."""
        # On macOS, distinguish between scroll and zoom gestures
        # If Ctrl/Cmd is held down, zoom; otherwise, scroll normally
        if event.modifiers() & (
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier
        ):
            # Zoom when Ctrl/Cmd is held
            zoom_factor = 1.15
            if event.angleDelta().y() >= 0:
                zoom_factor = 1.0 / zoom_factor
            self.scale(zoom_factor, zoom_factor)
        else:
            # Normal scrolling behavior
            super().wheelEvent(event)

    def enterEvent(self, event: QEnterEvent):
        """Handle mouse entering the viewport."""
        # Signal that cursor tracking should resume
        self.mouse_entered = True
        self.mouse_entered_viewport.emit()
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Handle mouse leaving the viewport."""
        # Track when mouse leaves viewport
        self.mouse_entered = False
        super().leaveEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        """Handle key press events."""
        if event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace:
            # Delete the currently selected box
            if self.selected_box:
                self.remove_bounding_box(self.selected_box)
                self.selected_box = None
                self.box_deselected.emit()
                return
        elif event.key() == Qt.Key.Key_R:
            # Refine the currently selected box
            if self.selected_box and isinstance(self.selected_box, QuadBoundingBox):
                self.refine_bounding_box(self.selected_box, multiscale=True)
                return

        # Let parent handle other keys
        super().keyPressEvent(event)
