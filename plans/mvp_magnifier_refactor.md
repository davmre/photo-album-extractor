# MVP Magnifier Widget Refactor

## Current Problem

The magnifier widget and main image view are tightly coupled through event passing:
- `image_view.mouse_moved.connect(self.attributes_sidebar.magnifier.set_cursor_position)`
- `image_view.image_updated.connect(self.update_magnifier)`
- Manual synchronization of bounding boxes
- Direct widget-to-widget communication

This violates MVP principles where views should not know about each other.

## MVP Solution

In a proper MVP architecture, both views observe the same model, eliminating the need for complex event synchronization.

### Shared Model

```python
# core/models/image_editor_model.py
from dataclasses import dataclass, field
from typing import List, Optional, Set
from PyQt6.QtCore import QObject, pyqtSignal, QPointF
import PIL.Image

from photo_types import ImageCoordinate, BoxId

@dataclass
class ImageEditorModel(QObject):
    """Shared model for image editing state."""
    
    # Signals for model changes
    image_changed = pyqtSignal()
    cursor_position_changed = pyqtSignal(QPointF)
    boxes_changed = pyqtSignal()
    selection_changed = pyqtSignal()
    focused_point_changed = pyqtSignal(QPointF)
    zoom_changed = pyqtSignal(float)
    
    def __init__(self):
        super().__init__()
        self._image: Optional[PIL.Image.Image] = None
        self._cursor_position: QPointF = QPointF(0, 0)
        self._bounding_boxes: List[BoundingBox] = []
        self._selected_box_ids: Set[BoxId] = set()
        self._focused_point: Optional[QPointF] = None
        self._zoom_level: float = 1.0
    
    @property
    def image(self) -> Optional[PIL.Image.Image]:
        return self._image
    
    @image.setter
    def image(self, value: Optional[PIL.Image.Image]) -> None:
        self._image = value
        self.image_changed.emit()
    
    @property
    def cursor_position(self) -> QPointF:
        return self._cursor_position
    
    @cursor_position.setter
    def cursor_position(self, value: QPointF) -> None:
        if self._cursor_position != value:
            self._cursor_position = value
            self.cursor_position_changed.emit(value)
    
    @property
    def bounding_boxes(self) -> List[BoundingBox]:
        return self._bounding_boxes
    
    def add_bounding_box(self, box: BoundingBox) -> None:
        self._bounding_boxes.append(box)
        self.boxes_changed.emit()
    
    def remove_bounding_box(self, box_id: BoxId) -> None:
        self._bounding_boxes = [b for b in self._bounding_boxes if b.id != box_id]
        self._selected_box_ids.discard(box_id)
        self.boxes_changed.emit()
        self.selection_changed.emit()
    
    def update_box_corners(self, box_id: BoxId, corners: QuadArray) -> None:
        for box in self._bounding_boxes:
            if box.id == box_id:
                box.corners = corners
                self.boxes_changed.emit()
                break
    
    def select_box(self, box_id: BoxId, add_to_selection: bool = False) -> None:
        if not add_to_selection:
            self._selected_box_ids.clear()
        self._selected_box_ids.add(box_id)
        self.selection_changed.emit()
    
    def focus_on_point(self, point: QPointF) -> None:
        """Set a specific point as the focus (e.g., for corner dragging)."""
        self._focused_point = point
        self.focused_point_changed.emit(point)
    
    def clear_focus(self) -> None:
        """Clear the focused point."""
        self._focused_point = None
        self.focused_point_changed.emit(self.cursor_position)
```

### Presenter Pattern

```python
# presenters/image_editor_presenter.py
class ImageEditorPresenter:
    """Presenter coordinating image editing operations."""
    
    def __init__(
        self, 
        model: ImageEditorModel,
        main_view: ImageEditView,
        magnifier_view: MagnifierView,
        album_repository: AlbumRepository
    ):
        self.model = model
        self.main_view = main_view
        self.magnifier_view = magnifier_view
        self.album_repository = album_repository
        
        self._setup_connections()
    
    def _setup_connections(self):
        # View to Presenter connections
        self.main_view.mouse_moved.connect(self._on_mouse_moved)
        self.main_view.box_corner_drag_started.connect(self._on_corner_drag_started)
        self.main_view.box_corner_drag_ended.connect(self._on_corner_drag_ended)
        self.main_view.box_selected.connect(self._on_box_selected)
        
        # Model to View connections (automatic updates)
        self.model.image_changed.connect(self._update_views_image)
        self.model.cursor_position_changed.connect(self._update_cursor_displays)
        self.model.boxes_changed.connect(self._update_box_displays)
        self.model.focused_point_changed.connect(self._update_magnifier_focus)
    
    def _on_mouse_moved(self, scene_pos: QPointF) -> None:
        """Handle mouse movement from main view."""
        # Convert to image coordinates
        image_pos = self.main_view.scene_to_image_coords(scene_pos)
        self.model.cursor_position = image_pos
    
    def _on_corner_drag_started(self, box_id: BoxId, corner_index: int) -> None:
        """Handle corner drag start."""
        box = self._get_box_by_id(box_id)
        if box:
            corner_pos = QPointF(*box.corners[corner_index])
            self.model.focus_on_point(corner_pos)
    
    def _on_corner_drag_ended(self) -> None:
        """Handle corner drag end."""
        self.model.clear_focus()
    
    def _update_views_image(self) -> None:
        """Update both views when image changes."""
        if self.model.image:
            # Views handle their own rendering
            self.main_view.set_image(self.model.image)
            self.magnifier_view.set_image(self.model.image)
    
    def _update_cursor_displays(self, pos: QPointF) -> None:
        """Update cursor-related displays."""
        # Views automatically update based on model state
        # No need to manually sync
        pass
    
    def _update_magnifier_focus(self, point: QPointF) -> None:
        """Update magnifier focus point."""
        # Magnifier view observes model directly
        pass
```

### Refactored Views

```python
# gui/views/magnifier_view.py
class MagnifierView(QWidget):
    """Magnifier view observing the image editor model."""
    
    def __init__(self, model: ImageEditorModel, zoom_factor: int = 5):
        super().__init__()
        self.model = model
        self.zoom_factor = zoom_factor
        self._pixmap_cache: Optional[QPixmap] = None
        
        self._init_ui()
        self._connect_to_model()
    
    def _connect_to_model(self):
        """Connect to model signals."""
        self.model.image_changed.connect(self._on_image_changed)
        self.model.cursor_position_changed.connect(self._on_cursor_changed)
        self.model.focused_point_changed.connect(self._on_focus_changed)
        self.model.boxes_changed.connect(self._on_boxes_changed)
    
    def _on_image_changed(self):
        """Handle image change from model."""
        if self.model.image:
            # Convert to QPixmap for display
            self._pixmap_cache = self._pil_to_qpixmap(self.model.image)
        else:
            self._pixmap_cache = None
        self._update_display()
    
    def _on_cursor_changed(self, pos: QPointF):
        """Handle cursor position change."""
        if not self.model._focused_point:  # Only track cursor when not focused
            self._update_display()
    
    def _on_focus_changed(self, point: Optional[QPointF]):
        """Handle focus point change."""
        self._update_display()
    
    def _update_display(self):
        """Update the magnified display."""
        if not self._pixmap_cache:
            return
        
        # Determine center point
        center = self.model._focused_point or self.model.cursor_position
        
        # Extract region around center
        source_rect = self._calculate_source_rect(center)
        
        # Create magnified pixmap
        magnified = self._create_magnified_view(source_rect)
        
        # Draw overlay elements (boxes, crosshair)
        self._draw_overlays(magnified, source_rect)
        
        # Update display
        self.display_label.setPixmap(magnified)
    
    def _calculate_source_rect(self, center: QPointF) -> QRectF:
        """Calculate source rectangle centered on point."""
        half_size = self.source_size / 2
        return QRectF(
            center.x() - half_size,
            center.y() - half_size,
            self.source_size,
            self.source_size
        )

# gui/views/image_edit_view.py  
class ImageEditView(QGraphicsView):
    """Main image editing view observing the model."""
    
    def __init__(self, model: ImageEditorModel):
        super().__init__()
        self.model = model
        self._scene = QGraphicsScene()
        self.setScene(self._scene)
        
        self._connect_to_model()
        self.setMouseTracking(True)
    
    def _connect_to_model(self):
        """Connect to model signals."""
        self.model.image_changed.connect(self._on_image_changed)
        self.model.boxes_changed.connect(self._on_boxes_changed)
        self.model.selection_changed.connect(self._on_selection_changed)
    
    def mouseMoveEvent(self, event):
        """Track mouse movement."""
        super().mouseMoveEvent(event)
        scene_pos = self.mapToScene(event.pos())
        # Presenter will update model
        self.mouse_moved.emit(scene_pos)
    
    def _on_image_changed(self):
        """Handle image change from model."""
        # Update display based on model state
        self._update_image_display()
    
    def _on_boxes_changed(self):
        """Handle box changes from model."""
        # Recreate box graphics from model
        self._update_box_graphics()
```

## Benefits of MVP Approach

### 1. Eliminated Event Spaghetti
- No more direct widget-to-widget connections
- All state flows through the model
- Clear, traceable data flow

### 2. Simplified Synchronization
- Both views observe same model
- Automatic consistency
- No manual sync needed

### 3. Better Testability
```python
def test_magnifier_follows_cursor():
    model = ImageEditorModel()
    magnifier = MagnifierView(model)
    
    # Update model
    model.cursor_position = QPointF(100, 100)
    
    # Magnifier automatically updates
    assert magnifier._last_rendered_center == QPointF(100, 100)

def test_corner_focus():
    model = ImageEditorModel()
    magnifier = MagnifierView(model)
    
    # Focus on specific point
    model.focus_on_point(QPointF(50, 50))
    
    # Magnifier shows focused point, not cursor
    assert magnifier._last_rendered_center == QPointF(50, 50)
```

### 4. Extensibility
- Easy to add new views (minimap, histogram, etc.)
- Views can be shown/hidden without breaking others
- Can save/restore complete editing state

### 5. Cleaner Code
- Views focus on display logic only
- Model handles state management
- Presenter coordinates interactions

## Migration Strategy

### Phase 1: Create Shared Model
1. Implement `ImageEditorModel`
2. Add comprehensive tests
3. Keep existing code working

### Phase 2: Adapt Magnifier
1. Create `MagnifierView` observing model
2. Remove direct event connections
3. Test in isolation

### Phase 3: Adapt Main View  
1. Refactor `ImageView` to observe model
2. Move interaction logic to presenter
3. Remove widget coupling

### Phase 4: Create Presenter
1. Implement `ImageEditorPresenter`
2. Wire up all connections
3. Remove old event handling

## Additional Improvements

### 1. State Persistence
```python
def save_editor_state(model: ImageEditorModel) -> Dict:
    return {
        "zoom_level": model.zoom_level,
        "selected_boxes": list(model._selected_box_ids),
        "scroll_position": model.scroll_position
    }
```

### 2. Undo/Redo
Model changes can be easily tracked for undo:
```python
class UndoableImageEditorModel(ImageEditorModel):
    def __init__(self):
        super().__init__()
        self._history: List[ModelState] = []
        self._redo_stack: List[ModelState] = []
```

### 3. Multiple Magnifiers
Easy to support multiple magnifier views:
```python
# Each magnifier can have different zoom/settings
magnifier1 = MagnifierView(model, zoom_factor=3)
magnifier2 = MagnifierView(model, zoom_factor=10)
```

This refactoring transforms the magnifier from a hacky synchronized widget into a proper view in the MVP architecture, eliminating complexity while adding capabilities.