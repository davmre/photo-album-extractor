# MVP Refactor Implementation Guide

## Overview

This guide provides step-by-step instructions for implementing the MVP architecture refactor. Read the full architectural plans in `mvp_architecture_refactor.md` and `mvp_magnifier_refactor.md` before starting.

## Pre-flight Checklist

Before starting:
1. Ensure all tests pass: `pytest tests/test_workflow.py`
2. Create a new branch: `git checkout -b mvp-refactor`
3. Commit frequently - aim for one commit per step
4. Run tests after each major change

## Phase 1: Foundation (Days 1-2)

### Step 1.1: Create Core Package Structure

```bash
mkdir -p core/{models,services,presenters}
touch core/__init__.py
touch core/models/__init__.py
touch core/services/__init__.py
touch core/presenters/__init__.py
```

### Step 1.2: Define Base Model Types

Create `core/models/base.py`:

```python
from typing import Protocol, Any
from PyQt6.QtCore import QObject, pyqtSignal

class Model(Protocol):
    """Base protocol for all models."""
    def to_dict(self) -> dict[str, Any]: ...
    def from_dict(self, data: dict[str, Any]) -> None: ...

class ObservableModel(QObject):
    """Base class for models with change notifications."""
    changed = pyqtSignal()
    
    def notify_changed(self) -> None:
        """Emit changed signal."""
        self.changed.emit()
```

**Design Note**: We use both Protocol (for type checking) and QObject (for signals). This gives us type safety AND Qt integration.

### Step 1.3: Create Photo Domain Model

Create `core/models/photo.py`:

```python
from dataclasses import dataclass
from typing import Optional
import PIL.Image
from photo_types import BoxId, QuadArray

@dataclass
class Photo:
    """Domain model for a photo."""
    id: str
    image_data: PIL.Image.Image
    source_quad: QuadArray
    metadata: dict[str, str]
```

**Key Decision**: Keep models simple PODs (Plain Old Data) initially. Don't add methods yet.

### Step 1.4: Create BoundingBox Model

Transform the current `QuadBoundingBox` (which is a QGraphicsItem) into a pure data model.

Create `core/models/bounding_box.py`:

```python
from dataclasses import dataclass, field
from typing import Optional
import uuid
from photo_types import BoxId, QuadArray

@dataclass
class BoundingBox:
    """Pure data model for a bounding box."""
    id: BoxId
    corners: QuadArray
    attributes: dict[str, str] = field(default_factory=dict)
    
    @staticmethod
    def create_new(corners: QuadArray) -> 'BoundingBox':
        """Factory method to create new box with ID."""
        return BoundingBox(
            id=BoxId(str(uuid.uuid4())),
            corners=corners
        )
```

**Important**: This is NOT a QGraphicsItem! The view layer will create graphics items from these models.

### Step 1.5: Write Model Tests

Create `tests/test_models.py`:

```python
def test_bounding_box_creation():
    corners = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
    box = BoundingBox.create_new(corners)
    
    assert box.id  # Has ID
    assert np.array_equal(box.corners, corners)
    assert box.attributes == {}

def test_photo_model():
    image = PIL.Image.new('RGB', (100, 100))
    photo = Photo(
        id="test-123",
        image_data=image,
        source_quad=np.array([[0, 0], [100, 0], [100, 100], [0, 100]]),
        metadata={"date": "2024-01-01"}
    )
    assert photo.id == "test-123"
```

**Checkpoint**: Run tests. Commit: "Add core domain models"

## Phase 2: Image Editor Model (Days 3-4)

### Step 2.1: Create the Shared Model

This is the most critical piece. Create `core/models/image_editor_model.py`:

```python
from PyQt6.QtCore import QObject, pyqtSignal, QPointF
from typing import Optional, List
import PIL.Image

from .bounding_box import BoundingBox
from photo_types import BoxId, QuadArray

class ImageEditorModel(ObservableModel):
    """Central model for image editing state."""
    
    # Specific signals for different changes
    image_changed = pyqtSignal()
    cursor_moved = pyqtSignal(QPointF)  # Image coordinates
    boxes_changed = pyqtSignal()
    selection_changed = pyqtSignal(list)  # List of selected IDs
    focus_changed = pyqtSignal(object)  # Optional[QPointF]
    
    def __init__(self):
        super().__init__()
        # Core state
        self._image: Optional[PIL.Image.Image] = None
        self._image_path: Optional[str] = None
        
        # Interaction state
        self._cursor_pos = QPointF(0, 0)
        self._focused_point: Optional[QPointF] = None
        
        # Box management
        self._boxes: List[BoundingBox] = []
        self._selected_ids: set[BoxId] = set()
    
    # Image management
    def set_image(self, image: PIL.Image.Image, path: str) -> None:
        """Set the current image."""
        self._image = image
        self._image_path = path
        self.image_changed.emit()
    
    def clear_image(self) -> None:
        """Clear current image and all boxes."""
        self._image = None
        self._image_path = None
        self._boxes.clear()
        self._selected_ids.clear()
        self.image_changed.emit()
        self.boxes_changed.emit()
    
    # Cursor tracking
    def update_cursor_position(self, pos: QPointF) -> None:
        """Update cursor position in image coordinates."""
        if self._cursor_pos != pos:
            self._cursor_pos = pos
            self.cursor_moved.emit(pos)
    
    # Focus management (for magnifier)
    def focus_on_point(self, point: QPointF) -> None:
        """Set focus to specific point (e.g., during corner drag)."""
        self._focused_point = point
        self.focus_changed.emit(point)
    
    def clear_focus(self) -> None:
        """Clear focus, return to cursor tracking."""
        self._focused_point = None
        self.focus_changed.emit(None)
    
    @property
    def effective_focus_point(self) -> QPointF:
        """Get current point of interest (focused or cursor)."""
        return self._focused_point if self._focused_point else self._cursor_pos
    
    # Box management
    def add_box(self, box: BoundingBox) -> None:
        """Add a new bounding box."""
        self._boxes.append(box)
        self.boxes_changed.emit()
    
    def remove_box(self, box_id: BoxId) -> None:
        """Remove a bounding box."""
        self._boxes = [b for b in self._boxes if b.id != box_id]
        self._selected_ids.discard(box_id)
        self.boxes_changed.emit()
        if box_id in self._selected_ids:
            self.selection_changed.emit(list(self._selected_ids))
    
    def update_box_corners(self, box_id: BoxId, corners: QuadArray) -> None:
        """Update corners of existing box."""
        for box in self._boxes:
            if box.id == box_id:
                box.corners = corners
                self.boxes_changed.emit()
                break
    
    def get_box(self, box_id: BoxId) -> Optional[BoundingBox]:
        """Get box by ID."""
        for box in self._boxes:
            if box.id == box_id:
                return box
        return None
    
    # Selection management
    def select_box(self, box_id: BoxId, add_to_selection: bool = False) -> None:
        """Select a box."""
        if not add_to_selection:
            self._selected_ids.clear()
        self._selected_ids.add(box_id)
        self.selection_changed.emit(list(self._selected_ids))
    
    def deselect_all(self) -> None:
        """Clear selection."""
        self._selected_ids.clear()
        self.selection_changed.emit([])
    
    # Property access
    @property
    def image(self) -> Optional[PIL.Image.Image]:
        return self._image
    
    @property
    def boxes(self) -> List[BoundingBox]:
        return self._boxes.copy()  # Return copy to prevent external modification
    
    @property
    def selected_boxes(self) -> List[BoundingBox]:
        return [b for b in self._boxes if b.id in self._selected_ids]
```

**Design Decisions**:
1. Separate signals for different changes (not just one "changed" signal)
2. Image coordinates throughout (views handle their own coordinate transforms)
3. Return copies of lists to prevent external modification
4. Clear separation between interaction state (cursor, focus) and data state (boxes)

### Step 2.2: Test the Model

Create comprehensive tests for the model:

```python
def test_image_editor_model_signals(qtbot):
    model = ImageEditorModel()
    
    # Test image change signal
    with qtbot.waitSignal(model.image_changed):
        model.set_image(PIL.Image.new('RGB', (100, 100)), "test.jpg")
    
    # Test cursor signal
    with qtbot.waitSignal(model.cursor_moved) as blocker:
        model.update_cursor_position(QPointF(50, 50))
    assert blocker.args == [QPointF(50, 50)]
    
    # Test box operations
    box = BoundingBox.create_new(np.array([[0, 0], [10, 0], [10, 10], [0, 10]]))
    with qtbot.waitSignal(model.boxes_changed):
        model.add_box(box)
    assert len(model.boxes) == 1
```

**Checkpoint**: All model tests pass. Commit: "Add ImageEditorModel"

## Phase 3: View Adaptation (Days 5-7)

### Step 3.1: Create View Protocols

Define interfaces that views must implement. Create `core/views/protocols.py`:

```python
from typing import Protocol, Optional
from PyQt6.QtCore import pyqtSignal, QPointF
import PIL.Image

class ImageView(Protocol):
    """Protocol for image display views."""
    # Signals
    mouse_moved: pyqtSignal  # Emits scene coordinates
    mouse_pressed: pyqtSignal
    box_interaction_started: pyqtSignal  # (box_id, interaction_type)
    box_interaction_ended: pyqtSignal
    
    def set_image(self, image: PIL.Image.Image) -> None: ...
    def clear_image(self) -> None: ...
    def scene_to_image_coords(self, scene_pos: QPointF) -> QPointF: ...

class MagnifierView(Protocol):
    """Protocol for magnifier view."""
    def set_image(self, image: PIL.Image.Image) -> None: ...
    def update_focus_point(self, point: QPointF) -> None: ...
    def update_boxes(self, boxes: List[BoundingBox]) -> None: ...
```

### Step 3.2: Create Model-Aware Views

Start with the magnifier as it's simpler. Create `gui/views/model_magnifier_view.py`:

```python
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor

from core.models.image_editor_model import ImageEditorModel
from core.models.bounding_box import BoundingBox

class ModelMagnifierView(QWidget):
    """Magnifier view that observes ImageEditorModel."""
    
    def __init__(self, model: ImageEditorModel, zoom_factor: int = 5):
        super().__init__()
        self.model = model
        self.zoom_factor = zoom_factor
        self.source_size = 150 // zoom_factor
        
        # Cache
        self._pixmap: Optional[QPixmap] = None
        
        self._init_ui()
        self._connect_to_model()
    
    def _init_ui(self):
        # Similar to current magnifier UI setup
        self.setFixedSize(150, 150)
        # ... rest of UI
    
    def _connect_to_model(self):
        """Connect to model signals."""
        self.model.image_changed.connect(self._on_image_changed)
        self.model.cursor_moved.connect(self._on_cursor_moved)
        self.model.focus_changed.connect(self._on_focus_changed)
        self.model.boxes_changed.connect(self._on_boxes_changed)
    
    def _on_image_changed(self):
        """Handle new image from model."""
        if self.model.image:
            # Convert PIL to QPixmap
            self._pixmap = self._pil_to_qpixmap(self.model.image)
        else:
            self._pixmap = None
        self._update_display()
    
    def _on_cursor_moved(self, pos: QPointF):
        """Handle cursor movement."""
        if not self.model._focused_point:  # Only update if not focused
            self._update_display()
    
    def _on_focus_changed(self, point: Optional[QPointF]):
        """Handle focus change."""
        self._update_display()
    
    def _update_display(self):
        """Redraw magnified view."""
        if not self._pixmap:
            return
        
        center = self.model.effective_focus_point
        # ... rest similar to current implementation
```

**Key Changes from Current Magnifier**:
1. No `set_cursor_position()` method - observes model instead
2. No `set_bounding_boxes()` method - gets from model
3. Automatically updates when model changes

### Step 3.3: Adapter Pattern for Existing Views

We can't rewrite ImageView all at once. Create an adapter:

Create `gui/views/image_view_adapter.py`:

```python
class ImageViewAdapter:
    """Adapts existing ImageView to work with ImageEditorModel."""
    
    def __init__(self, image_view: ImageView, model: ImageEditorModel):
        self.view = image_view
        self.model = model
        
        # Connect view to model
        self.view.mouseMoveEvent = self._wrap_mouse_move(self.view.mouseMoveEvent)
        
        # Connect model to view
        self.model.boxes_changed.connect(self._sync_boxes_to_view)
        self.model.image_changed.connect(self._sync_image_to_view)
    
    def _wrap_mouse_move(self, original_handler):
        """Wrap mouse move to update model."""
        def wrapped(event):
            result = original_handler(event)
            # Update model with new position
            scene_pos = self.view.mapToScene(event.pos())
            image_pos = self._scene_to_image_coords(scene_pos)
            self.model.update_cursor_position(image_pos)
            return result
        return wrapped
    
    def _sync_boxes_to_view(self):
        """Sync model boxes to view."""
        # Clear existing
        self.view.clear_boxes()
        
        # Add model boxes
        for box in self.model.boxes:
            graphics_box = self._create_graphics_box(box)
            self.view.add_bounding_box_object(graphics_box)
```

**This lets us migrate incrementally!**

## Phase 4: Presenter Implementation (Days 8-9)

### Step 4.1: Create Base Presenter

Create `core/presenters/base.py`:

```python
from abc import ABC, abstractmethod

class Presenter(ABC):
    """Base presenter class."""
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize presenter (connect signals, etc)."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup presenter (disconnect signals, etc)."""
        pass
```

### Step 4.2: Implement Image Editor Presenter

Create `core/presenters/image_editor_presenter.py`:

```python
from PyQt6.QtCore import QPointF, QObject

from .base import Presenter
from core.models.image_editor_model import ImageEditorModel
from core.services.extraction_service import ExtractionService

class ImageEditorPresenter(Presenter, QObject):
    """Presenter for image editing operations."""
    
    def __init__(
        self,
        model: ImageEditorModel,
        main_view: ImageView,
        magnifier_view: Optional[MagnifierView] = None,
        extraction_service: Optional[ExtractionService] = None
    ):
        super().__init__()
        self.model = model
        self.main_view = main_view
        self.magnifier_view = magnifier_view
        self.extraction_service = extraction_service
        
        self._dragging_box_id: Optional[BoxId] = None
        self._dragging_corner_index: Optional[int] = None
    
    def initialize(self):
        """Connect all signals."""
        # View -> Presenter
        self.main_view.mouse_moved.connect(self._on_view_mouse_moved)
        self.main_view.box_corner_drag_started.connect(self._on_corner_drag_started)
        self.main_view.box_corner_drag_ended.connect(self._on_corner_drag_ended)
        self.main_view.box_corner_dragged.connect(self._on_corner_dragged)
        
        # Model -> View connections happen in view constructors
    
    def _on_view_mouse_moved(self, scene_pos: QPointF):
        """Handle mouse movement from view."""
        # Convert to image coordinates
        image_pos = self.main_view.scene_to_image_coords(scene_pos)
        self.model.update_cursor_position(image_pos)
    
    def _on_corner_drag_started(self, box_id: BoxId, corner_index: int):
        """Handle start of corner dragging."""
        box = self.model.get_box(box_id)
        if box:
            # Focus magnifier on this corner
            corner_point = QPointF(*box.corners[corner_index])
            self.model.focus_on_point(corner_point)
            
            # Track what we're dragging
            self._dragging_box_id = box_id
            self._dragging_corner_index = corner_index
    
    def _on_corner_dragged(self, new_pos: QPointF):
        """Handle corner being dragged."""
        if self._dragging_box_id and self._dragging_corner_index is not None:
            box = self.model.get_box(self._dragging_box_id)
            if box:
                # Update corner position
                new_corners = box.corners.copy()
                new_corners[self._dragging_corner_index] = [new_pos.x(), new_pos.y()]
                self.model.update_box_corners(self._dragging_box_id, new_corners)
                
                # Keep magnifier focused on new position
                self.model.focus_on_point(new_pos)
    
    def _on_corner_drag_ended(self):
        """Handle end of corner dragging."""
        self.model.clear_focus()
        self._dragging_box_id = None
        self._dragging_corner_index = None
    
    # Business operations
    def add_box_at_position(self, scene_pos: QPointF):
        """Add a new box at the given position."""
        # Convert to image coordinates
        image_pos = self.main_view.scene_to_image_coords(scene_pos)
        
        # Create default box around position
        size = 100  # Default size
        corners = np.array([
            [image_pos.x() - size/2, image_pos.y() - size/2],
            [image_pos.x() + size/2, image_pos.y() - size/2],
            [image_pos.x() + size/2, image_pos.y() + size/2],
            [image_pos.x() - size/2, image_pos.y() + size/2]
        ])
        
        box = BoundingBox.create_new(corners)
        self.model.add_box(box)
        self.model.select_box(box.id)
```

**Design Notes**:
1. Presenter is stateful (tracks dragging state)
2. Converts between coordinate systems
3. Orchestrates complex operations
4. Knows about services but not their implementation

## Phase 5: Integration (Days 10-12)

### Step 5.1: Create Application Controller

This wires everything together. Create `core/app_controller.py`:

```python
class ApplicationController:
    """Main application controller that creates and wires components."""
    
    def __init__(self):
        # Create models
        self.image_model = ImageEditorModel()
        
        # Services will be created as needed
        self._extraction_service: Optional[ExtractionService] = None
        
        # Presenters will be created with views
        self._presenters: List[Presenter] = []
    
    def setup_image_editor(self, main_view: ImageView, magnifier_view: MagnifierView):
        """Set up image editing with given views."""
        presenter = ImageEditorPresenter(
            model=self.image_model,
            main_view=main_view,
            magnifier_view=magnifier_view,
            extraction_service=self._get_extraction_service()
        )
        presenter.initialize()
        self._presenters.append(presenter)
        
        return presenter
    
    def _get_extraction_service(self):
        """Lazy creation of extraction service."""
        if not self._extraction_service:
            self._extraction_service = ExtractionService()
        return self._extraction_service
    
    def cleanup(self):
        """Clean up all presenters."""
        for presenter in self._presenters:
            presenter.cleanup()
```

### Step 5.2: Modify Main Window

Update `gui/main_window.py` to use the new architecture:

```python
class PhotoExtractorApp(QMainWindow):
    def __init__(self, ...):
        super().__init__()
        
        # Create application controller
        self.app_controller = ApplicationController()
        
        # Create UI
        self.init_ui()
        
        # Wire up MVP after UI is created
        self._setup_mvp()
    
    def _setup_mvp(self):
        """Set up MVP architecture."""
        # Create model-aware magnifier
        self.model_magnifier = ModelMagnifierView(self.app_controller.image_model)
        self.attributes_sidebar.layout().replaceWidget(
            self.attributes_sidebar.magnifier,
            self.model_magnifier
        )
        
        # Set up presenter
        self.image_presenter = self.app_controller.setup_image_editor(
            main_view=self.image_view,
            magnifier_view=self.model_magnifier
        )
        
        # Remove old signal connections
        # (They're now handled by presenter)
    
    def load_image_from_path(self, path: str):
        """Load image - now updates model instead of view directly."""
        try:
            image = image_processor.load_image(path)
            self.app_controller.image_model.set_image(image, path)
            # Views update automatically!
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load image: {e}")
```

## Phase 6: Testing & Cleanup (Days 13-14)

### Step 6.1: Test the Integration

Create `tests/test_mvp_integration.py`:

```python
def test_magnifier_tracks_cursor(qtbot):
    """Test that magnifier follows cursor through MVP."""
    # Create components
    model = ImageEditorModel()
    model.set_image(PIL.Image.new('RGB', (200, 200)), "test.jpg")
    
    main_view = MockImageView()
    magnifier = ModelMagnifierView(model)
    
    presenter = ImageEditorPresenter(model, main_view, magnifier)
    presenter.initialize()
    
    # Simulate mouse movement
    main_view.simulate_mouse_move(QPointF(100, 100))
    
    # Verify magnifier is centered on cursor
    assert model.cursor_position == QPointF(100, 100)
    assert magnifier.last_rendered_center == QPointF(100, 100)
```

### Step 6.2: Remove Old Code

Once tests pass:
1. Remove old signal connections in `main_window.py`
2. Remove `set_cursor_position` from old magnifier
3. Remove direct widget-to-widget connections

### Step 6.3: Document the New Architecture

Update docstrings to reflect MVP:

```python
class ImageEditorModel(ObservableModel):
    """Central model for image editing state.
    
    This model is observed by multiple views (main image view, magnifier, etc)
    and updated by presenters in response to user actions.
    
    Emits specific signals when state changes:
    - image_changed: When the image is loaded/cleared
    - cursor_moved: When cursor position updates
    - boxes_changed: When boxes are added/removed/modified
    - selection_changed: When selection changes
    - focus_changed: When focus point changes (for magnifier)
    """
```

## Common Pitfalls to Avoid

1. **Don't let views talk to each other directly** - Always go through model
2. **Don't put business logic in views** - That goes in presenters/services
3. **Don't forget to disconnect signals** - Memory leaks are real
4. **Don't update model from within model signal handlers** - Infinite loops!
5. **Keep coordinate systems clear** - Document whether using scene, view, or image coordinates

## Migration Checklist

- [ ] All tests still pass
- [ ] No direct view-to-view connections remain
- [ ] Model has no UI dependencies (no Qt widgets)
- [ ] Views have no business logic
- [ ] Presenters handle all user interactions
- [ ] Old code is removed (not just commented out)
- [ ] New architecture is documented

## Success Criteria

You'll know the refactor is successful when:
1. You can test the magnifier without creating a main window
2. You can add a new view (e.g., minimap) without touching existing views
3. You can save/load the complete editing state
4. The codebase is easier to understand and modify

Good luck! Remember to commit frequently and ask for help if you get stuck.