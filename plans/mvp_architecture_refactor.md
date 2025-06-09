# MVP Architecture Refactoring Plan

## Overview

This document outlines the plan to refactor the Photo Album Extractor from its current mixed-concern architecture to a clean Model-View-Presenter (MVP) pattern. This will improve testability, maintainability, and separation of concerns.

## Why MVP?

MVP is chosen over MVC or MVVM because:
- **Testability**: Presenters can be unit tested without GUI dependencies
- **PyQt Compatibility**: Works naturally with Qt's signal/slot mechanism
- **Clear Separation**: Views become purely presentational
- **Passive View**: Minimizes logic in hard-to-test GUI code

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                      View Layer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ MainWindow  │  │ ImageView   │  │ Sidebars    │    │
│  │   View      │  │   View      │  │   Views     │    │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │
│         │                 │                 │            │
│         ▼                 ▼                 ▼            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   Main      │  │   Image     │  │  Sidebar    │    │
│  │ Presenter   │  │ Presenter   │  │ Presenters  │    │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │
└─────────┼─────────────────┼─────────────────┼──────────┘
          │                 │                 │
┌─────────▼─────────────────▼─────────────────▼──────────┐
│                     Model Layer                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   Photo     │  │ BoundingBox │  │   Album     │    │
│  │   Model     │  │   Model     │  │   Model     │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  Extract    │  │  Detection  │  │ Refinement  │    │
│  │  Service    │  │  Service    │  │  Service    │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Model Layer (`core/models/`)

#### Domain Models
```python
# core/models/photo.py
@dataclass
class Photo:
    id: str
    image_data: PIL.Image.Image
    metadata: PhotoMetadata
    source_album: Optional['Album'] = None

# core/models/bounding_box.py
@dataclass
class BoundingBox:
    id: BoxId
    corners: QuadArray
    attributes: Dict[str, str]
    
    def to_quad_points(self) -> List[QPointF]:
        """Convert to Qt points for view layer"""
        pass

# core/models/album.py
@dataclass
class Album:
    path: ImagePath
    image: PIL.Image.Image
    bounding_boxes: List[BoundingBox]
    directory: DirectoryPath
```

#### Services (`core/services/`)
```python
# core/services/extraction_service.py
class ExtractionService:
    def extract_photos(
        self, 
        album: Album, 
        output_dir: DirectoryPath
    ) -> List[Photo]:
        """Extract all photos from album"""
        pass

# core/services/detection_service.py
class DetectionService:
    def __init__(self, strategy: DetectionStrategy):
        self._strategy = strategy
    
    def detect_photos(self, album: Album) -> List[BoundingBox]:
        """Detect photo boundaries in album"""
        pass

# core/services/refinement_service.py
class RefinementService:
    def __init__(self, strategy: BoundaryRefinementStrategy):
        self._strategy = strategy
    
    def refine_boundaries(
        self, 
        album: Album, 
        boxes: List[BoundingBox]
    ) -> List[BoundingBox]:
        """Refine bounding box boundaries"""
        pass
```

### 2. Presenter Layer (`presenters/`)

```python
# presenters/main_presenter.py
class MainPresenter:
    def __init__(
        self,
        view: MainView,
        album_repository: AlbumRepository,
        extraction_service: ExtractionService,
        detection_service: DetectionService,
        refinement_service: RefinementService,
        settings: Settings
    ):
        self._view = view
        self._album_repository = album_repository
        self._extraction_service = extraction_service
        # ... other dependencies
        
        self._current_album: Optional[Album] = None
        self._setup_view_connections()
    
    def load_image(self, path: ImagePath) -> None:
        """Handle image loading request from view"""
        try:
            album = self._album_repository.load(path)
            self._current_album = album
            self._view.display_album(album)
            self._view.show_message("Image loaded successfully")
        except Exception as e:
            self._view.show_error(f"Failed to load image: {e}")
    
    def detect_photos(self) -> None:
        """Handle photo detection request"""
        if not self._current_album:
            return
            
        self._view.show_progress("Detecting photos...")
        try:
            boxes = self._detection_service.detect_photos(self._current_album)
            self._current_album.bounding_boxes = boxes
            self._view.update_bounding_boxes(boxes)
            self._album_repository.save(self._current_album)
        except Exception as e:
            self._view.show_error(f"Detection failed: {e}")
        finally:
            self._view.hide_progress()
```

### 3. View Layer (`gui/views/`)

Views become thin wrappers that:
- Display data passed by presenters
- Forward user interactions to presenters
- Know nothing about business logic

```python
# gui/views/main_view.py
class MainView(Protocol):
    """Interface for main window view"""
    
    # Signals for user actions
    image_load_requested: pyqtSignal  # Emits ImagePath
    detect_photos_requested: pyqtSignal
    refine_all_requested: pyqtSignal
    extract_photos_requested: pyqtSignal
    
    def display_album(self, album: Album) -> None: ...
    def update_bounding_boxes(self, boxes: List[BoundingBox]) -> None: ...
    def show_message(self, message: str) -> None: ...
    def show_error(self, message: str) -> None: ...
    def show_progress(self, message: str) -> None: ...
    def hide_progress(self) -> None: ...

# gui/views/main_window_view.py
class MainWindowView(QMainWindow):
    """Concrete implementation of MainView"""
    
    def __init__(self):
        super().__init__()
        self._init_ui()
    
    def display_album(self, album: Album) -> None:
        # Convert model to view representation
        pixmap = self._album_to_pixmap(album)
        self.image_view.set_image(pixmap)
        
        # Update bounding boxes
        self.update_bounding_boxes(album.bounding_boxes)
```

## Migration Strategy

### Phase 1: Create Core Models (Week 1)
1. Create `core/` package structure
2. Define domain models (Photo, BoundingBox, Album)
3. Create service interfaces
4. Write comprehensive unit tests

### Phase 2: Extract Services (Week 2)
1. Move business logic from GUI to services:
   - Extract photo extraction logic from `main_window.py`
   - Move detection orchestration to `DetectionService`
   - Create `RefinementService` wrapper
2. Keep existing GUI working with minimal changes

### Phase 3: Implement Presenters (Week 3)
1. Create presenter interfaces and implementations
2. Start with `MainPresenter` for core workflows
3. Gradually move logic from views to presenters
4. Maintain backward compatibility

### Phase 4: Refactor Views (Week 4)
1. Strip business logic from views
2. Convert views to passive implementations
3. Ensure all user actions go through presenters
4. Remove direct model manipulation

### Phase 5: Integration & Testing (Week 5)
1. Wire up dependency injection
2. Comprehensive integration tests
3. Update documentation
4. Performance testing

## Testing Strategy

### Unit Tests
- Models: Property-based testing for geometry operations
- Services: Mock repositories and strategies
- Presenters: Mock views and services

### Integration Tests
- Full MVP triads with real implementations
- End-to-end workflows
- Performance benchmarks

### Example Test
```python
def test_main_presenter_detect_photos():
    # Arrange
    mock_view = Mock(spec=MainView)
    mock_detection = Mock(spec=DetectionService)
    album = Album(path="test.jpg", image=Mock(), bounding_boxes=[])
    
    presenter = MainPresenter(
        view=mock_view,
        detection_service=mock_detection,
        # ... other mocks
    )
    presenter._current_album = album
    
    # Act
    presenter.detect_photos()
    
    # Assert
    mock_view.show_progress.assert_called_once()
    mock_detection.detect_photos.assert_called_once_with(album)
    mock_view.hide_progress.assert_called_once()
```

## Benefits

1. **Testability**: 80%+ code coverage becomes achievable
2. **Maintainability**: Clear separation of concerns
3. **Flexibility**: Easy to swap implementations
4. **Reusability**: Core logic can be used in CLI or web versions
5. **Team Productivity**: Parallel development of views and logic

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Large refactoring scope | Incremental migration with backward compatibility |
| Performance regression | Benchmark critical paths before/after |
| Team learning curve | Provide examples and pair programming |
| Over-engineering | Start simple, add complexity only when needed |

## Success Metrics

- Unit test coverage > 80%
- Zero business logic in view layer
- All user actions traceable through presenters
- Reduced coupling (measured by import analysis)
- Faster feature development (after initial refactoring)

## Next Steps

1. Review and approve this plan
2. Create `core/` package structure
3. Begin Phase 1 implementation
4. Set up CI/CD for automated testing

## References

- [Martin Fowler on MVP](https://martinfowler.com/eaaDev/uiArchs.html)
- [Passive View Pattern](https://martinfowler.com/eaaDev/PassiveScreen.html)
- [PyQt MVP Example](https://github.com/pyqt/examples/mvp)