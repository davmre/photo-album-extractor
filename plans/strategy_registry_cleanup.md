# Strategy Registry Cleanup Plan

## Overview

This document outlines the plan to refactor the refinement strategy registry from its current lambda-based implementation to a type-safe, extensible plugin architecture. This will improve maintainability, enable better testing, and provide clearer interfaces for adding new strategies.

## Current Problems

```python
# Current implementation in refine_bounds.py
REFINEMENT_STRATEGIES: Dict[str, Callable[..., QuadArray]] = {
    "Original (200px res)": 
        (lambda image, corner_points, debug_dir=None: 
         refine_bounding_box(image, corner_points, resolution=200, 
                             enforce_parallel_sides=True, 
                             debug_dir=debug_dir)),
    # ... more lambdas
}
```

**Issues:**
1. **Type Safety**: Lambdas bypass protocol checking
2. **Testing**: Difficult to test individual strategies
3. **Configuration**: Parameters hard-coded in lambdas
4. **Documentation**: No way to document strategies properly
5. **Debugging**: Lambda names in stack traces are unhelpful
6. **Extensibility**: No clean way to add custom strategies

## Proposed Architecture

### Strategy Class Hierarchy

```python
# image_processing/strategies/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

from photo_types import BGRImage, DirectoryPath, QuadArray

@dataclass
class StrategyMetadata:
    """Metadata for a strategy."""
    name: str
    display_name: str
    description: str
    category: str
    version: str
    author: Optional[str] = None
    experimental: bool = False

class RefinementStrategy(ABC):
    """Base class for all refinement strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._validate_config()
    
    @property
    @abstractmethod
    def metadata(self) -> StrategyMetadata:
        """Return strategy metadata."""
        pass
    
    @abstractmethod
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        pass
    
    @abstractmethod
    def refine_boundary(
        self, 
        image: BGRImage, 
        initial_quad: QuadArray,
        debug_dir: Optional[DirectoryPath] = None
    ) -> QuadArray:
        """Refine the boundary of a photo in an image."""
        pass
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Return JSON schema for configuration."""
        return {}
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.metadata.display_name})"
```

### Concrete Strategy Implementation

```python
# image_processing/strategies/original.py
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from pydantic import BaseModel, Field

from .base import RefinementStrategy, StrategyMetadata
from photo_types import BGRImage, DirectoryPath, QuadArray

class OriginalRefinementConfig(BaseModel):
    """Configuration for original refinement strategy."""
    resolution: int = Field(200, ge=50, le=1000, description="Working resolution")
    enforce_parallel_sides: bool = Field(True, description="Enforce parallel sides")
    border_width: int = Field(50, ge=10, le=200, description="Border search width")
    edge_threshold: float = Field(0.1, ge=0.0, le=1.0, description="Edge detection threshold")

class OriginalRefinementStrategy(RefinementStrategy):
    """Original refinement strategy with configurable resolution."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._config = OriginalRefinementConfig(**self.config)
    
    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="original",
            display_name=f"Original ({self._config.resolution}px)",
            description="Edge-based refinement at fixed resolution",
            category="edge_detection",
            version="1.0.0",
            author="Photo Extractor Team"
        )
    
    def _validate_config(self) -> None:
        # Validation handled by Pydantic in __init__
        pass
    
    def get_config_schema(self) -> Dict[str, Any]:
        return OriginalRefinementConfig.schema()
    
    def refine_boundary(
        self, 
        image: BGRImage, 
        initial_quad: QuadArray,
        debug_dir: Optional[DirectoryPath] = None
    ) -> QuadArray:
        """Refine boundary using original edge detection algorithm."""
        # Import here to avoid circular imports
        from image_processing.refine_bounds import refine_bounding_box
        
        return refine_bounding_box(
            image=image,
            corner_points=initial_quad,
            resolution=self._config.resolution,
            enforce_parallel_sides=self._config.enforce_parallel_sides,
            border_width=self._config.border_width,
            debug_dir=debug_dir
        )
```

### Strategy Registry

```python
# image_processing/strategies/registry.py
from typing import Dict, List, Optional, Type

from .base import RefinementStrategy, StrategyMetadata

class StrategyRegistry:
    """Registry for refinement strategies."""
    
    def __init__(self):
        self._strategies: Dict[str, Type[RefinementStrategy]] = {}
        self._instances: Dict[str, RefinementStrategy] = {}
    
    def register(
        self, 
        strategy_class: Type[RefinementStrategy],
        override: bool = False
    ) -> None:
        """Register a strategy class."""
        # Create temporary instance to get metadata
        temp_instance = strategy_class()
        name = temp_instance.metadata.name
        
        if name in self._strategies and not override:
            raise ValueError(f"Strategy '{name}' already registered")
        
        self._strategies[name] = strategy_class
    
    def unregister(self, name: str) -> None:
        """Unregister a strategy."""
        self._strategies.pop(name, None)
        self._instances.pop(name, None)
    
    def get(self, name: str, config: Optional[Dict[str, Any]] = None) -> RefinementStrategy:
        """Get a strategy instance."""
        if name not in self._strategies:
            raise ValueError(f"Unknown strategy: {name}")
        
        # Create new instance with config
        return self._strategies[name](config)
    
    def get_cached(self, name: str, config: Optional[Dict[str, Any]] = None) -> RefinementStrategy:
        """Get a cached strategy instance."""
        cache_key = f"{name}:{hash(frozenset(config.items()) if config else 0)}"
        
        if cache_key not in self._instances:
            self._instances[cache_key] = self.get(name, config)
        
        return self._instances[cache_key]
    
    def list_strategies(self, category: Optional[str] = None) -> List[StrategyMetadata]:
        """List all available strategies."""
        strategies = []
        
        for strategy_class in self._strategies.values():
            instance = strategy_class()
            metadata = instance.metadata
            
            if category is None or metadata.category == category:
                strategies.append(metadata)
        
        return sorted(strategies, key=lambda m: m.display_name)
    
    def get_config_schema(self, name: str) -> Dict[str, Any]:
        """Get configuration schema for a strategy."""
        if name not in self._strategies:
            raise ValueError(f"Unknown strategy: {name}")
        
        instance = self._strategies[name]()
        return instance.get_config_schema()

# Global registry instance
refinement_registry = StrategyRegistry()

def get_refinement_strategy(name: str, config: Optional[Dict[str, Any]] = None) -> RefinementStrategy:
    """Get a refinement strategy by name."""
    return refinement_registry.get(name, config)
```

### Auto-Registration with Decorators

```python
# image_processing/strategies/decorators.py
from typing import Type

from .base import RefinementStrategy
from .registry import refinement_registry

def register_strategy(name: Optional[str] = None):
    """Decorator to register a strategy class."""
    def decorator(cls: Type[RefinementStrategy]) -> Type[RefinementStrategy]:
        refinement_registry.register(cls)
        return cls
    return decorator

# Usage in strategy files
@register_strategy()
class StripsMultiscaleStrategy(RefinementStrategy):
    # ... implementation
```

### Strategy Discovery

```python
# image_processing/strategies/discovery.py
import importlib
import pkgutil
from pathlib import Path
from typing import List

from .base import RefinementStrategy
from .registry import refinement_registry

def discover_strategies(package_path: Path) -> List[str]:
    """Discover and register all strategies in a package."""
    discovered = []
    
    # Import all modules in strategies package
    package = importlib.import_module(package_path.name)
    
    for importer, modname, ispkg in pkgutil.iter_modules(
        package.__path__, 
        prefix=package.__name__ + "."
    ):
        if not ispkg and not modname.endswith('base'):
            try:
                importlib.import_module(modname)
                discovered.append(modname)
            except Exception as e:
                logger.warning(f"Failed to import strategy module {modname}: {e}")
    
    return discovered

def auto_discover_strategies() -> None:
    """Automatically discover and register all built-in strategies."""
    strategies_path = Path(__file__).parent
    discover_strategies(strategies_path)
```

## Migration Plan

### Phase 1: Create Strategy Infrastructure
1. Implement base strategy class and metadata
2. Create strategy registry with registration
3. Add configuration validation with Pydantic
4. Write comprehensive unit tests

### Phase 2: Refactor Existing Strategies
1. Convert each lambda to a proper strategy class:
   ```python
   # Before
   "Original (200px res)": lambda image, corners, debug_dir=None: ...
   
   # After
   class OriginalStrategy200(OriginalRefinementStrategy):
       def __init__(self):
           super().__init__({"resolution": 200})
   ```

2. Maintain backward compatibility:
   ```python
   # Compatibility layer
   def get_legacy_strategies() -> Dict[str, Callable]:
       """Get strategies in legacy format."""
       legacy = {}
       for metadata in refinement_registry.list_strategies():
           strategy = refinement_registry.get(metadata.name)
           legacy[metadata.display_name] = strategy.refine_boundary
       return legacy
   ```

### Phase 3: Update UI Integration
1. Modify strategy selection to use registry:
   ```python
   # In GUI
   strategies = refinement_registry.list_strategies()
   for strategy in strategies:
       combo_box.addItem(strategy.display_name, strategy.name)
   ```

2. Add configuration UI:
   ```python
   def show_strategy_config(self, strategy_name: str):
       schema = refinement_registry.get_config_schema(strategy_name)
       dialog = ConfigDialog(schema)
       if dialog.exec():
           config = dialog.get_config()
           self.current_strategy = refinement_registry.get(strategy_name, config)
   ```

### Phase 4: Enable Plugin Support
1. Create plugin interface:
   ```python
   # plugins/example_strategy.py
   from photo_extractor.strategies import RefinementStrategy, register_strategy
   
   @register_strategy()
   class CustomStrategy(RefinementStrategy):
       # ... implementation
   ```

2. Add plugin loading:
   ```python
   def load_plugins(plugin_dir: Path):
       """Load strategy plugins from directory."""
       sys.path.insert(0, str(plugin_dir))
       discover_strategies(plugin_dir)
   ```

## Testing Improvements

### Unit Testing
```python
def test_strategy_registration():
    """Test strategy registration."""
    registry = StrategyRegistry()
    
    # Register strategy
    registry.register(OriginalRefinementStrategy)
    
    # Get strategy
    strategy = registry.get("original", {"resolution": 300})
    assert isinstance(strategy, OriginalRefinementStrategy)
    assert strategy._config.resolution == 300

def test_strategy_implementation():
    """Test individual strategy."""
    strategy = OriginalRefinementStrategy({"resolution": 100})
    
    # Create test image and quad
    image = np.zeros((500, 500, 3), dtype=np.uint8)
    quad = np.array([[10, 10], [100, 10], [100, 100], [10, 100]])
    
    # Test refinement
    refined = strategy.refine_boundary(image, quad)
    assert refined.shape == (4, 2)
```

### Integration Testing
```python
def test_strategy_with_gui(qtbot):
    """Test strategy integration with GUI."""
    window = MainWindow()
    qtbot.addWidget(window)
    
    # Select strategy
    window.refinement_combo.setCurrentText("Original (200px)")
    
    # Verify strategy loaded
    assert window.current_strategy.metadata.name == "original"
    assert window.current_strategy._config.resolution == 200
```

## Documentation

### Strategy Development Guide
```markdown
# Creating a Custom Refinement Strategy

1. Create a new file in `image_processing/strategies/`
2. Inherit from `RefinementStrategy`
3. Implement required methods
4. Register with `@register_strategy()`
5. Add tests in `tests/strategies/`

Example:
```python
from photo_extractor.strategies import RefinementStrategy, register_strategy

@register_strategy()
class MyCustomStrategy(RefinementStrategy):
    @property
    def metadata(self):
        return StrategyMetadata(
            name="my_custom",
            display_name="My Custom Strategy",
            description="A custom refinement approach",
            category="custom",
            version="1.0.0"
        )
    
    def refine_boundary(self, image, initial_quad, debug_dir=None):
        # Implementation here
        return refined_quad
```

### API Documentation
- Auto-generate from docstrings and type hints
- Include configuration schemas
- Provide example usage

## Benefits

1. **Type Safety**: Full protocol implementation
2. **Testability**: Easy to test individual strategies
3. **Extensibility**: Clean plugin architecture
4. **Configuration**: Integrated with config system
5. **Documentation**: Self-documenting strategies
6. **Debugging**: Clear class names in stack traces
7. **Performance**: Optional caching of instances

## Performance Considerations

- Lazy import of strategy implementations
- Cache strategy instances when using same config
- Benchmark strategy initialization overhead
- Profile memory usage of strategy instances

## Success Metrics

- 100% type coverage for strategies
- All strategies have unit tests
- Configuration schemas for all strategies
- Plugin loading mechanism working
- No performance regression

## Timeline

- Week 1: Base infrastructure and registry
- Week 2: Refactor existing strategies
- Week 3: UI integration updates
- Week 4: Plugin support and discovery
- Week 5: Documentation and examples