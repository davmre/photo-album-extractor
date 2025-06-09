# Configuration Centralization Plan

## Overview

This document outlines the plan to centralize all configuration in the Photo Album Extractor, including user preferences, algorithm parameters, API keys, and runtime settings. The goal is to create a unified, type-safe, and hierarchical configuration system.

## Current State Problems

1. **Scattered Configuration**: Settings spread across multiple locations
2. **Hard-coded Values**: Algorithm parameters embedded in code
3. **No Validation**: Settings can have invalid values
4. **Poor Discoverability**: Difficult to know what can be configured
5. **No Hierarchy**: Can't override settings at different levels

## Proposed Architecture

### Configuration Hierarchy

```
Default Config (built-in)
    ↓
System Config (/etc/photo_extractor/)
    ↓
User Config (~/.config/photo_extractor/)
    ↓
Project Config (./.photo_extractor.yml)
    ↓
Environment Variables (PHOTO_EXTRACTOR_*)
    ↓
Command Line Arguments
    ↓
Runtime Updates (GUI settings)
```

### Configuration Structure

```yaml
# Example configuration file
version: "1.0"

ui:
  theme: "light"
  language: "en"
  magnifier:
    enabled: true
    zoom_level: 3.0
    size: 200
  
storage:
  cache_dir: "~/.cache/photo_extractor"
  auto_save: true
  save_interval_seconds: 30
  
algorithms:
  detection:
    default_strategy: "gemini"
    strategies:
      gemini:
        api_key: "${GEMINI_API_KEY}"  # Environment variable reference
        model: "gemini-2.5-flash-preview-05-20"
        timeout_seconds: 30
        resize_dimension: 768
        
  refinement:
    default_strategy: "strips_multiscale"
    strategies:
      original:
        resolution: 200
        enforce_parallel_sides: true
        border_width: 50
      strips_multiscale:
        scale_factors: [1.0, 0.5, 0.25]
        strip_width: 100
        enforce_parallel_sides: true
        edge_threshold: 0.1
        
  extraction:
    output_format: "jpeg"
    jpeg_quality: 95
    preserve_metadata: true
    auto_rotate: true
    
logging:
  level: "INFO"
  file: "~/.logs/photo_extractor/app.log"
  max_size_mb: 10
  backup_count: 5
  modules:
    "photo_extractor.algorithms": "DEBUG"
    "PIL": "WARNING"
    
development:
  debug_mode: false
  profile_performance: false
  save_debug_images: false
  debug_output_dir: "./debug"
```

## Implementation

### 1. Configuration Schema (`core/config/schema.py`)

```python
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

class Theme(str, Enum):
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"

class MagnifierConfig(BaseModel):
    enabled: bool = True
    zoom_level: float = Field(3.0, ge=1.0, le=10.0)
    size: int = Field(200, ge=100, le=500)

class UIConfig(BaseModel):
    theme: Theme = Theme.LIGHT
    language: str = "en"
    magnifier: MagnifierConfig = MagnifierConfig()

class DetectionStrategyConfig(BaseModel):
    """Base config for detection strategies"""
    class Config:
        extra = "allow"  # Allow strategy-specific fields

class GeminiConfig(DetectionStrategyConfig):
    api_key: Optional[str] = None
    model: str = "gemini-2.5-flash-preview-05-20"
    timeout_seconds: int = 30
    resize_dimension: int = 768
    
    @validator("api_key")
    def validate_api_key(cls, v):
        if v and not v.startswith("${") and len(v) < 20:
            raise ValueError("Invalid API key format")
        return v

class RefinementStrategyConfig(BaseModel):
    """Base config for refinement strategies"""
    enforce_parallel_sides: bool = True
    
    class Config:
        extra = "allow"

class OriginalRefinementConfig(RefinementStrategyConfig):
    resolution: int = Field(200, ge=50, le=1000)
    border_width: int = Field(50, ge=10, le=200)

class StripsRefinementConfig(RefinementStrategyConfig):
    scale_factors: List[float] = [1.0, 0.5, 0.25]
    strip_width: int = Field(100, ge=50, le=500)
    edge_threshold: float = Field(0.1, ge=0.0, le=1.0)

class AlgorithmConfig(BaseModel):
    detection: Dict[str, Any] = Field(default_factory=dict)
    refinement: Dict[str, Any] = Field(default_factory=dict)
    extraction: Dict[str, Any] = Field(default_factory=dict)

class Config(BaseModel):
    """Root configuration model"""
    version: str = "1.0"
    ui: UIConfig = UIConfig()
    storage: StorageConfig = StorageConfig()
    algorithms: AlgorithmConfig = AlgorithmConfig()
    logging: LoggingConfig = LoggingConfig()
    development: DevelopmentConfig = DevelopmentConfig()
    
    class Config:
        validate_assignment = True  # Validate on field updates
```

### 2. Configuration Manager (`core/config/manager.py`)

```python
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from .schema import Config
from .sources import (
    ConfigSource,
    DefaultsSource,
    YamlFileSource,
    EnvironmentSource,
    CommandLineSource,
    RuntimeSource
)

T = TypeVar('T', bound=BaseModel)

class ConfigManager:
    """Manages hierarchical configuration with validation."""
    
    def __init__(self):
        self._sources: List[ConfigSource] = []
        self._config: Optional[Config] = None
        self._runtime_overrides: Dict[str, Any] = {}
        self._change_listeners: List[Callable[[str, Any], None]] = []
        
    def initialize(
        self,
        system_config_path: Optional[Path] = None,
        user_config_path: Optional[Path] = None,
        project_config_path: Optional[Path] = None,
        env_prefix: str = "PHOTO_EXTRACTOR_",
        cli_args: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize configuration from multiple sources."""
        
        # Add sources in priority order (lowest to highest)
        self._sources = [
            DefaultsSource(),
            YamlFileSource(system_config_path) if system_config_path else None,
            YamlFileSource(user_config_path) if user_config_path else None,
            YamlFileSource(project_config_path) if project_config_path else None,
            EnvironmentSource(prefix=env_prefix),
            CommandLineSource(cli_args or {}),
            RuntimeSource(self._runtime_overrides)
        ]
        
        # Filter out None sources
        self._sources = [s for s in self._sources if s is not None]
        
        # Load and validate
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from all sources."""
        merged_dict = {}
        
        for source in self._sources:
            try:
                source_dict = source.load()
                merged_dict = deep_merge(merged_dict, source_dict)
            except Exception as e:
                logger.warning(f"Failed to load config from {source}: {e}")
        
        # Resolve environment variables
        merged_dict = self._resolve_env_vars(merged_dict)
        
        # Validate with Pydantic
        try:
            self._config = Config(**merged_dict)
        except ValidationError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    def _resolve_env_vars(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve ${ENV_VAR} references in configuration."""
        def resolve_value(value: Any) -> Any:
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                return os.environ.get(env_var, value)
            elif isinstance(value, dict):
                return {k: resolve_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_value(v) for v in value]
            return value
        
        return resolve_value(config_dict)
    
    def get(self, path: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated path."""
        if not self._config:
            return default
            
        try:
            value = self._config
            for part in path.split('.'):
                if hasattr(value, part):
                    value = getattr(value, part)
                elif isinstance(value, dict):
                    value = value[part]
                else:
                    return default
            return value
        except (AttributeError, KeyError):
            return default
    
    def get_typed(self, path: str, type_: Type[T]) -> Optional[T]:
        """Get typed configuration section."""
        value = self.get(path)
        if value is None:
            return None
            
        try:
            if isinstance(value, dict):
                return type_(**value)
            return value if isinstance(value, type_) else None
        except ValidationError:
            return None
    
    def set(self, path: str, value: Any, persist: bool = False) -> None:
        """Set configuration value at runtime."""
        self._runtime_overrides[path] = value
        self._load_config()  # Reload to validate
        
        # Notify listeners
        for listener in self._change_listeners:
            listener(path, value)
        
        if persist:
            self._persist_user_config()
    
    def watch(self, callback: Callable[[str, Any], None]) -> None:
        """Register callback for configuration changes."""
        self._change_listeners.append(callback)
    
    def export_schema(self) -> Dict[str, Any]:
        """Export JSON schema for configuration."""
        return Config.schema()
    
    def validate_file(self, file_path: Path) -> List[str]:
        """Validate a configuration file and return errors."""
        try:
            with open(file_path) as f:
                data = yaml.safe_load(f)
            Config(**data)
            return []
        except ValidationError as e:
            return [f"{err['loc']}: {err['msg']}" for err in e.errors()]
        except Exception as e:
            return [str(e)]

# Global configuration instance
config = ConfigManager()

def get_config() -> Config:
    """Get current configuration."""
    return config._config

def get_algorithm_config(algorithm_type: str, strategy: str) -> Dict[str, Any]:
    """Get configuration for specific algorithm strategy."""
    return config.get(f"algorithms.{algorithm_type}.strategies.{strategy}", {})
```

### 3. Strategy Integration

```python
# image_processing/refine_bounds.py
from core.config import get_algorithm_config

def create_refinement_strategy(name: str) -> BoundaryRefinementStrategy:
    """Create refinement strategy with configuration."""
    config = get_algorithm_config("refinement", name)
    
    if name == "original":
        return OriginalRefinement(
            resolution=config.get("resolution", 200),
            enforce_parallel_sides=config.get("enforce_parallel_sides", True),
            border_width=config.get("border_width", 50)
        )
    elif name == "strips_multiscale":
        return StripsMultiscaleRefinement(
            scale_factors=config.get("scale_factors", [1.0, 0.5, 0.25]),
            strip_width=config.get("strip_width", 100),
            enforce_parallel_sides=config.get("enforce_parallel_sides", True),
            edge_threshold=config.get("edge_threshold", 0.1)
        )
    # ... other strategies
```

### 4. GUI Integration

```python
# gui/settings_dialog.py
class SettingsDialog(QDialog):
    def __init__(self, config_manager: ConfigManager):
        super().__init__()
        self.config_manager = config_manager
        self._init_ui()
        self._load_current_settings()
    
    def _create_algorithm_tab(self):
        """Create algorithm configuration tab."""
        # Generate UI from configuration schema
        schema = self.config_manager.export_schema()
        algorithm_schema = schema["properties"]["algorithms"]
        
        # Create form fields dynamically
        for category, cat_schema in algorithm_schema["properties"].items():
            group = QGroupBox(category.title())
            layout = QFormLayout()
            
            # Add controls based on schema
            self._add_schema_controls(layout, cat_schema, f"algorithms.{category}")
            
            group.setLayout(layout)
            self.algorithm_layout.addWidget(group)
    
    def save_settings(self):
        """Save settings to user configuration."""
        # Collect all values from UI
        for path, widget in self.config_widgets.items():
            value = self._get_widget_value(widget)
            self.config_manager.set(path, value, persist=True)
```

## Migration Plan

### Phase 1: Create Configuration Infrastructure
1. Implement configuration schema with Pydantic
2. Create ConfigManager class
3. Add configuration sources (file, env, CLI)
4. Write comprehensive tests

### Phase 2: Extract Hard-coded Values
1. Identify all hard-coded configuration
2. Move to default configuration
3. Update code to use config manager
4. Document all configuration options

### Phase 3: Integrate with Existing Code
1. Update strategy registries to use configuration
2. Modify GUI settings to use config manager
3. Add configuration validation on startup
4. Create configuration migration tool

### Phase 4: User Experience
1. Add configuration wizard for first run
2. Create configuration documentation
3. Add config validation CLI command
4. Implement configuration export/import

## Configuration Discovery

### CLI Commands
```bash
# Validate configuration
photo-extractor config validate

# Show current configuration
photo-extractor config show

# Show configuration schema
photo-extractor config schema

# Export configuration
photo-extractor config export > my-config.yml

# List all available options
photo-extractor config list --descriptions
```

### GUI Configuration Browser
- Tree view of all configuration options
- Inline documentation
- Validation feedback
- Reset to defaults option

## Testing Strategy

### Unit Tests
```python
def test_config_validation():
    """Test configuration validation."""
    # Valid config
    config = Config(
        algorithms=AlgorithmConfig(
            refinement={
                "original": {
                    "resolution": 200,
                    "enforce_parallel_sides": True
                }
            }
        )
    )
    assert config.algorithms.refinement["original"]["resolution"] == 200
    
    # Invalid config
    with pytest.raises(ValidationError):
        Config(
            algorithms=AlgorithmConfig(
                refinement={
                    "original": {
                        "resolution": -100  # Invalid
                    }
                }
            )
        )

def test_config_hierarchy():
    """Test configuration source hierarchy."""
    manager = ConfigManager()
    
    # Add sources in order
    manager.add_source(DefaultsSource({"ui": {"theme": "light"}}))
    manager.add_source(YamlFileSource({"ui": {"theme": "dark"}}))
    manager.add_source(EnvironmentSource({"UI_THEME": "auto"}))
    
    manager.initialize()
    
    # Environment should win
    assert manager.get("ui.theme") == "auto"
```

## Benefits

1. **Discoverability**: All configuration in one place
2. **Type Safety**: Pydantic validation prevents errors
3. **Flexibility**: Multiple configuration sources
4. **Testability**: Easy to mock configuration
5. **Documentation**: Schema serves as documentation
6. **User-Friendly**: GUI for non-technical users

## Performance Considerations

- Lazy loading for large configurations
- Caching of parsed configuration
- Minimal overhead for config access
- Watch for file changes efficiently

## Security Considerations

- Sensitive values (API keys) in separate files
- Environment variable support for secrets
- No secrets in version control
- Encrypted storage option for sensitive data

## Success Metrics

- Zero hard-coded configuration values
- All algorithms parameterized
- Configuration validation on startup
- User-modifiable settings persist correctly
- Clear documentation for all options

## Timeline

- Week 1: Schema definition and ConfigManager
- Week 2: Extract hard-coded values
- Week 3: Integration with existing code
- Week 4: GUI configuration interface
- Week 5: Documentation and testing