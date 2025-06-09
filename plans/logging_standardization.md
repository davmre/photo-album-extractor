# Logging Standardization Plan

## Overview

This document outlines the plan to implement consistent, structured logging throughout the Photo Album Extractor application, replacing ad-hoc print statements and silent error handling with a robust logging framework.

## Goals

1. **Consistency**: Uniform logging patterns across all modules
2. **Debuggability**: Rich context for troubleshooting issues
3. **Performance**: Minimal overhead in production
4. **Flexibility**: Easy to adjust verbosity and outputs
5. **Structured Data**: Machine-readable logs for analysis

## Technology Choice

**Primary Framework**: Python's built-in `logging` module with `structlog` enhancement

**Rationale**:
- `logging` is standard and well-understood
- `structlog` adds structured logging without replacing `logging`
- JSON output support for log aggregation
- Minimal dependencies
- Good PyQt integration

## Architecture

### Logging Hierarchy

```
photo_extractor (root logger)
├── photo_extractor.gui
│   ├── photo_extractor.gui.main_window
│   ├── photo_extractor.gui.image_view
│   └── photo_extractor.gui.dialogs
├── photo_extractor.core
│   ├── photo_extractor.core.models
│   └── photo_extractor.core.services
├── photo_extractor.image_processing
│   ├── photo_extractor.image_processing.detection
│   └── photo_extractor.image_processing.refinement
└── photo_extractor.storage
```

### Log Levels Usage

| Level | Usage | Example |
|-------|-------|---------|
| DEBUG | Detailed execution flow | "Refining boundary: initial_corners=[...]" |
| INFO | Normal operations | "Loaded image: photo.jpg" |
| WARNING | Recoverable issues | "EXIF date parsing failed, using current date" |
| ERROR | Operation failures | "Failed to save bounding boxes: Permission denied" |
| CRITICAL | Application-breaking | "Required dependency missing: cv2" |

## Implementation

### 1. Core Logging Module (`core/logging.py`)

```python
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
from structlog.processors import JSONRenderer, TimeStamper, add_log_level
from structlog.stdlib import LoggerFactory, ProcessorFormatter

def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    json_output: bool = False,
    gui_handler: Optional[logging.Handler] = None
) -> None:
    """Configure application-wide logging."""
    
    # Configure structlog
    processors = [
        TimeStamper(fmt="iso"),
        add_log_level,
        structlog.processors.add_logger_name,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    if json_output:
        processors.append(JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer()
    ))
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10_000_000,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(ProcessorFormatter(
            processor=JSONRenderer() if json_output else structlog.dev.ConsoleRenderer()
        ))
        root_logger.addHandler(file_handler)
    
    # GUI handler if provided (for status bar updates)
    if gui_handler:
        gui_handler.setLevel(logging.INFO)
        root_logger.addHandler(gui_handler)

def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger for a module."""
    return structlog.get_logger(name)

class LogContext:
    """Context manager for temporary log context."""
    
    def __init__(self, logger: structlog.BoundLogger, **kwargs: Any):
        self.logger = logger
        self.context = kwargs
        self.token: Optional[Any] = None
    
    def __enter__(self) -> structlog.BoundLogger:
        self.token = structlog.contextvars.bind_contextvars(**self.context)
        return self.logger
    
    def __exit__(self, *args: Any) -> None:
        if self.token:
            structlog.contextvars.unbind_contextvars(self.token)
```

### 2. Logger Usage Patterns

#### Basic Usage
```python
from photo_extractor.core.logging import get_logger

logger = get_logger(__name__)

class PhotoExtractor:
    def extract_photo(self, album_path: str, box_id: str) -> Photo:
        logger.info("extracting_photo", album_path=album_path, box_id=box_id)
        
        try:
            photo = self._do_extraction(album_path, box_id)
            logger.debug("extraction_complete", 
                        photo_id=photo.id, 
                        dimensions=(photo.width, photo.height))
            return photo
        except Exception as e:
            logger.error("extraction_failed", 
                        album_path=album_path, 
                        box_id=box_id,
                        error=str(e),
                        exc_info=True)
            raise
```

#### Context-Rich Logging
```python
def process_album(self, album: Album) -> None:
    with LogContext(logger, album_id=album.id, album_path=album.path):
        logger.info("processing_album")
        
        for box in album.bounding_boxes:
            with LogContext(logger, box_id=box.id):
                logger.debug("processing_box", corners=box.corners.tolist())
                self._process_box(box)
```

#### Performance Logging
```python
import time

def detect_photos(self, image: Image) -> List[BoundingBox]:
    start_time = time.time()
    logger.info("detection_started", strategy=self.strategy.name)
    
    try:
        boxes = self.strategy.detect(image)
        duration = time.time() - start_time
        
        logger.info("detection_completed",
                   strategy=self.strategy.name,
                   duration_seconds=duration,
                   boxes_found=len(boxes))
        return boxes
    except Exception as e:
        duration = time.time() - start_time
        logger.error("detection_failed",
                    strategy=self.strategy.name,
                    duration_seconds=duration,
                    error=str(e))
        raise
```

### 3. GUI Integration

```python
class GuiLogHandler(logging.Handler):
    """Handler that emits log records to GUI status bar."""
    
    def __init__(self, status_bar: QStatusBar):
        super().__init__()
        self.status_bar = status_bar
        
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            # Show INFO and above in status bar
            if record.levelno >= logging.INFO:
                self.status_bar.showMessage(msg, 5000)  # 5 second timeout
        except Exception:
            self.handleError(record)
```

### 4. Configuration Integration

```python
# In settings.py
@dataclass
class LoggingSettings:
    level: str = "INFO"
    file_path: Optional[str] = None
    json_output: bool = False
    gui_logging: bool = True
    
    # Per-module overrides
    module_levels: Dict[str, str] = field(default_factory=lambda: {
        "photo_extractor.image_processing": "DEBUG",
        "PIL": "WARNING",  # Quiet noisy libraries
    })
```

## Migration Plan

### Phase 1: Infrastructure Setup
1. Add structlog to requirements.txt
2. Create `core/logging.py` module
3. Add logging configuration to settings
4. Update `main.py` to initialize logging

### Phase 2: Replace Print Statements
Use automated script to convert print statements:

```python
# migration/convert_prints.py
import ast
import astor

class PrintToLoggerTransformer(ast.NodeTransformer):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            if node.value.func.id == 'print':
                # Convert print(...) to logger.info(...)
                return self._convert_print_to_log(node)
        return node
    
    def _convert_print_to_log(self, node):
        # Transform print statement to appropriate log level
        # Analysis of message content to determine level
        pass
```

### Phase 3: Add Structured Context

Enhance existing logging with structured data:

Before:
```python
print(f"Saved: {filename}")
```

After:
```python
logger.info("file_saved", filename=filename, size_bytes=file_size)
```

### Phase 4: Error Handling Enhancement

Replace silent catches with proper logging:

Before:
```python
try:
    result = risky_operation()
except Exception:
    return None
```

After:
```python
try:
    result = risky_operation()
except Exception as e:
    logger.error("risky_operation_failed", 
                error=str(e), 
                exc_info=True)
    return None
```

## Testing

### Unit Test Support
```python
import logging
from unittest.mock import Mock

def test_with_log_capture(caplog):
    """Test using pytest's caplog fixture."""
    with caplog.at_level(logging.DEBUG):
        my_function()
    
    assert "processing_started" in caplog.text
    assert caplog.records[0].levelno == logging.INFO
```

### Log Assertion Helpers
```python
class LogTesting:
    @staticmethod
    def assert_logged(caplog, level: str, message: str, **fields):
        """Assert structured log entry exists."""
        for record in caplog.records:
            if record.levelname == level and message in record.message:
                # Check structured fields
                for key, value in fields.items():
                    assert getattr(record, key, None) == value
                return
        raise AssertionError(f"Log entry not found: {level} {message}")
```

## Monitoring & Analysis

### Log Aggregation
- JSON format enables easy parsing
- Can integrate with ELK stack or similar
- Structured fields enable powerful queries

### Example Queries
```sql
-- Find slow operations
SELECT * FROM logs 
WHERE event = 'operation_completed' 
AND duration_seconds > 5.0

-- Error analysis
SELECT error, COUNT(*) as count 
FROM logs 
WHERE level = 'ERROR' 
GROUP BY error 
ORDER BY count DESC
```

## Best Practices

### DO:
- Use structured fields for data (not string formatting)
- Include relevant context (IDs, paths, sizes)
- Log at appropriate levels
- Use consistent event names (snake_case)
- Include timing for performance-critical operations

### DON'T:
- Log sensitive data (passwords, API keys)
- Use f-strings for structured data
- Catch and suppress exceptions without logging
- Log in tight loops without rate limiting
- Use different formats in same module

## Performance Considerations

- Lazy evaluation for expensive operations:
  ```python
  logger.debug("data_dump", data=lambda: expensive_serialization())
  ```
- Rate limiting for high-frequency events
- Async logging for I/O intensive applications
- Conditional logging based on level

## Success Metrics

- Zero print statements in codebase
- 100% of errors logged with context
- Structured data for all INFO+ logs
- Sub-millisecond logging overhead
- Actionable error messages

## Timeline

- Week 1: Infrastructure setup and core module
- Week 2: Automated migration of print statements
- Week 3: Manual enhancement of error handling
- Week 4: Testing and documentation
- Week 5: Monitoring setup and training