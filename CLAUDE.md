# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Photo Album Extractor is a PyQt6-based GUI application for extracting individual photos from scanned album pages. It supports perspective correction, AI-based photo detection, boundary refinement, and EXIF metadata preservation.

## Development Commands

### Environment

Use the virtual environment under `.venv`: run `source .venv/bin/activate` 
before any other Python commands.

### Running

```bash
# Launch GUI mode
python3 main.py gui
```

### Code Formatting

Format code with Ruff:

```bash
ruff format .
```

Check for linting issues:

```bash
ruff check .
```

Fix auto-fixable linting issues:

```bash
ruff check . --fix
```

### Type Checking Setup

```bash
# Always validate types before committing
source .venv/bin/activate
pyright  # Should show 0 errors, 0 warnings
```

The codebase uses `pyproject.toml` for pyright configuration with gradual typing enabled.


### Testing

To run a quick integration test of much of the main workflow (opening the app,
loading an image with saved bounding boxes, refining the boxes, extracting
photos to a file):

```
pytest tests/test_workflow.py
```

This should pass. To run all tests (some may not be passing currently):

```
pytest tests
```

## Development principles

Prefer simple designs with pure functions operating on well-defined data types when
possible. Avoid boilerplate OOP structures.

For new code, always consider the data model first. Factor out fundamental structures
and logic under `core`, and then add interfaces under `gui` and/or `cli` as appropriate.

All new code should have type annotations. Use `from __future__ import annotations`
in new files for deferred evaluation of type annotations.


## Architecture

### Key Components

1. **Core app functionality** (`core/`):
   - `photo_types.py` and `errors.py`: custom types and exceptions used throughout
   - `bounding_box_storage.py` - Saves bounding boxes as JSON in `.photo_extractor_data.json` files per directory
   - `images.py`: Handles perspective correction, EXIF metadata, and image saving
   - `settings.py`: Defines app configuration.

2. **GUI Layer** (`gui/`): PyQt6-based interface
   - `main_window.py`: Main application window orchestrating all components
   - `image_view.py`: Custom QGraphicsView handling image display and bounding box interactions
   - `quad_bounding_box.py`: Quadrilateral (4-point) bounding box implementation - NOT restricted to rectangles

3. **CLI Layer** (`cli/`): command-line interface
   - Commands `info`, `detect`, `extract`, etc. operate on single images or batches of
     images, using saved bounding-box data.

4. **Photo detection** (`photo_detection/`): Methods to detect and refine photo bounding boxes
   - `detection_strategies.py`: Multiple photo detection strategies including Gemini AI
   - `refinement_strategies.py`: Multiple algorithms for refining bounding boxes.

### File Organization

**Type-First Development**: When adding new modules:

1. Start with type definitions in appropriate module
2. Import semantic types from `core/photo_types.py`
3. Add proper type annotations from the beginning
4. Validate with `pyright` and `ruff` frequently
