# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Photo Album Extractor is a PyQt6-based GUI application for extracting individual photos from a directory of scanned album pages. It supports perspective correction, AI-based photo detection, boundary refinement, and EXIF metadata preservation.

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

All new code should have type annotations. Define new semantic types as appropriate, putting any globally-useful types in `core/photo_types.py`. Validate with `pyright` and `ruff` frequently.

Test-driven development: before adding new functionality, consider what semantics any new methods or classes should satisfy. If appropriate, offer to write tests, and check the implementation against these tests. Prefer a test-driven methodology especially for core methods that can be easily tested in isolation.

### Key Components

1. **Core app functionality** (`core/`):
   - `bounding_box.py`: core data model for a bounding box / photo to be extracted.
   - `photo_types.py` and `errors.py`: custom types and exceptions used throughout
   - `bounding_box_storage.py` - Loads and stores bounding boxes as JSON in `.photo_extractor_data.json` files per directory, and functions as a global registry of bounding boxes.
   - `images.py`: Handles perspective correction, EXIF metadata, and image saving
   - `settings.py`: Defines app configuration.
   - `detection_strategies.py`: Multiple photo detection strategies including Gemini AI
   - `refinement_strategies.py`: Multiple algorithms for refining bounding boxes.

2. **GUI Layer** (`gui/`): PyQt6-based interface
   - `main_window.py`: Main application window orchestrating all components
   - `image_view.py`: Custom QGraphicsView handling image display and bounding box interactions
   - `quad_bounding_box.py`: Quadrilateral (4-point) bounding box implementation - NOT restricted to rectangles

3. **CLI Layer** (`cli/`): command-line interface
   - Commands `info`, `detect`, `extract`, `refine`, etc. operate on single images
   or batches of images, using saved bounding-box data.
