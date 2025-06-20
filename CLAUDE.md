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
# Basic launch
python3 main.py
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

Before any sizable change, write a plan and save it under
`plans/<change_name>.md`. Refer to the plan as you work and update it
at the end of each phase with the current status, any issues encountered and if
or how they were resolved.

Remember that you are an AI language model, writing for yourself. You
do not need to include schedule estimates since you don't work on a human
schedule (though it is fine to assess difficulty of specific tasks if that is
helpful for your own planning). You do want to be thoughtful in recording
context and any changes to your understanding of the task, since
you have no other means of persisting memory from session to session.

## Architecture

### Key Components

1. **Core app functionality** (`core/`):
   - `bounding_box_storage.py` - Saves bounding boxes as JSON in `.photo_extractor_data.json` files per directory
   - `images.py`: Handles perspective correction, EXIF metadata, and image saving
   - `settings.py`: Defines app configuration.
   - `photo_types.py` and `errors.py`: custom types and exceptions used throughout

2. **GUI Layer** (`gui/`): PyQt6-based interface
   - `main_window.py`: Main application window orchestrating all components
   - `image_view.py`: Custom QGraphicsView handling image display and bounding box interactions
   - `quad_bounding_box.py`: Quadrilateral (4-point) bounding box implementation - NOT restricted to rectangles

3. **CLI Layer** (`cli/`): command-line interface
   - Commands `info`, `detect`, `extract`, etc. operate on single images or batches of
     images, using saved bounding-box data.

4. **Photo detection** (`photo_detection/`): Methods to detect and refine photo bounding boxes
   - `detection_strategies.py`: Multiple photo detection strategies including Gemini AI
   - `refine_bounds.py`: Multiple refinement algorithms (original, multiscale, strips-based)

### Important Patterns

- **Quadrilateral Support**: Bounding boxes are 4-point quadrilaterals. These are
  typically represented as a Numpy array of floats in image coordinate space
  (`QuadArray` in `photo_types.py`).
- **PyQt Signals**: Extensive use of signals for GUI communication (e.g., `boundsChanged`, `metadataChanged`)
- **EXIF Metadata**: All extracted photos preserve original EXIF data and add custom metadata

### File Organization

**Type-First Development**: When adding new modules:

1. Start with type definitions in appropriate module
2. Import semantic types from `core/photo_types.py`
3. Add proper type annotations from the beginning
4. Validate with `pyright` and `ruff` frequently
