# Photo Album Extractor

A cross-platform GUI application for extracting individual photos from scanned album pages.

## Features

- **Image Display**: Load and view scanned album pages with zoom/pan capabilities
- **Interactive Bounding Boxes**: Create draggable, resizable selection rectangles
- **Context Menu**: Right-click to add new boxes or remove existing ones
- **Batch Export**: Extract all selected regions with auto-incrementing filenames
- **Cross-Platform**: Built with PyQt6 for Windows, macOS, and Linux support

## Installation

1. Install Python 3.8 or higher
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   python main.py
   ```

2. **Load an image**: Click "Load Image" or use Ctrl+O to select a scanned album page

3. **Set output folder**: Click "Set Output Folder" to choose where extracted photos will be saved

4. **Create bounding boxes**: Right-click on the image and select "Add Bounding Box"

5. **Adjust boxes**: 
   - Drag boxes to move them
   - Drag corner/edge handles to resize
   - Right-click on a box to remove it

6. **Extract photos**: Click "Extract Photos" to save all cropped images

## Interface

- **Menu Bar**: File operations and editing commands
- **Toolbar**: Quick access to main functions
- **Image View**: Scrollable, zoomable image display with overlay boxes
- **Status Bar**: Current operation feedback

## Keyboard Shortcuts

- `Ctrl+O`: Load image
- `Ctrl+Q`: Exit application
- Mouse wheel: Zoom in/out

## File Formats

- **Input**: PNG, JPG, JPEG, BMP, TIFF, GIF
- **Output**: JPEG files with customizable base names

## Tips

- Use mouse wheel to zoom for precise box placement
- Boxes can be moved and resized after creation
- Output files are automatically numbered (e.g., photo_001.jpg, photo_002.jpg)
- If a filename already exists, a counter is added to ensure uniqueness