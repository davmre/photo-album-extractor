"""
Shared utilities for CLI commands.
"""

from __future__ import annotations

from pathlib import Path


def get_image_files(paths: list[str]) -> list[Path]:
    """Get list of image files from input paths (files or directories)."""
    image_extensions = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}
    image_files = []

    for path_str in paths:
        path = Path(path_str)
        if path.is_file():
            if path.suffix.lower() in image_extensions:
                image_files.append(path)
            else:
                print(f"Warning: {path} is not a supported image file")
        elif path.is_dir():
            for ext in image_extensions:
                image_files.extend(path.glob(f"*{ext}"))
                image_files.extend(path.glob(f"*{ext.upper()}"))
        else:
            print(f"Error: Path not found: {path}")
            return []

    return sorted(image_files)
