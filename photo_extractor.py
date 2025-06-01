"""
Legacy photo_extractor module - classes have been moved to separate modules.

This file is kept for backward compatibility but all classes have been extracted to:
- gui.main_window.PhotoExtractorApp
- gui.image_view.ImageView
- gui.directory_sidebar.DirectoryImageList
- gui.settings_dialog.Settings, SettingsDialog
- storage.bounding_box_storage.BoundingBoxStorage
"""

# Import the main app class for backward compatibility
from gui.main_window import PhotoExtractorApp

__all__ = ['PhotoExtractorApp']