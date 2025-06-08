#!/usr/bin/env python3
"""
Photo Album Extractor
Main entry point for the photo extraction application.
"""

import sys
import argparse
import os
from PyQt6.QtWidgets import QApplication
from gui.main_window import PhotoExtractorApp

def main():
    parser = argparse.ArgumentParser(description='Photo Album Extractor')
    parser.add_argument('path', nargs='?', help='Path to image file or directory to load on startup')
    parser.add_argument('--refine_debug_dir', default=None, help="Directory to save debugging images for boundary refinement.")
    args = parser.parse_args()
    
    # Validate path if provided
    initial_image = None
    initial_directory = None
    
    if args.path:
        if os.path.isfile(args.path):
            initial_image = args.path
        elif os.path.isdir(args.path):
            initial_directory = args.path
        else:
            print(f"Error: File or directory not found: {args.path}")
            return 1
    
    app = QApplication(sys.argv)
    window = PhotoExtractorApp(initial_image=initial_image, initial_directory=initial_directory,
                               refine_debug_dir = args.refine_debug_dir)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()