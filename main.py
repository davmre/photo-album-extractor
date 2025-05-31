#!/usr/bin/env python3
"""
Photo Album Extractor
Main entry point for the photo extraction application.
"""

import sys
import argparse
import os
from PyQt6.QtWidgets import QApplication
from photo_extractor import PhotoExtractorApp

def main():
    parser = argparse.ArgumentParser(description='Photo Album Extractor')
    parser.add_argument('image', nargs='?', help='Path to image file to load on startup')
    args = parser.parse_args()
    
    # Validate image path if provided
    image_path = None
    if args.image:
        if os.path.isfile(args.image):
            image_path = args.image
        else:
            print(f"Error: Image file not found: {args.image}")
            return 1
    
    app = QApplication(sys.argv)
    window = PhotoExtractorApp(initial_image=image_path)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()