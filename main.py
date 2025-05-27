#!/usr/bin/env python3
"""
Photo Album Extractor
Main entry point for the photo extraction application.
"""

import sys
from PyQt6.QtWidgets import QApplication
from photo_extractor import PhotoExtractorApp

def main():
    app = QApplication(sys.argv)
    window = PhotoExtractorApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()