#!/usr/bin/env python3
"""
Photo Album Extractor
Main entry point for the photo extraction application.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from PyQt6.QtWidgets import QApplication

from cli.detect import cmd_detect
from cli.extract import cmd_extract
from cli.info import cmd_info
from cli.interpolate_dates import cmd_interpolate_dates
from cli.refine import cmd_refine
from gui.main_window import PhotoExtractorApp


def main():
    parser = argparse.ArgumentParser(description="Photo Album Extractor")

    # Global arguments
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="WARNING",
        help="Set the logging level (default: WARNING)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # GUI mode (default when no command specified)
    gui_parser = subparsers.add_parser("gui", help="Launch GUI application (default)")
    gui_parser.add_argument(
        "path", nargs="?", help="Path to image file or directory to load on startup"
    )
    gui_parser.add_argument(
        "--refine_debug_dir",
        default=None,
        help="Directory to save debugging images for boundary refinement.",
    )

    # Info command
    info_parser = subparsers.add_parser("info", help="Show bounding box information")
    info_parser.add_argument("paths", nargs="+", help="Image files or directories")

    # Detect command
    detect_parser = subparsers.add_parser("detect", help="Detect photos in images")
    detect_parser.add_argument("paths", nargs="+", help="Image files or directories")
    detect_parser.add_argument(
        "--force", action="store_true", help="Overwrite existing bounding box data"
    )

    # Detect command
    refine_parser = subparsers.add_parser(
        "refine", help="Refine detected bounding boxes"
    )
    refine_parser.add_argument("paths", nargs="+", help="Image files or directories")
    refine_parser.add_argument(
        "--strategy", help="Optional refinement strategy to use (default: app settings)"
    )
    refine_parser.add_argument(
        "--debug_dir", help="Optional directory to dump debugging images."
    )
    refine_parser.add_argument(
        "--tolerance", type=float, help="Tolerance for refinement convergence (default: app settings)"
    )

    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract photos from images")
    extract_parser.add_argument("paths", nargs="+", help="Image files or directories")
    extract_parser.add_argument(
        "--output-dir", required=True, help="Directory to save extracted photos"
    )
    extract_parser.add_argument(
        "--base-name", help="Base name for extracted files (default: source image name)"
    )

    # Extract command
    interpolate_dates_parser = subparsers.add_parser(
        "interpolate_dates",
        help="Fill in dates where unspecified based on dates of surrounding photos.",
    )
    interpolate_dates_parser.add_argument("directory", help="Directory to process")
    interpolate_dates_parser.add_argument(
        "--trial-run",
        action="store_true",
        help="Print interpolated dates but do not save them.",
    )

    # Placeholder for future commands
    subparsers.add_parser("clear", help="Clear bounding box data (not yet implemented)")

    # Legacy support: if no subcommand, assume GUI mode
    args = parser.parse_args()

    # Configure logging based on command line argument
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if args.command is None:
        # Try to parse as legacy GUI arguments
        legacy_parser = argparse.ArgumentParser(description="Photo Album Extractor")
        legacy_parser.add_argument(
            "path", nargs="?", help="Path to image file or directory to load on startup"
        )
        legacy_parser.add_argument(
            "--refine_debug_dir",
            default=None,
            help="Directory to save debugging images for boundary refinement.",
        )
        args = legacy_parser.parse_args()
        args.command = "gui"

    # Handle commands
    if args.command == "info":
        return cmd_info(paths=args.paths)
    if args.command == "interpolate_dates":
        return cmd_interpolate_dates(directory=args.directory, trial_run=args.trial_run)
    elif args.command == "detect":
        return cmd_detect(paths=args.paths, force=args.force)
    elif args.command == "extract":
        return cmd_extract(
            paths=args.paths,
            output_dir=args.output_dir,
            base_name=args.base_name,
        )
    elif args.command == "refine":
        return cmd_refine(
            paths=args.paths,
            debug_dir=args.debug_dir,
            strategy=args.strategy,
            tolerance=args.tolerance,
        )
    elif args.command in ["clear"]:
        print(f"Command '{args.command}' is not yet implemented")
        return 1
    else:  # GUI mode
        # Validate path if provided
        initial_image = None
        initial_directory = None

        if hasattr(args, "path") and args.path:
            if os.path.isfile(args.path):
                initial_image = args.path
            elif os.path.isdir(args.path):
                initial_directory = args.path
            else:
                print(f"Error: File or directory not found: {args.path}")
                return 1

        app = QApplication(sys.argv)
        refine_debug_dir = getattr(args, "refine_debug_dir", None)
        window = PhotoExtractorApp(
            initial_image=initial_image,
            initial_directory=initial_directory,
            refine_debug_dir=refine_debug_dir,
        )
        window.show()
        sys.exit(app.exec())


if __name__ == "__main__":
    main()
