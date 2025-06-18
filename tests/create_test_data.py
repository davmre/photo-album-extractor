#!/usr/bin/env python3
"""
Script to generate test data for the photo extractor app.
Creates dummy images with EXIF data and sample bounding box/attribute data.
"""

import json
import os

from PIL import Image, ImageDraw


def create_test_images():
    """Create test album page images in various formats."""
    test_dir = "test_data/album1"

    # Create test album page images (scanned pages, not individual photos)
    images = [
        ("album_page1.jpg", "JPEG", (2400, 1800), "#F5F5F5"),
        ("album_page2.png", "PNG", (2048, 1536), "white"),
        ("album_page3.tiff", "TIFF", (1920, 1440), "#F5F5DC"),
        ("album_page4.jpg", "JPEG", (2560, 1920), "#FFFDD0"),
    ]

    for filename, format_type, size, color in images:
        # Create album page with multiple photo placeholders
        img = Image.new("RGB", size, color)
        draw = ImageDraw.Draw(img)

        # Add album page border
        draw.rectangle([20, 20, size[0] - 20, size[1] - 20], outline="gray", width=4)

        # Add title
        draw.text((50, 50), f"Album Page: {filename}", fill="black")

        # Add multiple "photo" rectangles that represent photos on the album page
        photo_positions = [
            (100, 150, 400, 450),  # Photo 1
            (500, 150, 800, 450),  # Photo 2
            (100, 500, 400, 800),  # Photo 3
            (500, 500, 800, 800),  # Photo 4
        ]

        for i, (x1, y1, x2, y2) in enumerate(photo_positions):
            # Draw photo placeholder
            draw.rectangle([x1, y1, x2, y2], fill="white", outline="black", width=2)
            draw.text((x1 + 10, y1 + 10), f"Photo {i + 1}", fill="black")

            # Add some details inside each "photo"
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            draw.rectangle(
                [center_x - 30, center_y - 30, center_x + 30, center_y + 30],
                fill="lightblue",
                outline="blue",
                width=1,
            )

        filepath = os.path.join(test_dir, filename)
        img.save(
            filepath, format=format_type, quality=95 if format_type == "JPEG" else None
        )

    print(f"Created {len(images)} test album page images in {test_dir}")


def create_bounding_box_data():
    """Create sample bounding box data for album pages."""
    test_dir = "test_data/album1"

    # Sample bounding box data that matches the storage format
    # These represent individual photos extracted from album pages
    bounding_data = {
        "album_page1.jpg": [
            {
                "type": "quad",
                "id": "photo1-page1",
                "corners": [[100, 150], [400, 150], [400, 450], [100, 450]],
                "attributes": {
                    "date_time": "1985-06-20",
                    "comments": "Birthday party - Sarah blowing out candles",
                },
            },
            {
                "type": "quad",
                "id": "photo2-page1",
                "corners": [[500, 150], [800, 150], [800, 450], [500, 450]],
                "attributes": {
                    "date_time": "1985-06-20",
                    "comments": "Birthday cake - chocolate with strawberries",
                },
            },
            {
                "type": "quad",
                "id": "photo3-page1",
                "corners": [[100, 500], [400, 500], [400, 800], [100, 800]],
                "attributes": {
                    "date_time": "1985-06-21",
                    "comments": "Family group photo after party",
                },
            },
        ],
        "album_page2.png": [
            {
                "type": "quad",
                "id": "photo1-page2",
                "corners": [[120, 170], [420, 180], [410, 470], [110, 460]],
                "attributes": {
                    "date_time": "1985-07-04",
                    "comments": "4th of July fireworks at the lake",
                },
            },
            {
                "type": "quad",
                "id": "photo2-page2",
                "corners": [[520, 160], [810, 165], [805, 455], [515, 450]],
                "attributes": {
                    "date_time": "1985-07-04",
                    "comments": "Kids playing in the lake",
                },
            },
        ],
        "album_page3.tiff": [
            {
                "type": "quad",
                "id": "photo1-page3",
                "corners": [[100, 150], [400, 150], [400, 450], [100, 450]],
                "attributes": {
                    "date_time": "1985-08-15",
                    "comments": "First day of school - backyard photo",
                },
            },
            {
                "type": "quad",
                "id": "photo2-page3",
                "corners": [[500, 500], [800, 500], [800, 800], [500, 800]],
                "attributes": {
                    "date_time": "1985-08-16",
                    "comments": "School lunch preparation",
                },
            },
        ],
    }

    # Save the bounding box data
    data_file = os.path.join(test_dir, ".photo_extractor_data.json")
    with open(data_file, "w") as f:
        json.dump(bounding_data, f, indent=2)

    print(f"Created bounding box data in {data_file}")


if __name__ == "__main__":
    create_test_images()
    create_bounding_box_data()
    print("Test data creation complete!")
