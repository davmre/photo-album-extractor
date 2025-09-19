from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from core.photo_types import AnyArray, IntArray

try:
    from matplotlib import pylab as plt  # pyright: ignore[reportMissingImports]
except ImportError:
    pass


# Utils to save debugging images.


def annotate_image(
    img: AnyArray | Image.Image,
    contours=None,
    edges: list[IntArray] | tuple[IntArray, ...] | None = None,
) -> AnyArray:
    if isinstance(img, Image.Image):
        img = np.array(img)
    img = img.copy()
    if contours:
        cv2.drawContours(
            img,
            np.array(np.round(contours), dtype=int),  # type: ignore
            -1,
            (0, 0, 255),
            1,  # type: ignore
        )  # type: ignore
    if edges:
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 0, 255),
            (255, 255, 0),
            (0, 255, 255),
        ]
        for edge, color in zip(edges, colors):
            cv2.line(img, (edge[0, 0], edge[0, 1]), (edge[1, 0], edge[1, 1]), color, 1)
    return img


def save_image(file_path: str, img: AnyArray | Image.Image) -> None:
    dir = os.path.dirname(file_path)
    pd = Path(dir).expanduser()
    if not pd.exists():
        pd.mkdir(parents=True)

    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    if img.mode == "F":
        # PNG supports greyscale images with 8-bit int pixels.
        img = img.convert("L")

    img.save(file_path)
    print(f"saved: {file_path}")


def save_histogram(filename, bins, scores, title):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    bin_width = bins[1] - bins[0]
    ax.bar(bins[:-1], scores, width=bin_width)
    ax.set_title(title)
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def save_plot(filename, xs, ys, title):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.plot(xs, ys)
    ax.set_title(title)
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)
