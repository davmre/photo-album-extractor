import os

import cv2
import numpy as np
import pytest

import core.geometry as geometry
from core import refinement_strategies

DEBUG_IMAGES_BASE_DIR = "/Users/dave/photos_tests/"  # Edit to save debug images.

# Bounding box of a single pixel. Shapes drawn using OpenCV polyFill actually
# extend one pixel beyond the specified bounds because the pixels themselves
# take up space. For example, the trivial rectangle `[(0, 0)] * 4` fills the
# single pixel at `[0, 0]` with lower right corner at `[1, 1]`.
# Thus we'd expect to detect a rectangle with lower-right corner `[1, 1]`, even
# though we only drew at `[0, 0]`.
PIXEL_BOUNDS = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])


def axis_aligned_square():
    image = np.zeros((120, 100, 3), np.uint8)
    true_corners = np.array([(20, 20), (80, 20), (80, 80), (20, 80)])
    true_corners = np.asarray(true_corners, dtype=int)
    cv2.fillPoly(
        image, [true_corners - PIXEL_BOUNDS], (255, 255, 255), lineType=cv2.LINE_AA
    )
    return (image, [true_corners], "axis_aligned_square")


def axis_aligned_rect():
    image = np.zeros((600, 500, 3), np.uint8)
    # Long and skinny rectangle to accentuate any issues w non-square proportions.
    true_corners = np.array([(100, 100), (120, 100), (120, 500), (100, 500)])
    true_corners = np.asarray(true_corners, dtype=int)
    cv2.fillPoly(
        image, [true_corners - PIXEL_BOUNDS], (255, 255, 255), lineType=cv2.LINE_AA
    )
    return (image, [true_corners], "axis_aligned_rect")


def skewed_rect():
    image = np.zeros((600, 500, 3), np.uint8)
    base_corners = np.array([(100, 100), (400, 100), (400, 500), (100, 500)])
    true_corners = np.round(rotate_around_center(base_corners, -np.pi / 180))
    true_corners = np.asarray(true_corners, dtype=int)

    cv2.fillPoly(image, [true_corners], (255, 255, 255), lineType=cv2.LINE_AA)

    return (image, [true_corners], "skewed_rect")


def diamond():
    img_height = 600
    img_width = 500
    image = np.zeros((img_height, img_width, 3), np.uint8)
    base_corners = np.array([(100, 100), (400, 100), (400, 500), (100, 500)])
    true_corners = np.round(rotate_around_center(base_corners, -np.pi / 4))
    true_corners = np.asarray(true_corners, dtype=int)
    cv2.fillPoly(image, [true_corners], (255, 255, 255), lineType=cv2.LINE_AA)
    return (image, [true_corners], "test_diamond")


def almost_rect():
    img_height = 600
    img_width = 500

    image = np.zeros((img_height, img_width, 3), np.uint8)
    base_corners = np.array([(102, 102), (400, 100), (400, 500), (100, 500)])

    true_corners = np.round(rotate_around_center(base_corners, -np.pi / 180))
    true_corners = np.asarray(true_corners, dtype=int)
    cv2.fillPoly(image, [true_corners], (255, 255, 255), lineType=cv2.LINE_AA)
    return (image, [true_corners], "almost_rect")


def one_corner_offscreen():
    img_height = 600
    img_width = 500

    image = np.zeros((img_height, img_width, 3), np.uint8)

    base_corners = np.array([(100, 100), (400, 100), (400, 500), (100, 500)])

    true_corners = np.round(rotate_around_center(base_corners, -6 * np.pi / 180))
    true_corners[:, 0] += 100
    true_corners = np.asarray(true_corners, dtype=int)
    cv2.fillPoly(image, [true_corners], (255, 255, 255), lineType=cv2.LINE_AA)
    return (image, [true_corners], "one_corner_offscreen")


def two_corners_offscreen():
    img_height = 600
    img_width = 500

    image = np.zeros((img_height, img_width, 3), np.uint8)

    base_corners = np.array([(100, 100), (400, 100), (400, 500), (100, 500)])

    true_corners = np.round(rotate_around_center(base_corners, -6 * np.pi / 180))
    true_corners[:, 0] += 140

    true_corners = np.asarray(true_corners, dtype=int)
    cv2.fillPoly(image, [true_corners], (255, 255, 255), lineType=cv2.LINE_AA)
    return (image, [true_corners], "two_corners_offscreen")


def adjacent_boxes():
    img_height = 600
    img_width = 500
    image = np.zeros((img_height, img_width, 3), np.uint8)
    true_corners1 = np.array([(100, 100), (300, 100), (300, 300), (100, 300)])
    true_corners2 = np.array([(150, 310), (400, 310), (400, 500), (150, 500)])

    true_corners1 = np.asarray(true_corners1, dtype=int)
    true_corners2 = np.asarray(true_corners2, dtype=int)
    cv2.fillPoly(
        image, [true_corners1 - PIXEL_BOUNDS], (255, 255, 255), lineType=cv2.LINE_AA
    )
    cv2.fillPoly(
        image, [true_corners2 - PIXEL_BOUNDS], (255, 255, 255), lineType=cv2.LINE_AA
    )
    return (image, [true_corners1, true_corners2], "adjacent_boxes")


def overlapping_boxes():
    img_height = 600
    img_width = 500
    image = np.zeros((img_height, img_width, 3), np.uint8)

    base_corners1 = np.array([(100, 100), (300, 100), (300, 300), (100, 300)])
    base_corners2 = np.array([(150, 270), (400, 270), (400, 500), (150, 500)])
    true_corners1 = np.round(rotate_around_center(base_corners1, 1 * np.pi / 180))
    true_corners2 = np.round(rotate_around_center(base_corners2, -3 * np.pi / 180))

    true_corners1 = np.asarray(true_corners1, dtype=int)
    true_corners2 = np.asarray(true_corners2, dtype=int)
    cv2.fillPoly(
        image, [true_corners1 - PIXEL_BOUNDS], (255, 255, 255), lineType=cv2.LINE_AA
    )
    cv2.fillPoly(
        image, [true_corners2 - PIXEL_BOUNDS], (255, 255, 255), lineType=cv2.LINE_AA
    )
    return (image, [true_corners1, true_corners2], "overlapping_boxes")


def colored_edge():
    img_height = 400
    img_width = 400
    image = np.zeros((img_height, img_width, 3), np.uint8) + np.array(
        (255, 0, 0), dtype=np.uint8
    )
    true_corners = np.array([(100, 100), (300, 100), (300, 300), (100, 300)])
    true_corners = np.asarray(true_corners, dtype=int)
    cv2.fillPoly(image, [true_corners - PIXEL_BOUNDS], (0, 130, 0))

    gray = np.asarray(
        cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY), dtype=np.float32
    )
    print("GRAY", gray)
    assert np.max(gray) == np.min(gray)

    return (image, [true_corners], "colored_edge")


def hires_overlapping_boxes():
    img_height = 6000
    img_width = 5000

    image = np.zeros((img_height, img_width, 3), np.uint8) + np.asarray(
        [255, 255, 255], dtype=np.uint8
    )

    base_corners1 = np.array([(1000, 1000), (3000, 1000), (3000, 3000), (1000, 3000)])
    base_corners2 = np.array([(1500, 2700), (4000, 2700), (4000, 5000), (1500, 5000)])

    true_corners1 = np.round(rotate_around_center(base_corners1, 1 * np.pi / 180))
    true_corners2 = np.round(rotate_around_center(base_corners2, -3 * np.pi / 180))

    true_corners1 = np.asarray(true_corners1, dtype=int)
    true_corners2 = np.asarray(true_corners2, dtype=int)
    cv2.fillPoly(
        image, [true_corners1 - PIXEL_BOUNDS], (100, 100, 100), lineType=cv2.LINE_AA
    )
    cv2.fillPoly(
        image, [true_corners2 - PIXEL_BOUNDS], (100, 100, 100), lineType=cv2.LINE_AA
    )
    return (image, [true_corners1, true_corners2], "hires_overlapping_boxes")


def max_deviation(rect1, rect2):
    """Score box pair by distance between corresponding corner points."""
    rect1 = geometry.sort_clockwise(rect1)
    rect2 = geometry.sort_clockwise(rect2)
    d = 0.0
    for corner1, corner2 in zip(rect1, rect2):
        d = max(d, np.linalg.norm(corner2 - corner1))
    return d


def rotate_around_center(points, angle):
    center = np.mean(points, axis=0)
    relative_coords = points - center
    rotated_coords = np.dot(
        relative_coords,
        np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]),
    )
    return rotated_coords + center


def perturb_corners(rect, reltol):
    """Randomly perturb the given shape within the allowed tolerance."""
    coordinates = geometry.PatchCoordinatesConverter(rect)
    perturbed_unit_square = geometry.UNIT_SQUARE + np.random.uniform(
        low=-reltol, high=reltol, size=rect.shape
    )
    perturbed_in_image_coords = coordinates.unit_square_to_image(perturbed_unit_square)
    return perturbed_in_image_coords


class _TestImageRefinement:
    """Test that refinement recovers correct coords in easy synthetic cases."""

    test_strategy: refinement_strategies.RefinementStrategy

    def check_refinement_recovers_gold_corners(
        self,
        image,
        input_box,
        gold_box,
        strategy: refinement_strategies.RefinementStrategy,
        allowed_deviation=3,
        debug_name="",
    ):
        refined_box = strategy.refine(
            image,
            input_box,
            debug_dir=(
                os.path.join(DEBUG_IMAGES_BASE_DIR, debug_name)
                if DEBUG_IMAGES_BASE_DIR
                else None
            ),
        )
        deviation = max_deviation(gold_box, refined_box)
        print("GOLD", gold_box)
        print("INPUT", input_box)
        print("REFINED", refined_box)
        print("DEVIATION", deviation)
        assert deviation <= allowed_deviation

    @pytest.mark.parametrize(
        "test_generator_fn",
        [
            axis_aligned_square,
            axis_aligned_rect,
            skewed_rect,
            diamond,
            adjacent_boxes,
            overlapping_boxes,
            one_corner_offscreen,
            colored_edge,
            hires_overlapping_boxes,
        ],
    )
    def test_refinement_of_perturbed_corners_recovers_gold_corners(
        self, test_generator_fn, reltol=0.05, num_perturbations=1
    ):
        image, gold_boxes, name = test_generator_fn()
        strategy = self.test_strategy
        for box_idx, gold_box in enumerate(gold_boxes):
            perturbations = [
                perturb_corners(gold_box, reltol=reltol)
                for _ in range(num_perturbations)
            ]
            for i, input_box in enumerate(perturbations):
                self.check_refinement_recovers_gold_corners(
                    image,
                    input_box,
                    gold_box,
                    strategy=strategy,
                    debug_name=f"_{name}_box{box_idx}_perturb{i}",
                )

    @pytest.mark.parametrize(
        "test_generator_fn",
        [
            axis_aligned_square,
            axis_aligned_rect,
            skewed_rect,
            diamond,
            adjacent_boxes,
            overlapping_boxes,
            one_corner_offscreen,
            colored_edge,
            hires_overlapping_boxes,
        ],
    )
    def test_gold_corners_are_fixed_point(
        self, test_generator_fn, reltol=0.05, resolution_scale_factor=1.0
    ):
        image, gold_boxes, name = test_generator_fn()
        strategy = self.test_strategy
        for box_idx, gold_box in enumerate(gold_boxes):
            debug_dir = (
                os.path.join(DEBUG_IMAGES_BASE_DIR, f"{name}_box{box_idx}_gold_init")
                if DEBUG_IMAGES_BASE_DIR
                else None
            )
            refined_box = strategy.refine(
                image,
                gold_box,
                debug_dir=debug_dir,
            )
            deviation = max_deviation(gold_box, refined_box)
            print("GOLD INPUT", gold_box)
            print("REFINED", refined_box)
            print("DEVIATION", deviation)
            assert deviation <= 1.0

    def test_special_one_corner(self):
        image, gold_boxes, name = one_corner_offscreen()
        init_corners = np.array(
            [
                [223.57437551, 81.27825425],
                [523.98298468, 117.23571912],
                [476.42859094, 531.86912888],
                [193.25137172, 498.92439837],
            ]
        )
        self.check_refinement_recovers_gold_corners(
            image,
            init_corners,
            gold_boxes[0],
            strategy=self.test_strategy,
            debug_name=f"{name}_special",
        )


class TestSyntheticImageRefinementStrips(_TestImageRefinement):
    test_strategy = refinement_strategies.RefinementStrategyStrips()


class TestSyntheticImageRefinementHough(_TestImageRefinement):
    test_strategy = refinement_strategies.RefinementStrategyHoughGreedy()
