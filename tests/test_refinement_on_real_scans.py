import json
import os

import numpy as np
import PIL.Image as Image

import image_processing.refine_bounds as refine_bounds
from image_processing import geometry

# Utility to evaluate refinement strategies by comparing results to gold boxes
# on real scanned album pages.
# TODOs:
# 1. Command-line argument to select strategies to test, and whether to dump debug info for failures
# 2. Prettier / more informative logging of results
# 3. Suppress console noise from debug logging (at least during initial refinement --- if we're rerunning and dumping data for failure cases maybe we also want to capture and save debug logs for those)
# 4. Maybe add a field to the json to specify allowed deviations for a box (some are harder than others)
# 4. Set up a test class and test methods for pytest (but we still want to support running main() directly also)
# 5. Clean up and check in the test images and bounding box data

REFINEMENT_TEST_DATA_DIR = "/Users/dave/Pictures/refinement_evals"


def load_json(file_name):
    with open(file_name) as f:
        return json.load(f)


GOLD_DATA = load_json(os.path.join(REFINEMENT_TEST_DATA_DIR, "gold_boxes.json"))
INIT_DATA = load_json(os.path.join(REFINEMENT_TEST_DATA_DIR, "init_boxes.json"))


def get_corner_deviations(rect1, rect2):
    """Score box pair by avg distance between corresponding corner points."""
    rect1 = geometry.sort_clockwise(rect1)
    rect2 = geometry.sort_clockwise(rect2)
    return [np.linalg.norm(corner2 - corner1) for corner1, corner2 in zip(rect1, rect2)]


def get_matched_corner_deviations(rect, gold_rects):
    deviations = np.array([get_corner_deviations(rect, gold) for gold in gold_rects])
    best_avg_match_idx = np.argmin(np.mean(deviations, axis=-1))
    return deviations[best_avg_match_idx]


class ImageWithBoxes:
    def __init__(self, file_name):
        self.file_name = file_name
        self.image = Image.open(os.path.join(REFINEMENT_TEST_DATA_DIR, file_name))
        self.gold_boxes = np.array(
            [box_data["corners"] for box_data in GOLD_DATA[file_name]]
        )
        self.init_boxes = np.array(
            [box_data["corners"] for box_data in INIT_DATA[file_name]]
        )
        self.refined_boxes = {}
        self.corner_deviations = {}

    def refine_all_boxes(self, refine_strategy: str):
        refine_fn = refine_bounds.REFINEMENT_STRATEGIES[refine_strategy]
        refined_boxes = []
        for i, box in enumerate(self.init_boxes):
            refined = refine_fn(self.image, box, debug_dir=None)
            refined_boxes.append(refined)
        self.refined_boxes[refine_strategy] = refined_boxes

    def score_refinements(self):
        for strategy in self.refined_boxes.keys():
            corner_deviations = []
            for box in self.refined_boxes[strategy]:
                corner_deviations.append(
                    get_matched_corner_deviations(box, self.gold_boxes)
                )
            self.corner_deviations[strategy] = corner_deviations

    def dump_debug_images_for_failures(
        self,
        debug_dir: str,
        refine_strategy=None,
        allowed_average_deviation=2,
        allowed_max_deviation=3,
    ):
        strategies = [refine_strategy] if refine_strategy else self.refined_boxes.keys()
        for strategy in strategies:
            refine_fn = refine_bounds.REFINEMENT_STRATEGIES[strategy]
            for i in range(len(self.corner_deviations[strategy])):
                box_deviations = self.corner_deviations[strategy][i]
                if (
                    np.max(box_deviations) > allowed_max_deviation
                    or np.mean(box_deviations) > allowed_average_deviation
                ):
                    # Rerun refinement to dump debugging info
                    refine_fn(
                        self.image,
                        self.init_boxes[i],
                        debug_dir=os.path.join(
                            debug_dir, self.file_name, f"box_{i}", strategy
                        ),
                    )


def main():
    strategies = ["Strips (native res)", "Original (multiscale)"]

    test_image_filenames = [
        name
        for name in os.listdir(REFINEMENT_TEST_DATA_DIR)
        if os.path.splitext(name)[1] in (".png", ".jpg")
    ]

    corner_deviations = {s: [] for s in strategies}

    for filename in test_image_filenames:
        test_object = ImageWithBoxes(filename)
        for strategy in strategies:
            test_object.refine_all_boxes(strategy)
        test_object.score_refinements()
        for strategy in strategies:
            corner_deviations[strategy].extend(test_object.corner_deviations[strategy])
            print(f"{filename} {strategy}: {test_object.corner_deviations[strategy]}")

    print("OVERALL RESULTS")
    for strategy in strategies:
        avg_case = np.mean(corner_deviations[strategy])
        worst_case = np.max(corner_deviations[strategy])
        median = np.median(corner_deviations[strategy])
        print(
            f"{strategy}: median {median: .2f} avg {avg_case: .2f} worst {worst_case: .2f}"
        )


if __name__ == "__main__":
    main()
