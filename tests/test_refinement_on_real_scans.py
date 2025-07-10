# Evaluate refinement algorithms on test images.
# To run:
# python3 -m tests/test_refinement_on_real_scans

import dataclasses
import json
import os
import time
from collections.abc import Sequence

import numpy as np
import PIL.Image as Image

from core import geometry
from core.bounding_box import BoundingBox
from core.bounding_box_storage import BoundingBoxStorage
from core.refinement_strategies import (  # RefinementStrategyHoughGreedy,; RefinementStrategyStrips,
    REFINEMENT_STRATEGIES,
    RefinementStrategy,
    RefinementStrategyHoughGreedy,
    RefinementStrategyHoughReranked,
    RefinementStrategyHoughSoft,
)

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


def get_matched_corner_deviations(rect: np.ndarray, gold_boxes: Sequence[BoundingBox]):
    deviations = np.array(
        [geometry.get_corner_deviations(rect, gold.corners) for gold in gold_boxes]
    )
    best_avg_match_idx = np.argmin(np.mean(deviations, axis=-1))
    return deviations[best_avg_match_idx]


class ImageWithBoxes:
    def __init__(self, file_name, storage_objects: dict[str, BoundingBoxStorage]):
        self.file_name = file_name
        self.image = Image.open(os.path.join(REFINEMENT_TEST_DATA_DIR, file_name))

        self.storage_objects = storage_objects
        self.gold_boxes = storage_objects["gold"].get_bounding_boxes(file_name)
        self.init_boxes = storage_objects["init"].get_bounding_boxes(file_name)

        self.refined_corners = {}
        self.corner_deviations = {}
        self.times = {}

    def refine_all_boxes(self, refine_strategy: RefinementStrategy):
        refined_boxes = []
        times = []
        for box in self.init_boxes:
            t0 = time.time()
            refined = refine_strategy.refine(self.image, box.corners, debug_dir=None)
            t1 = time.time()
            refined_boxes.append(refined)
            times.append(t1 - t0)
            self.storage_objects[refine_strategy.name].update_box_data(
                self.file_name,
                dataclasses.replace(box, corners=refined),
                save_data=True,
            )
        self.refined_corners[refine_strategy.name] = refined_boxes
        self.times[refine_strategy.name] = times

    def score_refinements(self):
        for strategy_name in self.refined_corners.keys():
            corner_deviations = []
            for corners in self.refined_corners[strategy_name]:
                corner_deviations.append(
                    get_matched_corner_deviations(corners, self.gold_boxes)
                )
            self.corner_deviations[strategy_name] = corner_deviations

    def dump_debug_images_for_failures(
        self,
        debug_dir: str,
        refine_strategy=None,
        allowed_average_deviation=2.0,
        allowed_max_deviation=3.0,
        box_idxs=None,
    ):
        strategy_names = (
            [refine_strategy] if refine_strategy else self.refined_corners.keys()
        )
        for strategy_name in strategy_names:
            strategy = REFINEMENT_STRATEGIES[strategy_name]
            if box_idxs is None:
                box_idxs = []
                for i in range(len(self.corner_deviations[strategy_name])):
                    box_deviations = self.corner_deviations[strategy_name][i]
                    if (
                        np.max(box_deviations) > allowed_max_deviation
                        or np.mean(box_deviations) > allowed_average_deviation
                    ):
                        box_idxs.append(i)
            for i in box_idxs:
                # Rerun refinement to dump debugging info
                strategy.refine(
                    self.image,
                    self.init_boxes[i].corners,
                    debug_dir=os.path.join(
                        debug_dir,
                        self.file_name,
                        f"{sanitize_filename(strategy_name)}_box_{i}",
                    ),
                )


def sanitize_filename(s):
    return s.lower().replace("(", "").replace(")", "").replace(" ", "_")


def main():
    strategies = [
        RefinementStrategyHoughReranked(),
        RefinementStrategyHoughGreedy(),
        RefinementStrategyHoughSoft(),
    ]  # REFINEMENT_STRATEGIES.values()

    test_image_filenames = [
        name
        for name in os.listdir(REFINEMENT_TEST_DATA_DIR)
        if os.path.splitext(name)[1] in (".png", ".jpg")
    ]

    # test_image_filenames = ["2025-05-27-0010 small.jpg"]

    corner_deviations = {s.name: [] for s in strategies}
    times = {s.name: [] for s in strategies}

    storage_objects = {
        s: BoundingBoxStorage(
            REFINEMENT_TEST_DATA_DIR,
            json_file_name=f"{sanitize_filename(s)}_boxes.json",
        )
        for s in list(REFINEMENT_STRATEGIES.keys()) + ["gold", "init"]
    }
    for k, s in storage_objects.items():
        print(f"{k}: data file {s.data_file}")

    for filename in test_image_filenames:
        test_object = ImageWithBoxes(filename, storage_objects)

        for strategy in strategies:
            test_object.refine_all_boxes(strategy)
        test_object.score_refinements()
        for strategy in strategies:
            corner_deviations[strategy.name].extend(
                test_object.corner_deviations[strategy.name]
            )
            times[strategy.name].extend(test_object.times[strategy.name])
            print(
                f"{filename} {strategy.name}: {test_object.corner_deviations[strategy.name]}"
            )

        # test_object.dump_debug_images_for_failures(
        #    debug_dir=os.path.join(REFINEMENT_TEST_DATA_DIR, "debugging_dumps"),
        #    refine_strategy="Hough transform (greedy)",
        #    allowed_average_deviation=4.0,
        #    # refine_strategy="Strips (native res)",
        # )

    print("OVERALL RESULTS")
    for strategy in strategies:
        avg_case = np.mean(corner_deviations[strategy.name])
        worst_case = np.max(corner_deviations[strategy.name])
        median = np.median(corner_deviations[strategy.name])
        avg_time = np.mean(times[strategy.name])
        stddev_time = np.std(times[strategy.name])
        print(
            f"{strategy.name}: median {median: .2f} avg {avg_case: .2f} worst {worst_case: .2f}"
        )
        print(f"  time: avg {avg_time} stddev {stddev_time}")


if __name__ == "__main__":
    main()
