import numpy as np

from invdetect.metrics import dice_score, precision_score


def test_segmentation_metrics() -> None:
    prediction = np.array([[1, 1, 0, 0]], dtype=np.uint8)
    target = np.array([[1, 0, 1, 0]], dtype=np.uint8)
    assert dice_score(prediction, target) == 0.5
    assert precision_score(prediction, target) == 0.5


def test_empty_masks_are_perfect() -> None:
    empty = np.zeros((2, 2), dtype=np.uint8)
    assert dice_score(empty, empty) == 1.0
    assert precision_score(empty, empty) == 1.0

