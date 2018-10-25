"""
Test module for confusion_matrix utils (cxflow.utils.confusion_matrix).
"""
import numpy as np
import pytest

from cxflow.utils.confusion_matrix import confusion_matrix


_INVALID_INPUTS = [(np.array([0]), np.array([0]), 'a', TypeError),
                   ([0], np.array([0]), 3, AttributeError),
                   (np.array([0]), [0], 3, AttributeError),  # wrong argument types
                   (np.array([[0]]), np.array([0]), 3, AssertionError),
                   (np.array([0]), np.array([[0]]), 3, AssertionError),
                   (np.array([0, 1]), np.array([0]), 3, AssertionError),  # wrong array shapes
                   (np.array([0, 2]), np.array([0, 2]), 2, AssertionError),  # wrong value of `num_classes` input
                   (np.array([0.]), np.array([0]), 3, AssertionError),
                   (np.array([0]), np.array([0.]), 3, AssertionError),  # wrong `np.dtype` of input arrays
                   (np.array([-1]), np.array([0]), 3, AssertionError),
                   (np.array([0]), np.array([-1]), 3, AssertionError)]


@pytest.mark.parametrize('exp, pred, num, error', _INVALID_INPUTS)
def test_invalid_inputs(exp, pred, num, error):
    with pytest.raises(error):
        confusion_matrix(expected=exp, predicted=pred, num_classes=num)


_VALID_INPUTS = [(np.array([0, 1, 2]), np.array([0, 1, 1]), 3, np.array([[1, 0, 0],
                                                                   [0, 1, 0],
                                                                   [0, 1, 0]])),
           (np.array([0, 1, 2]), np.array([0, 1, 1]), 4, np.array([[1, 0, 0, 0],
                                                                   [0, 1, 0, 0],
                                                                   [0, 1, 0, 0],
                                                                   [0, 0, 0, 0]])),
           (np.array([0, 1, 2, 1, 0, 1, 2, 2, 1, 0, 0, 0, 1, 2, 2]),
            np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]), 3, np.array([[2, 0, 3],
                                                                                  [2, 2, 1],
                                                                                  [1, 2, 2]]))]


@pytest.mark.parametrize('exp, pred, num, output', _VALID_INPUTS)
def test_confusion_matrix(exp, pred, num, output):
    calculated_cm = confusion_matrix(expected=exp, predicted=pred, num_classes=num)
    groundtruth_cm = output
    np.testing.assert_equal(calculated_cm, groundtruth_cm)
