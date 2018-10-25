"""
Test module for accumulating hook (cxflow.hooks.accumulate_variables_hook).
"""

import numpy as np
import pytest

from cxflow.hooks.accumulate_variables import AccumulateVariables


_ITERS = 9
_EXAMPLES = 5
_FEATURES = 6


def get_batch():
    batch = {'input': np.ones((_EXAMPLES, _FEATURES)),
             'target': np.zeros(_EXAMPLES),
             'accuracy': np.ones(_EXAMPLES),
             'cost': np.ones(_EXAMPLES),
             'not_iter': 1}
    return batch


def test_accumulating_present_variables():
    """Test accumulating selected variables which are present in a batch."""

    selected_vars = ["accuracy", "cost"]
    stream_name = "train"
    accum_hook = AccumulateVariables(variables=selected_vars)

    for _ in range(_ITERS):
        batch = get_batch()
        accum_hook.after_batch(stream_name, batch)

    for var in selected_vars:
        assert len(accum_hook._accumulator[stream_name][var]) == _EXAMPLES * _ITERS
        assert np.array_equal(accum_hook._accumulator[stream_name][var],
                                       np.ones(_EXAMPLES * _ITERS))


_VARS_ERROR = [(["accuracy", "cost", "classes"], KeyError),
               (["accuracy", "cost", "not_iter"], TypeError)]

@pytest.mark.parametrize('vars, error', _VARS_ERROR)
def test_raise_exception(vars, error):
    """Test raising an exception if variable is not present in a batch/variable is not iterable."""

    selected_vars = vars
    stream_name = "train"
    accum_hook = AccumulateVariables(variables=selected_vars)
    batch = get_batch()
    with pytest.raises(error):
        accum_hook.after_batch(stream_name, batch)


def test_init_accumulator():
    """Test reseting accumulator after epoch."""

    selected_vars = ["accuracy", "cost"]
    stream_name = "train"
    accum_hook = AccumulateVariables(variables=selected_vars)

    for _ in range(_ITERS):
        batch = get_batch()
        accum_hook.after_batch(stream_name, batch)

    accum_hook.after_epoch()
    assert not accum_hook._accumulator
