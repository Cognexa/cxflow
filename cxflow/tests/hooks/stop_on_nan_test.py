"""
Module with stop on NaN hook test case (see :py:class:`cxflow.hooks.StopOnNan`).
"""
import logging

import numpy as np
import pytest

from cxflow.hooks import StopOnNaN, TrainingTerminated


def _get_data(val):
    return {
        'train': {'var': val, 'zero': 0},
        'valid': {'var': 0, 'zero': 0}
    }


def test_variables():
    """Test stopping of NaNs of all, individual and missing variables. Check handling of infinities."""

    for nan in [np.nan, float('nan'), np.inf, float('inf')]:
        for val in [nan, [1, 2, nan], {1: {3: nan}, 2: 0}]:
            with pytest.raises(TrainingTerminated):
                StopOnNaN(variables=['var'], stop_on_inf=True).after_epoch(epoch_data=_get_data(val))
            if not np.isnan(nan):
                StopOnNaN(variables=['var']).after_epoch(epoch_data=_get_data(val))
                StopOnNaN().after_epoch(epoch_data=_get_data(val))
            StopOnNaN(variables=['zero']).after_epoch(epoch_data=_get_data(val))

    with pytest.raises(KeyError):
        StopOnNaN(variables=['missing']).after_epoch(epoch_data=_get_data(8))


def test_unsupported(caplog):
    """Test handling of unsupported types."""

    with pytest.raises(ValueError):
        StopOnNaN(on_unknown_type='error').after_epoch(epoch_data=_get_data(lambda: 0))

    with pytest.raises(AssertionError):
        StopOnNaN(on_unknown_type='bad value')

    caplog.clear()
    StopOnNaN(on_unknown_type='warn').after_epoch(epoch_data=_get_data(lambda: 0))
    assert caplog.record_tuples == [
        ('root', logging.WARNING, 'Variable `var` of type `<class \'function\'>` can not be checked for NaNs.')
    ]

    StopOnNaN().after_epoch(epoch_data=_get_data(lambda: 0))


def test_periodicity():
    """Test if checking after batch and after epoch according to configuration"""

    StopOnNaN(after_batch=True, after_epoch=False).after_epoch(epoch_data=_get_data(np.nan))
    with pytest.raises(TrainingTerminated):
        StopOnNaN(after_batch=True, after_epoch=False).after_batch(stream_name='train',
                                                                   batch_data=_get_data(np.nan)['train'])
    with pytest.raises(TrainingTerminated):
        StopOnNaN().after_epoch(epoch_data=_get_data(np.nan))
    StopOnNaN().after_batch(stream_name='train', batch_data=_get_data(np.nan)['train'])
