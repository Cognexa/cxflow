"""
Test module for :py:class:`cxflow.hooks.Check`.
"""

import numpy as np
import collections
import pytest

from cxflow.hooks.check import Check
from cxflow.hooks.abstract_hook import TrainingTerminated


_VAR = "accuracy"
_MIN_ACCURACY = 0.95
_MAX_EPOCH = 10
_CURRENT_EPOCH = 5


def _get_epoch_data():
    epoch_data = collections.OrderedDict([
        ('train', collections.OrderedDict([
            ('accuracy', 1),
        ])),
        ('test', collections.OrderedDict([
            ('accuracy', 0.5),
        ])),
        ('valid', collections.OrderedDict([
            ('accuracy', np.ones(10)),
        ]))
    ])
    return epoch_data


_PARAMETERS = [[_VAR, _MIN_ACCURACY, _MAX_EPOCH, "unknown"],
               ["not_present", _MIN_ACCURACY, _MAX_EPOCH],
               [_VAR, _MIN_ACCURACY, _MAX_EPOCH],
               [_VAR, _MIN_ACCURACY, _MAX_EPOCH, "train"],
               [_VAR, _MIN_ACCURACY, _MAX_EPOCH, "test"]]
_ERRORS = [KeyError, KeyError, TypeError, TrainingTerminated, ValueError]
_EPOCH = [_CURRENT_EPOCH, _CURRENT_EPOCH, _CURRENT_EPOCH, _CURRENT_EPOCH, 11]


@pytest.mark.parametrize('params, error, epoch', zip(_PARAMETERS, _ERRORS, _EPOCH))
def test_stream_raise(params, error, epoch):
    """Test raising error, when stream not in epoch_data."""
    hook = Check(*params)
    with pytest.raises(error):
        hook.after_epoch(epoch, _get_epoch_data())
