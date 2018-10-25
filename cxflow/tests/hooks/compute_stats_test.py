"""
Test module for stats hook (cxflow.hooks.compute_stats_hook).
"""

import numpy as np
import pytest

from cxflow.hooks.compute_stats import ComputeStats


def get_batch(batch_id):
    accuracy = batch_id * (np.ones(5) + 1)
    nan_accuracy = batch_id * (np.ones(5) + 2)
    nan_accuracy[0] = nan_accuracy[3] = np.nan
    loss = batch_id * (np.ones(5) + 3)
    return {'accuracy': accuracy,
            'nan_accuracy': nan_accuracy,
            'loss': loss}


def test_raise_on_init():
    """Tests raising error if specified aggregation is not supported."""

    variables = [{'loss': ['mean', 'median', 'max']},
                 {'accuracy': ['mean', 'median', 'not_supported']}]
    with pytest.raises(ValueError):
        ComputeStats(variables=variables)


def test_compute_save_stats():
    """Tests correctness of computed aggregations and their saving."""

    variables = ['loss',
                 {'accuracy': ['mean', 'std', 'min', 'max', 'median',
                               'nanmean', 'nanfraction', 'nancount']},
                 {'nan_accuracy': ['mean', 'nanmean', 'nanfraction', 'nancount']}]

    hook = ComputeStats(variables=variables)

    epoch_data = {'train': {'accuracy': None, 'nan_accuracy': None, 'loss': None},
                  'test': {'accuracy': None, 'nan_accuracy': None, 'loss': None}}

    for batch_id in range(1, 3):
        for stream in epoch_data.keys():
            hook.after_batch(stream, get_batch(batch_id))

    hook.after_epoch(epoch_data)

    valid_aggrs = {'loss': {'mean': 6.0},
                   'accuracy': {'mean': 3, 'std': 1, 'min': 2, 'max': 4, 'median': 3,
                                'nanmean': 3, 'nanfraction': 0., 'nancount': 0},
                   'nan_accuracy': {'mean': np.nan, 'nanmean': 4.5,
                                    'nanfraction': 0.4, 'nancount': 4}}
    valid_aggrs = {'train': valid_aggrs, 'test': valid_aggrs}

    assert epoch_data.keys() == \
                     valid_aggrs.keys()
    for stream in epoch_data.keys():
        assert epoch_data[stream].keys() == \
                         valid_aggrs[stream].keys()
        for variable in epoch_data[stream]:
            assert epoch_data[stream][variable].keys() == \
                             valid_aggrs[stream][variable].keys()
            for aggr in epoch_data[stream][variable]:
                # to compare NaN values, NumPy assert is required
                np.testing.assert_equal(epoch_data[stream][variable][aggr],
                                        valid_aggrs[stream][variable][aggr])
