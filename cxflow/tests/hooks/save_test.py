"""
Test module for saver hooks (:py:mod:`cxflow.hooks.save`).
"""

from typing import Mapping, List
import collections
import pytest

from cxflow.hooks.save import SaveEvery, SaveBest, SaveLatest
from cxflow.models.abstract_model import AbstractModel
from cxflow.types import EpochData


def _get_epoch_data(valid_loss_mean_val: float=3) -> EpochData:
    """Return testing epoch data."""

    epoch_data = collections.OrderedDict([
        ('train', collections.OrderedDict([
            ('accuracy', 1),
            ('loss', collections.OrderedDict([('mean', (3, 4))])),
        ])),
        ('test', collections.OrderedDict([
            ('accuracy', 0.5),
            ('loss', ('mean', 3)),
        ])),
        ('valid', collections.OrderedDict([
            ('accuracy', 0.8),
            ('loss', collections.OrderedDict([('mean', valid_loss_mean_val)])),
        ]))
    ])
    return epoch_data


class EmptyModel(AbstractModel):
    """The model which raises ``ValueError`` if save method is called."""

    def __init__(self, **kwargs):
        pass

    def run(self, batch: Mapping[str, object], train: bool) -> Mapping[str, object]:
        pass

    def save(self, name_suffix: str) -> str:
        raise ValueError

    @property
    def input_names(self) -> List[str]:
        pass

    @property
    def output_names(self) -> List[str]:
        pass

    @property
    def restore_fallback(self) -> str:
        return ''


class TestSaveAfter:
    """Test case for :py:class:`cxflow.hooks.SaverEvery`."""

    def test_raise_on_save_failure(self):
        """
        Test raising an exception if ``on_save_failure``
        parameter is not: error/warn/ignore.
        """
        with pytest.raises(AssertionError):
            SaveEvery(model=EmptyModel(), on_failure='unknown')

    def test_every_n_epochs(self):
        """Test saving/not saving every n epoch."""
        with pytest.raises(IOError):
            SaveEvery(model=EmptyModel(), n_epochs=3,
                      on_failure='error').after_epoch(epoch_id=30)
        SaveEvery(model=EmptyModel(), n_epochs=3).after_epoch(epoch_id=29)

    def test_no_raise(self):
        """
        Test whether an exception is not raised
        if ``on_save_failure`` is set to warn/ignore.
        """
        SaveEvery(model=EmptyModel(), n_epochs=3,
                  on_failure='warn').after_epoch(epoch_id=30)
        SaveEvery(model=EmptyModel(), n_epochs=3,
                  on_failure='ignore').after_epoch(epoch_id=30)


class TestSaveBest:
    """Test case for :py:class:`cxflow.hooks.SaveBest` hook."""

    def test_raise_invalid_on_save_failure(self):
        """
        Test raising an exception if ``on_save_failure``
        parameter is not: error/warn/ignore.
        """
        with pytest.raises(AssertionError):
            SaveBest(model=EmptyModel(), on_save_failure='unknown')

    def test_raise_invalid_condition(self):
        """
        Test raising an exception if condition
        parameter is not: min/max.
        """
        with pytest.raises(AssertionError):
            SaveBest(model=EmptyModel(), condition='unknown')

    _INVALID_DATA = [({'stream': 'unknown'}, KeyError),
                     ({'variable': 'unknown'}, KeyError),
                     ({'stream': 'test'}, TypeError),
                     ({'stream': 'train', 'aggregation': 'unknown'}, KeyError),
                     ({'stream': 'train'}, ValueError)]

    @pytest.mark.parametrize('params, error', _INVALID_DATA)
    def test_raise_invalid_epoch_data(self, params, error):
        """
        Test raising an exception if the hook is created
        with invalid arguments with respect to epoch data.
        """
        with pytest.raises(error):
            SaveBest(model=EmptyModel(), **params)._get_value(_get_epoch_data())

    def test_get_value(self):
        """Test getting proper value from epoch data."""
        assert SaveBest(model=EmptyModel())._get_value(_get_epoch_data()) == 3

    def test_save_value_better(self):
        """Test a model saving/not saving with respect to cond parameter."""

        def test_max_min_cond(cond, val_save_1, val_save_2, val_save_3):
            hook = SaveBest(model=EmptyModel(), stream='valid',
                            condition=cond, variable='loss')

            with pytest.raises(IOError):
                hook.after_epoch(_get_epoch_data(val_save_1))

            with pytest.raises(IOError):
                hook.after_epoch(_get_epoch_data(val_save_2))

            hook.after_epoch(_get_epoch_data(val_save_3))

        test_max_min_cond('max', 3, 5, 2)
        test_max_min_cond('min', 5, 3, 3)


class TestLatest:
    """Test case for :py:class:`cxflow.hooks.SaveLatest` hook."""

    def test_raise_invalid_on_save_failure(self):
        """
        Test raising an exception if ``on_save_failure``
        parameter is not: error/warn/ignore.
        """
        with pytest.raises(AssertionError):
            SaveLatest(model=EmptyModel(), on_save_failure='unknown')

    def test_save_latest(self):
        """Test a model saving."""

        hook = SaveLatest(model=EmptyModel())

        with pytest.raises(IOError):
            hook.after_epoch()

        with pytest.raises(IOError):
            hook.after_epoch()
