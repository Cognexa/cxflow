"""
Test module for saver hook (cxflow.hooks.saver_hook).
"""

from typing import Mapping, List
import collections

from cxflow.tests.test_core import CXTestCase
from cxflow.hooks.saver_hook import SaverHook, BestSaverHook, AbstractHook
from cxflow.models.abstract_model import AbstractModel


def _get_epoch_data(valid_loss_mean_val: float=3) -> AbstractHook.EpochData:
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
    """The model which raises `ValueError` if save method is called."""

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
    def restore_fallback_module(self) -> str:
        return ''

    @property
    def restore_fallback_class(self) -> str:
        return ''


class SaverHookTest(CXTestCase):
    """Test case for SaverHook."""

    def test_raise_on_save_failure(self):
        """
        Test raising an exception if `on_save_failure`
        parameter is not: error/warn/ignore.
        """
        with self.assertRaises(AssertionError):
            SaverHook(model=EmptyModel(), on_save_failure='unknown')

    def test_every_n_epochs(self):
        """Test saving/not saving every n epoch."""
        with self.assertRaises(IOError):
            SaverHook(model=EmptyModel(), save_every_n_epochs=3,
                      on_save_failure='error').after_epoch(epoch_id=30)
        SaverHook(model=EmptyModel(), save_every_n_epochs=3).after_epoch(epoch_id=29)

    def test_no_raise(self):
        """
        Test whether an exception is not raised
        if `on_save_failure` is set to warn/ignore.
        """
        SaverHook(model=EmptyModel(), save_every_n_epochs=3,
                  on_save_failure='warn').after_epoch(epoch_id=30)
        SaverHook(model=EmptyModel(), save_every_n_epochs=3,
                  on_save_failure='ignore').after_epoch(epoch_id=30)


class BestSaverHookTest(CXTestCase):
    """Test case for BestSaverHook."""

    def test_raise_invalid_on_save_failure(self):
        """
        Test raising an exception if `on_save_failure`
        parameter is not: error/warn/ignore.
        """
        with self.assertRaises(AssertionError):
            BestSaverHook(model=EmptyModel(), on_save_failure='unknown')

    def test_raise_invalid_condition(self):
        """
        Test raising an exception if condition
        parameter is not: min/max.
        """
        with self.assertRaises(AssertionError):
            BestSaverHook(model=EmptyModel(), condition='unknown')

    def test_raise_invalid_epoch_data(self):
        """
        Test raising an exception if the hook is created
        with invalid arguments with respect to epoch data.
        """
        with self.assertRaises(KeyError):
            BestSaverHook(model=EmptyModel(), stream='unknown')._get_value(_get_epoch_data())

        with self.assertRaises(KeyError):
            BestSaverHook(model=EmptyModel(), variable='unknown')._get_value(_get_epoch_data())

        with self.assertRaises(TypeError):
            BestSaverHook(model=EmptyModel(), stream='test')._get_value(_get_epoch_data())

        with self.assertRaises(KeyError):
            BestSaverHook(model=EmptyModel(), stream='train', aggregation='unknown')._get_value(_get_epoch_data())

        with self.assertRaises(ValueError):
            BestSaverHook(model=EmptyModel(), stream='train')._get_value(_get_epoch_data())

    def test_get_value(self):
        """Test getting proper value from epoch data."""
        self.assertEqual(BestSaverHook(model=EmptyModel())._get_value(_get_epoch_data()), 3)

    def test_save_value_better(self):
        """Test a model saving/not saving with respect to cond parameter."""

        def test_max_min_cond(cond, val_save_1, val_save_2, val_save_3):
            hook = BestSaverHook(model=EmptyModel(), stream='valid',
                                 condition=cond, variable='loss')

            with self.assertRaises(IOError):
                hook.after_epoch(_get_epoch_data(val_save_1))

            with self.assertRaises(IOError):
                hook.after_epoch(_get_epoch_data(val_save_2))

            hook.after_epoch(_get_epoch_data(val_save_3))

        test_max_min_cond('max', 3, 5, 2)
        test_max_min_cond('min', 5, 3, 3)
