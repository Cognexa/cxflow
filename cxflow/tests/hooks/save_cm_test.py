"""
Test case for :py:class:`cxflow.hooks.SaveConfusionMatrix hook.
"""
import os
import matplotlib
import pytest

from cxflow.hooks.save_cm import SaveConfusionMatrix
from ..main_loop_test import SimpleDataset


class MockDataset(SimpleDataset):

    @staticmethod
    def num_classes():
        return 4

    @staticmethod
    def num_classes_bad():
        return 1  # too small number

    @staticmethod
    def classes_names():
        return ['a', 'b', 'c', 'd']

    @staticmethod
    def classes_names_bad():
        return ['a']  # too few names


def run_hook(hook,
             batch_data: dict = {'labels': [0, 1], 'predictions': [0, 1], 'masks': [1, 0]},
             epoch_data: dict = {'train': {'accuracy': 1}}):
    """
    Run hook's methods `after_batch` and `after_epoch`
    Returns modified epoch_data
    """
    hook.after_batch(stream_name='train', batch_data=batch_data)
    hook.after_epoch(epoch_id=0, epoch_data=epoch_data)
    return epoch_data


_WRONG_INPUTS = [({'labels_name': 'fake'}, KeyError),
                 ({'predictions_name': 'fake'}, KeyError),
                 ({'cmap': 'fake'}, ValueError),
                 ({'figure_action': 'non_existing'}, ValueError),
                 ({'classes_names': ['just_one']}, AssertionError),
                 ({'classes_names_method_name': 'classes_names_bad'}, AssertionError),
                 ({'num_classes_method_name': 'num_classes_bad', 'classes_names_method_name': 'not_existing'},
                  AssertionError)]


@pytest.mark.parametrize('params, error', _WRONG_INPUTS)
def test_wrong_inputs(params, error, tmpdir):
    with pytest.raises(error):
        hook = SaveConfusionMatrix(dataset=MockDataset(), output_dir=tmpdir, **params)
        run_hook(hook)


_CORRECT_INPUTS = [{'classes_names': ['first', 'second']},
                   {'classes_names_method_name': 'not_existing'},
                   {'classes_names_method_name': 'not_existing', 'num_classes_method_name': 'not_existing'},
                   {'cmap': 'Greens'},
                   {'normalize': False}]


@pytest.mark.parametrize('params', _CORRECT_INPUTS)
def test_correct_inputs(params, tmpdir):
    hook = SaveConfusionMatrix(dataset=MockDataset(), output_dir=tmpdir, **params)
    run_hook(hook)


def test_after_epoch(tmpdir):

    # test saving .png
    hook = SaveConfusionMatrix(dataset=MockDataset(), output_dir=tmpdir)
    run_hook(hook)
    assert os.path.exists(os.path.join(tmpdir, 'confusion_matrix_epoch_0_train.png'))
    # test storing .png
    hook = SaveConfusionMatrix(dataset=MockDataset(), output_dir='', figure_action='store')
    epoch_data = run_hook(hook)
    assert tuple(epoch_data['train']['confusion_heatmap'].shape) == (480, 640, 3)

    # test changing figure size
    hook = SaveConfusionMatrix(dataset=MockDataset(), output_dir='', figure_action='store', figsize=(10, 15))
    epoch_data = run_hook(hook)
    dpi = matplotlib.rcParams['figure.dpi']
    assert tuple(epoch_data['train']['confusion_heatmap'].shape) == (15*dpi, 10*dpi, 3)

    # test whether using mask_name does not crash
    hook = SaveConfusionMatrix(dataset=MockDataset(), output_dir=tmpdir,
                               classes_names=['first', 'second'], mask_name='masks')
    run_hook(hook)

    # test correct input parameters with batch data
    hook = SaveConfusionMatrix(dataset=MockDataset(), output_dir=tmpdir,
                               labels_name='special_labels', predictions_name='special_predictions')
    run_hook(hook, batch_data={'special_labels': [0, 1], 'special_predictions': [0, 1]})
