"""Sequence model function test case."""

import os
import os.path as path

import numpy as np
import cxflow as cx
import pytest

from cxflow.constants import CXF_CONFIG_FILE
from cxflow.datasets import StreamWrapper
from cxflow.models.sequence import Sequence


_STEP1_CONFIG = """
model:
  name: step1
  class: cxflow.tests.models.sequence_test.Step1
  inputs: [images]
  outputs: [masks]

"""


_STEP2_CONFIG = """
model:
  name: step2
  class: cxflow.tests.models.sequence_test.Step2
  inputs: [images, masks]
  outputs: [classes]

"""


class Step1:
    def __init__(self, dataset, **_):
        self.dataset = dataset

    @property
    def input_names(self):
        return ['images']

    @property
    def output_names(self):
        return ['masks']

    def run(self, batch: cx.Batch, train: bool, stream: StreamWrapper) -> cx.Batch:
        return {'masks': np.zeros_like(batch['images'])}


class Step2:
    def __init__(self, **_):
        pass

    @property
    def input_names(self):
        return ['images', 'masks']

    @property
    def output_names(self):
        return ['classes']

    def run(self, batch: cx.Batch, train: bool, stream: StreamWrapper) -> cx.Batch:
        assert 'images' in batch
        assert 'masks' in batch
        return {'classes': np.ones(np.array(batch['images']).shape[0])}


@pytest.fixture
def create_models(tmpdir):

    def _create_models():
        """Create two step models in the tmp dir."""
        for name, config in zip(['step1', 'step2'], [_STEP1_CONFIG, _STEP2_CONFIG]):
            model_dir = path.join(tmpdir, name)
            os.mkdir(model_dir)
            with open(path.join(model_dir, CXF_CONFIG_FILE), 'w') as config_file:
                config_file.write(config)

    return _create_models


def test_init(create_models, tmpdir):
    """Test if Sequence model ``__init__`` works properly"""
    create_models()
    sequence = Sequence(models_root=tmpdir, model_paths=['step1', 'step2'])
    assert sequence._models is None

    # test eager loading
    sequence2 = Sequence(models_root=tmpdir, model_paths=['step1', 'step2'], eager_loading=True)
    assert sequence2._models is not None

    assert sequence2.input_names == ['images']
    assert list(sequence2.output_names) == ['masks', 'classes']


def test_run(create_models, tmpdir):
    """Test if Sequence model accumulates the outputs properly."""
    create_models()
    sequence = Sequence(models_root=tmpdir, model_paths=['step1', 'step2'], dataset='my_dataset')

    # outputs accumulating
    images = [[2.], [2.], [2.]]
    output = sequence.run({'images': images}, False, None)
    assert 'masks' in output
    assert 'classes' in output
    np.testing.assert_array_equal(np.zeros_like(images), output['masks'])
    np.testing.assert_array_equal(np.ones((3,)), output['classes'])

    # test dataset is propagated
    assert 'my_dataset' == sequence._models[0].dataset


def test_raising(create_models, tmpdir):
    """Test if Sequence model raises the exceptions as expected."""
    create_models()
    sequence = Sequence(models_root=tmpdir, model_paths=['step1', 'step2'])

    with pytest.raises(ValueError):
        sequence.run(None, True, None)
    with pytest.raises(NotImplementedError):
        sequence.save()
    assert sequence.restore_fallback is None
