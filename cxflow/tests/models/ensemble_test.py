import os
import os.path as path

import numpy as np
import cxflow as cx
import pytest

from cxflow.constants import CXF_CONFIG_FILE
from cxflow.datasets import StreamWrapper
from cxflow.models.ensemble import major_vote, Ensemble


_DUMMY_CONFIG = """
model:
  name: test
  class: cxflow.tests.models.ensemble_test.DummyModel
  inputs: [images, extra_input]

"""


class DummyModel:
    """Dummy model multiplying the input with the last character of the ``restore_from`` path."""

    def __init__(self,
                 inputs: str,
                 outputs: str,
                 dataset: cx.AbstractDataset,
                 restore_from: str,
                 **_):

        self.dataset = dataset
        self.inputs = inputs
        self.outputs = outputs
        self.restore_from = restore_from

    def run(self, batch: cx.Batch, train: bool, stream: StreamWrapper) -> cx.Batch:
        return {'outputs': list(np.array(batch['inputs'])*int(self.restore_from[-1]))}


class TestMajorVoteTest:
    """major_vote function test case."""

    def test_major_vote(self):
        """Test if major_vote works properly."""
        vote1 = (1, 3, (5, 5), 12)
        vote2 = [1, 2, 2, 2]
        vote3 = [1, 2, (5, 5), 1]

        result = major_vote([vote1, vote2, vote3])
        assert [1, 2, (5, 5)] == list(result)[:3]
        assert result[-1] in {1, 2, 12}


@pytest.fixture
def create_models(tmpdir):

    def _create_models():
        """Create three dummy models in the tmp dir."""
        for i in [1, 2, 3]:
            model_dir = path.join(tmpdir, str(i))
            os.mkdir(model_dir)
            with open(path.join(model_dir, CXF_CONFIG_FILE), 'w') as config:
                config.write(_DUMMY_CONFIG)

    return _create_models


def test_init(tmpdir):
    """Test if Ensemble model ``__init__`` works properly"""
    with pytest.raises(AssertionError):
        Ensemble(inputs=['inputs'], outputs=['outputs'])
    with pytest.raises(AssertionError):
        Ensemble(inputs=['inputs'], outputs=['outputs'], models_root=tmpdir, aggregation='unknown')

    ensemble = Ensemble(inputs=['inputs'], outputs=['outputs'], models_root=tmpdir, aggregation='mean')
    assert ensemble._models is None

    # test eager loading
    ensemble2 = Ensemble(inputs=['inputs'], outputs=['outputs'], models_root=tmpdir, eager_loading=True)
    assert ensemble2._models is not None

    assert ensemble2.input_names == ['inputs']
    assert ensemble2.output_names == ['outputs']


def test_propagation(create_models, tmpdir):
    """Test if Ensemble model propagates all the arguments properly."""
    create_models()
    ensemble = Ensemble(inputs=['inputs'], outputs=['outputs'], models_root=tmpdir, aggregation='mean',
                        dataset='my_dataset')

    # test restore_from is propagated, test mean aggregation
    batch = {'inputs': [1., 1., 1.]}
    output = ensemble.run(batch, False, None)
    assert [2., 2., 2.] == list(output['outputs'])

    # test dataset is propagated
    assert 'my_dataset' == ensemble._models[0].dataset

    # test inputs parameter is propagated
    assert 'extra_input' not in batch


def test_major_vote(create_models, tmpdir):
    """Test if Ensemble model aggregates the results properly."""
    create_models()
    ensemble = Ensemble(inputs=['inputs'], outputs=['outputs'], models_root=tmpdir,
                        model_paths=['1', '1', '2'])

    # test major vote aggregation
    batch = {'inputs': [1., 1., 1.]}
    output = ensemble.run(batch, False, None)
    assert [1., 1., 1.] == list(output['outputs'])

    # test multi-dim data
    batch = {'inputs': [[1., 1., 2.], [1., 1., 2.]]}
    output = ensemble.run(batch, False, None)
    assert np.equal([[1., 1., 2.], [1., 1., 2.]], output['outputs']).all()


def test_raising(create_models, tmpdir):
    """Test if Ensemble model raises the exceptions as expected."""
    create_models()
    ensemble = Ensemble(inputs=['inputs'], outputs=['outputs'], models_root=tmpdir,
                        model_paths=['1', '1', '2'])

    with pytest.raises(ValueError):
        ensemble.run(None, True, None)
    with pytest.raises(NotImplementedError):
        ensemble.save()
    assert ensemble.restore_fallback is None
