"""
Test module for StopAfter hook (cxflow.hooks.stop_after_hook).
"""
import time

from cxflow.main_loop import MainLoop
from cxflow.hooks.stop_after import StopAfter
from cxflow.hooks.abstract_hook import TrainingTerminated
from cxflow.constants import CXF_DEFAULT_TRAIN_STREAM
import pytest

NOTRAIN_STREAM_NAME = 'valid'
assert CXF_DEFAULT_TRAIN_STREAM != NOTRAIN_STREAM_NAME


def test_no_conditions_raise():
    """Test if ``__init__`` raises ValueError when no stopping condition is specified."""
    with pytest.raises(ValueError):
        StopAfter()


def test_stop_after_epochs():
    """Test epochs stopping condition."""
    # Test hook does not terminate training prematurely.
    hook = StopAfter(epochs=10)
    hook.after_epoch(epoch_id=5, epoch_data=None)
    hook.after_epoch(epoch_id=9, epoch_data=None)

    # Test hook does terminate the training correctly
    with pytest.raises(TrainingTerminated):
        hook.after_epoch(10, None)
    with pytest.raises(TrainingTerminated):
        hook.after_epoch(20, None)


def test_stop_after_iters():
    """Test iterations stopping condition."""
    # Test hook does not terminate training prematurely.
    hook = StopAfter(iterations=10)
    for i in range(15):
        hook.after_batch(stream_name=NOTRAIN_STREAM_NAME, batch_data=None)

    for i in range(9):
        hook.after_batch(stream_name=CXF_DEFAULT_TRAIN_STREAM, batch_data=None)

    hook.after_batch(stream_name=NOTRAIN_STREAM_NAME, batch_data=None)

    # Test hook does terminate the training correctly
    with pytest.raises(TrainingTerminated):
        hook.after_batch(stream_name=CXF_DEFAULT_TRAIN_STREAM, batch_data=None)


def test_stop_after_minutes():
    """Test iterations stopping condition."""
    # Test hook does not terminate training prematurely.
    hook = StopAfter(minutes=1./60)
    hook.before_training()
    hook.after_batch(stream_name=NOTRAIN_STREAM_NAME, batch_data=None)
    hook.after_epoch(epoch_id=1, epoch_data=None)
    time.sleep(1)

    # Test hook does terminate the training correctly
    with pytest.raises(TrainingTerminated):
        hook.after_batch(stream_name=NOTRAIN_STREAM_NAME, batch_data=None)
    with pytest.raises(TrainingTerminated):
        hook.after_epoch(epoch_id=1, epoch_data=None)
