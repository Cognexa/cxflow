"""Test module for :py:class:`TrainingTrace`."""
import os.path as path
from datetime import datetime
import pytest

from cxflow.utils import TrainingTrace, TrainingTraceKeys
from cxflow.constants import CXF_TRACE_FILE


def test_bad_keys():
    """Test training trace key types being asserted."""
    trace = TrainingTrace()
    with pytest.raises(AssertionError):
        print(trace['train_begin'])
    with pytest.raises(AssertionError):
        trace['train_begin'] = 'value'


def test_set_get():
    """Test training trace __getitem__ and __setitem__."""
    value = 42
    trace = TrainingTrace(autosave=False)
    trace[TrainingTraceKeys.EPOCHS_DONE] = value
    assert trace[TrainingTraceKeys.EPOCHS_DONE] == value


def test_save_load(tmpdir):
    """Test training trace (auto)saving and loading."""
    trace = TrainingTrace(output_dir=tmpdir)
    now = datetime.now()
    trace[TrainingTraceKeys.TRAIN_BEGIN] = now  # should auto-save now
    loaded_trace = TrainingTrace.from_file(path.join(tmpdir, CXF_TRACE_FILE))
    assert loaded_trace[TrainingTraceKeys.TRAIN_BEGIN] == now


def test_output_dir():
    """Test ValueError is raised in save with not output_dir."""
    with pytest.raises(ValueError):
        TrainingTrace().save()
