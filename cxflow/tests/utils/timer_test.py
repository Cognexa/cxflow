"""
Test module for profile utils (cxflow.utils.profile).
"""
import time

from cxflow.utils.profile import Timer


def test_empty_timer():
    """Test near zero measured time for no-op."""
    log = {}
    with Timer('noop', log):
        pass

    assert 'noop' in log
    assert len(log['noop']) == 1
    assert round(abs(log['noop'][0]-0), 4) == 0


def test_appending():
    """Test correct usage of the given profile."""
    log = {}

    n_iter = 100

    for _ in range(n_iter):
        with Timer('noop', log):
            pass

    assert len(log['noop']) == n_iter


def test_nested():
    """Test nesting of Timer usage pattern."""
    log = {}

    with Timer('timerop', log):
        with Timer('noop', log):
            pass

    assert 'noop' in log
    assert 'timerop' in log

    assert log['timerop'][0] > log['noop'][0]


def test_time():
    """Test reasonable precision in measuring time."""
    log = {}

    sleep_time = 1

    with Timer('sleep', log):
        time.sleep(sleep_time)

    assert round(abs(log['sleep'][0]-sleep_time), 1) == 0


def test_order():
    """Test correct order of measurements in the profile."""
    log = {}

    for i in range(3):

        with Timer('sleep', log):
            time.sleep(i)

        assert round(abs(log['sleep'][i]-i), 1) == 0
