import time
import pytest

from cxflow.datasets.stream_wrapper import StreamWrapper
from cxflow.types import Stream

from ..main_loop_test import SimpleDataset, _DATASET_ITERS


class FailingDataset(SimpleDataset):
    """SimpleDataset extension which raises exception after first train batch."""

    def train_stream(self) -> Stream:
        for batch in super().train_stream():
            yield batch
            raise RuntimeError('This exception is thrown on purpose. 👑 Keep calm and do deep learning. 👑')


def test_iteration():
    """Test both buffered and non-buffered iterations."""
    dataset = SimpleDataset()
    stream = StreamWrapper(dataset.train_stream)

    epoch = list(stream)
    assert len(epoch) == _DATASET_ITERS  # 1st epoch
    assert epoch == dataset.batches['train']
    epoch2 = []
    for batch in stream:  # 2nd epoch
        epoch2.append(batch)
    assert len(epoch2) == _DATASET_ITERS
    assert epoch + epoch2 == dataset.batches['train']

    dataset2 = SimpleDataset()
    buferred_stream = StreamWrapper(dataset2.train_stream, buffer_size=4)
    with pytest.raises(ValueError):
        next(buferred_stream)  # used outside with-resource

    with buferred_stream:
        buffered_epoch = list(buferred_stream)
    with buferred_stream:
        buffered_epoch2 = []
        for batch in buferred_stream:
            buffered_epoch2.append(batch)

    assert buffered_epoch + buffered_epoch2 == dataset2.batches['train']


def test_epoch_size():
    """Test both buffered and non-buffered iterations with fixed epoch size."""
    epoch_size = 10
    dataset = SimpleDataset()
    with StreamWrapper(dataset.train_stream, epoch_size=epoch_size) as stream:
        epoch = list(stream)
        assert len(epoch) == epoch_size
        epoch2 = list(stream)
        assert epoch+epoch2 == dataset.batches['train']

    dataset2 = SimpleDataset()
    with StreamWrapper(dataset2.train_stream, buffer_size=4, epoch_size=epoch_size) as stream2:
        buffered_epoch = list(stream2)
        assert len(buffered_epoch) == epoch_size
        buffered_epoch2 = list(stream2)

    assert buffered_epoch + buffered_epoch2 == dataset2.batches['train']


def test_breaking():
    """Test buffered stream works correctly with stream cut into multiple with-resource directives."""
    dataset = SimpleDataset()
    buffered_stream = StreamWrapper(dataset.train_stream, buffer_size=4)
    epoch = []
    for i in range(5):
        with buffered_stream:
            epoch.append(next(buffered_stream))
    epoch += list(buffered_stream)
    assert epoch == dataset.batches['train']


def test_enqueueing_thread_exception():
    dataset = FailingDataset()
    buffered_stream = StreamWrapper(dataset.train_stream, buffer_size=4)
    with buffered_stream:
        with pytest.raises(ChildProcessError):
            list(buffered_stream)


def test_empty_stream():
    stream = StreamWrapper(list)
    assert list(stream) == []
    with StreamWrapper(list, buffer_size=4) as buffered_stream:
        assert list(buffered_stream) == []


def test_simple_iterator():
    stream = StreamWrapper(lambda: [1, 2])
    next(stream)
    assert next(stream) == 2
    with pytest.raises(StopIteration):
        next(stream)


def test_allow_buffering():
    dataset = SimpleDataset()
    buffered_stream = StreamWrapper(dataset.train_stream, buffer_size=4)
    buffered_epochs = []
    with buffered_stream:
        with buffered_stream.allow_buffering:
            time.sleep(0.5)
            pass
        buffered_epochs = list(buffered_stream)
        with buffered_stream.allow_buffering:
            buffered_epochs += list(buffered_stream)
        with buffered_stream.allow_buffering:
            with buffered_stream.allow_buffering:
                buffered_epochs += list(buffered_stream)
    assert buffered_epochs == dataset.batches['train']
