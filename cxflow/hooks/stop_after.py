"""
Module with a hook which stops the training after the specified number of epochs.
"""
import logging
from datetime import datetime
from typing import Optional

from . import AbstractHook, TrainingTerminated
from ..constants import CXF_DEFAULT_TRAIN_STREAM
from ..types import EpochData, Batch


class StopAfter(AbstractHook):
    """
    Stop the training after any of the specified conditions is met.

    .. code-block:: yaml

        :caption: stop the training after 500 epochs
        hooks:
          - StopAfter:
              epochs: 500


    .. code-block:: yaml

        :caption: stop the training after 1000 iterations of 1 hour whichever comes first
        hooks:
          - StopAfter:
              minutes: 60
              iterations: 1000
    """

    def __init__(self, epochs: Optional[int]=None, iterations: Optional[int]=None, minutes: Optional[float]=None,
                 train_stream_name: str=CXF_DEFAULT_TRAIN_STREAM, **kwargs):
        """
        Create new StopAfter hook.

        Possible stopping conditions are:

        - after the specified number of epochs
        - after the specified number of iterations (only train stream batches are counted as iterations)
        - after the model is trained for more than the specified number of minutes (and ``after_batch``, \
        ``after_epoch`` event is triggered)

        .. note:
            Multiple stopping conditions may be specified, the training is stopped when any of them is met.

        .. warning:
            At least one stopping condition must be specified.

        .. danger:
            When :py:class:`TrainingTerminated` exception is raised, no additional hook events are triggered except for
            :py:meth:`AbstractHook.after_training`.

        :param epochs: stop after the specified number of epochs
        :param iterations: stop after the specified number of iterations
        :param minutes: stop after the specified number minutes
        :raise ValueError: if no stopping condition is specified
        """
        super().__init__(**kwargs)
        if epochs is None and iterations is None and minutes is None:
            raise ValueError('No stopping condition was specified.')

        self._epochs = epochs
        self._iters = iterations
        self._minutes = minutes
        self._iters_done = 0
        self._training_start = None
        self._train_stream_name = train_stream_name

    def _check_train_time(self) -> None:
        """
        Stop the training if the training time exceeded ``self._minutes``.

        :raise TrainingTerminated: if the training time exceeded ``self._minutes``
        """
        if self._minutes is not None and (datetime.now() - self._training_start).total_seconds()/60 > self._minutes:
                raise TrainingTerminated('Training terminated after more than {} minutes'.format(self._minutes))

    def before_training(self):
        """Start measuring the train time."""
        self._training_start = datetime.now()

    def after_batch(self, stream_name: str, batch_data: Batch) -> None:
        """
        If ``stream_name`` equals to :py:attr:`cxflow.constants.TRAIN_STREAM`,
        increase the iterations counter and possibly stop the training; additionally, call :py:meth:`_check_train_time`.

        :param stream_name: stream name
        :param batch_data: ignored
        :raise TrainingTerminated: if the number of iterations reaches ``self._iters``
        """
        self._check_train_time()
        if self._iters is not None and stream_name == self._train_stream_name:
            self._iters_done += 1
            if self._iters_done >= self._iters:
                raise TrainingTerminated('Training terminated after iteration {}'.format(self._iters_done))

    def after_epoch(self, epoch_id: int, epoch_data: EpochData) -> None:
        """
        Stop the training if the ``epoch_id`` reaches ``self._epochs``; additionally, call :py:meth:`_check_train_time`.

        :param epoch_id: epoch id
        :param epoch_data: ignored
        :raise TrainingTerminated: if the ``epoch_id`` reaches ``self._epochs``
        """
        self._check_train_time()
        if self._epochs is not None and epoch_id >= self._epochs:
            logging.info('EpochStopperHook triggered')
            raise TrainingTerminated('Training terminated after epoch {}'.format(epoch_id))
