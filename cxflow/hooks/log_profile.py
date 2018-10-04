"""
Module with a hook which reports the time profile data in the stanard logging.
"""
import logging
from itertools import chain
from typing import Iterable

from . import AbstractHook
from ..types import TimeProfile


class LogProfile(AbstractHook):
    """
    Summarize and log epoch profile via standard :py:mod:`logging`.

    Epoch profile contains info about time spent training, reading data etc. For full reference, see
    :py:class:`cxflow.MainLoop`.

    .. code-block:: yaml
        :caption: log the time profile after each epoch

        hooks:
          - LogProfile

    """

    def after_epoch_profile(self, epoch_id, profile: TimeProfile, train_stream_name: str, extra_streams: Iterable[str]) -> None:
        """
        Summarize and log the given epoch profile.

        The profile is expected to contain at least:
            - ``read_data_train``, ``eval_batch_train`` and ``after_batch_hooks_train`` entries produced by the train
              stream (if train stream name is `train`)
            - ``after_epoch_hooks`` entry

        :param profile: epoch timings profile
        :param extra_streams: enumeration of additional stream names
        """

        read_data_total = 0
        eval_total = 0
        train_total = sum(profile.get('eval_batch_{}'.format(train_stream_name), []))
        hooks_total = sum(profile.get('after_epoch_hooks', []))

        for stream_name in chain(extra_streams, [train_stream_name]):
            read_data_total += sum(profile.get('read_batch_' + stream_name, []))
            hooks_total += sum(profile.get('after_batch_hooks_' + stream_name, []))
            if stream_name != train_stream_name:
                eval_total += sum(profile.get('eval_batch_' + stream_name, []))

        logging.info('\tT read data:\t%f', read_data_total)
        logging.info('\tT train:\t%f', train_total)
        logging.info('\tT eval:\t%f', eval_total)
        logging.info('\tT hooks:\t%f', hooks_total)
