from .datasets.abstract_dataset import AbstractDataset

from .hooks.abstract_hook import AbstractHook, TrainingTerminated
from .nets.abstract_net import AbstractNet

import numpy as np

from collections import defaultdict
import logging
import typing


class NetworkManager:
    """Train the network, manage hooks etc."""

    def __init__(self, net: AbstractNet, dataset: AbstractDataset, hooks: typing.Iterable[AbstractHook]=[]):
        """
        :param net: trained network
        :param dataset: loaded dataset
        :param hooks: list of hooks
        """

        self.net = net
        self.dataset = dataset
        self.hooks = hooks

        self._extra_sources_warned = False

        if self.net.skip_incomplete_batches and not hasattr(self.net, 'batch_size'):
            logging.error('Config does not contain `net.batch_size` even though `net.skip_incomplete_batches` is set '
                          'to true. Either disable `net.skip_incomplete_batches` or set `net.batch_size`.')
            raise ValueError('Invalid config')

        if not self.net.skip_incomplete_batches and hasattr(self.net, 'batch_size'):
            logging.warning('Config contains `net.batch_size` even though `net.skip_incomplete_batches` si set to '
                            'false.' 'Note that `net.batch_size` has no effect since the batch sizes are set in '
                            '`stream.<name>.batch_size`. Please consider removing `net.batch_size`.')


    def _run_batch(self, train: bool, batch: typing.Mapping[str, typing.Any]) -> typing.Mapping[str, np.ndarray]:
        """Process a single batch (either train or eval)."""

        # check stream sources
        if set(self.net.io['in']) < set(batch.keys()):
            if self.net.ignore_extra_sources:
                if not self._extra_sources_warned:
                    logging.warning('Some sources provided by the stream does not match net placeholders. This might '
                                    'be an error. Ignoring. Set `net.ignore_extra_sources` to False in order to make '
                                    'this' 'an error. Extra sources: %s', set(batch.keys()) - set(self.net.io['in']))
                    self._extra_sources_warned = True
            else:
                logging.error('Some sources provided by the stream does not match net placeholders. Set'
                              '`net.ignore_extra_sources` to True in order to ignore this error'
                              'Extra sources: %s', set(batch.keys()) - set(self.net.io['in']))
                raise ValueError()

        elif set(self.net.io['in']) > set(batch.keys()):
            logging.error('Stream does not provide all required sources. Missing sources: %s',
                          set(self.net.io['in']) - set(batch.keys()))

        # setup the feed dict
        feed_dict = {}
        for placeholder_name in self.net.io['in']:
            try:
                feed_dict[getattr(self.net, placeholder_name)] = batch[placeholder_name]
            except AttributeError as e:
                logging.error('Placeholder "%s" not found in the net. Make sure it is saved as an object property.',
                              placeholder_name)
                raise e

        # setup fetches
        fetches = [self.net.train_op] if train else []
        for to_eval in self.net.io['out']:
            try:
                fetches.append(getattr(self.net, to_eval))
            except AttributeError as e:
                logging.error('Net does not contain a required output "%s". Make sure it is saved as an object'
                              'property.', to_eval)

        # run the computational graph for one batch
        batch_res = self.net.session.run(fetches=fetches, feed_dict=feed_dict)

        # zip the string names with results
        if train:
            return dict(zip(self.net.io['out'], batch_res[1:]))
        else:
            return dict(zip(self.net.io['out'], batch_res))

    def _run_epoch(self, stream: AbstractDataset.Stream, train: bool, stream_type: str):
        """
        Iterate through the stream
        :param stream: Iterable stream
        :param train: if set to true, the network will be trained
        :param stream_type: {train, valid, test}
        :return: epoch summary results
        """

        n_batches = 0
        summed_results = defaultdict(float)

        for bid, batch in enumerate(stream):
            if self.net.skip_incomplete_batches:
                if len(batch[list(batch.keys())[0]]) < self.net.batch_size:
                    continue

            n_batches += 1
            batch_result = self._run_batch(train=train, batch=batch)

            for hook in self.hooks:
                hook.after_batch(stream_type=stream_type, results=batch_result)

            for name, value in batch_result.items():
                try:
                    summed_results[name] += value
                except Exception as e:
                    logging.error('Cannot sum results "%s"', name)
                    raise e

        for name in summed_results.keys():
            summed_results[name] /= n_batches

        return summed_results

    def train_by_stream(self, stream: AbstractDataset.Stream):
        """Given a stream and batch size, train the network on this stream."""

        return self._run_epoch(stream=stream, train=True, stream_type='train')

    def evaluate_stream(self, stream: AbstractDataset.Stream, stream_type: str):
        """Given a stream and batch size, evaluate the network on this stream."""

        return self._run_epoch(stream=stream, train=False, stream_type=stream_type)

    def run_main_loop(self, run_test_stream: bool) -> None:
        """
        Start the main loop
        :param run_test_stream: should the test stream be evaluated?
        """

        epoch_id = 0

        for hook in self.hooks:
            hook.before_training()

        valid_results = self.evaluate_stream(stream=self.dataset.create_valid_stream(), stream_type='valid')

        if run_test_stream:
            # logging.debug('%s', self.dataset.create_test_stream())
            # logging.debug('%s', [method for method in dir(self.dataset) if callable(getattr(self.dataset, method))])
            # quit()
            try:
                test_results = self.evaluate_stream(stream=self.dataset.create_test_stream(), stream_type='test')
            except AttributeError as e:
                logging.error('Dataset does not provide test stream even though it was specified in the config. Either'
                              'implement `create_test_stream` method or delete `stream.test` section in the config.')
                raise e
        else:
            test_results = None

        for hook in self.hooks:
            hook.before_first_epoch(valid_results=valid_results, test_results=test_results)

        while True:
            epoch_id += 1

            train_results = self.train_by_stream(stream=self.dataset.create_train_stream())
            valid_results = self.evaluate_stream(stream=self.dataset.create_valid_stream(), stream_type='valid')

            if run_test_stream:
                try:
                    test_results = self.evaluate_stream(stream=self.dataset.create_test_stream(), stream_type='test')
                except AttributeError as e:
                    logging.error(
                        'Dataset does not provide test stream even though it was specified in the config. Either'
                        'implement `create_test_stream` method or delete `stream.test` section in the config.')
                    raise e
            else:
                test_results = None

            try:
                for hook in self.hooks:
                    hook.after_epoch(epoch_id=epoch_id, train_results=train_results, valid_results=valid_results,
                                     test_results=test_results)
            except TrainingTerminated as e:
                logging.info('Training terminated by a hook: %s', e)
                break

        for hook in self.hooks:
            hook.after_training()
