"""
**cxflow** main loop for training models.

The MainLoop requires AbstractModel, AbstractDataset and a list of AbstractHooks.
Having all that, it manages iterating through streams, training and hooks execution.
"""
import logging
from typing import Iterable, Callable, List, Dict, Optional
from collections import OrderedDict

from .datasets import AbstractDataset
from .models.abstract_model import AbstractModel
from .hooks.abstract_hook import AbstractHook, TrainingTerminated
from .utils import Timer, TrainingTrace, TrainingTraceKeys
from .utils.misc import CaughtInterrupts
from .datasets.stream_wrapper import StreamWrapper
from .constants import CXF_DEFAULT_TRAIN_STREAM, CXF_PREDICT_STREAM
from .types import EpochData


class MainLoop(CaughtInterrupts):   # pylint: disable=too-many-instance-attributes
    """**cxflow** main loop for training and model inference."""

    EMPTY_ACTIONS = ['ignore', 'warn', 'error']
    """Possible actions to be taken when a batch/stream is empty."""
    UNUSED_SOURCE_ACTIONS = ['ignore', 'warn', 'error']
    """Possible actions to be taken when a stream source is unused by the trained model."""

    def __init__(self,   # pylint: disable=too-many-arguments
                 model: AbstractModel, dataset: AbstractDataset,
                 hooks: Iterable[AbstractHook]=(),
                 train_stream_name: str=CXF_DEFAULT_TRAIN_STREAM,
                 extra_streams: List[str]=(),  # pylint: disable=invalid-sequence-index
                 buffer: int=0,
                 on_empty_batch: str='error',
                 on_empty_stream: str='error',
                 on_unused_sources: str='warn',
                 fixed_batch_size: Optional[int]=None,
                 fixed_epoch_size: Optional[int]=None,
                 skip_zeroth_epoch: bool=False):
        """
        :param model: trained model
        :param dataset: loaded dataset
        :param hooks: training hooks
        :param train_stream_name: name of the training stream
        :param extra_streams: additional stream names to be evaluated between epochs
        :param buffer: size of the batch buffer, 0 means no buffer
        :param on_empty_batch: action to take when batch is empty; one of :py:attr:`MainLoop.EMPTY_ACTIONS`
        :param on_empty_stream: action to take when stream is empty; one of :py:attr:`MainLoop.EMPTY_ACTIONS`
        :param on_unused_sources: action to take when stream provides an unused sources; one of
            :py:attr:`UNUSED_SOURCE_ACTIONS`
        :param fixed_batch_size: if specified, main_loop removes all batches that do not have the specified size
        :param fixed_epoch_size: if specified, cut the train stream to epochs of at most ``fixed_epoch_size`` batches
        :param skip_zeroth_epoch: if specified, main loop skips the 0th epoch
        :raise AssertionError: in case of unsupported value of ``on_empty_batch``, ``on_empty_stream`` or \
        ``on_unused_sources``
        """
        assert on_empty_batch in MainLoop.EMPTY_ACTIONS
        assert on_empty_stream in MainLoop.EMPTY_ACTIONS
        assert on_unused_sources in MainLoop.UNUSED_SOURCE_ACTIONS

        self._model = model
        self._dataset = dataset
        self._hooks = hooks
        self._buffer = buffer
        self._on_empty_batch = on_empty_batch
        self._on_empty_stream = on_empty_stream
        self._on_unused_sources = on_unused_sources
        self._fixed_batch_size = fixed_batch_size
        self._fixed_epoch_size = fixed_epoch_size
        self._extra_sources_warned = False
        self._epoch_profile = {}
        self._train_stream_name = train_stream_name
        self._extra_streams = list(extra_streams)
        self._skip_zeroth_epoch = skip_zeroth_epoch
        self._streams = {}
        self._epochs_done = None

        super().__init__()

    @property
    def epochs_done(self) -> Optional[int]:
        """Number of training epochs done in the last call of :py:meth:`self._run_training`."""
        return self._epochs_done

    @property
    def fixed_epoch_size(self) -> Optional[int]:
        """Fixed epoch size parameter as specified in :py:meth:`self.__init__`."""
        return self._fixed_epoch_size

    @property
    def extra_streams(self) -> List[str]:
        """List of extra stream names as specified in :py:meth:`self.__init__`."""
        return self._extra_streams

    def _create_epoch_data(self, streams: Optional[Iterable[str]]=None) -> EpochData:
        """Create empty epoch data double dict."""
        if streams is None:
            streams = [self._train_stream_name] + self._extra_streams
        return OrderedDict([(stream_name, OrderedDict()) for stream_name in streams])

    def _check_sources(self, batch: Dict[str, object]) -> None:
        """
        Check for unused and missing sources.

        :param batch: batch to be checked
        :raise ValueError: if a source is missing or unused and ``self._on_unused_sources`` is set to ``error``
        """
        unused_sources = [source for source in batch.keys() if source not in self._model.input_names]
        missing_sources = [source for source in self._model.input_names if source not in batch.keys()]
        # check stream sources
        if unused_sources:
            if self._on_unused_sources == 'warn' and not self._extra_sources_warned:
                logging.warning('Some sources provided by the stream do not match model placeholders. Set '
                                '`main_loop.on_unused_sources` to `ignore` in order to suppress this warning. '
                                'Extra sources: %s', unused_sources)
                self._extra_sources_warned = True
            elif self._on_unused_sources == 'error':
                raise ValueError('Some sources provided by the stream do not match model placeholders. Set'
                                 '`main_loop.on_unused_sources` to `warn` in order to suppress this error.\n'
                                 'Extra sources: {}'.format(unused_sources))

        if missing_sources:
            raise ValueError('Stream does not provide all required sources. Missing sources: {}'
                             .format(missing_sources))

    def _run_epoch(self, stream: StreamWrapper, train: bool) -> None:
        """
        Iterate through the given stream and evaluate/train the model with the received batches.

        Calls :py:meth:`cxflow.hooks.AbstractHook.after_batch` events.

        :param stream: stream to iterate
        :param train: if set to ``True``, the model will be trained
        :raise ValueError: in case of empty batch when ``on_empty_batch`` is set to ``error``
        :raise ValueError: in case of empty stream when ``on_empty_stream`` is set to ``error``
        :raise ValueError: in case of two batch variables having different lengths
        """
        nonempty_batch_count = 0
        for i, batch_input in enumerate(stream):
            self.raise_check_interrupt()

            batch_sizes = {len(source) for source in batch_input.values()}
            if len(batch_sizes) == 0 or batch_sizes == {0}:
                if self._on_empty_batch == 'warn':
                    logging.warning('%i-th batch in stream `%s` appears to be empty (%i-th empty batch in total). Set '
                                    '`main_loop.on_empty_batch` to `ignore` in order to suppress this warning.',
                                    i, stream.name, nonempty_batch_count)
                elif self._on_empty_batch == 'error':
                    raise ValueError('{}-th batch in stream `{}` appears to be empty ({}-th empty batch in total). Set '
                                     '`main_loop.on_empty_batch` to `warn` in order to change this error into warning; '
                                     'set to `ignore` to remove it.'.format(i, stream.name, nonempty_batch_count))
                continue
            elif self._fixed_batch_size:
                if batch_sizes != {self._fixed_batch_size}:
                    var, len_ = [(k, len(v)) for k, v in batch_input.items() if len(v) != self._fixed_batch_size][0]
                    logging.debug('%i-th batch in stream `%s` has variable `%s` of length %i inconsistent with '
                                  '`main_loop.fixed_size` = %i', i, stream.name, var, len_, self._fixed_batch_size)
                    continue
            nonempty_batch_count += 1

            self._check_sources(batch_input)

            with Timer('eval_batch_{}'.format(stream.name), self._epoch_profile):
                batch_output = self._model.run(batch=batch_input, train=train, stream=stream)
            assert set(batch_input.keys()).isdisjoint(set(batch_output)), 'Batch inputs and outputs must not overlap.'

            with Timer('after_batch_hooks_{}'.format(stream.name), self._epoch_profile):
                batch_data = {**batch_input, **batch_output}
                for hook in self._hooks:
                    hook.after_batch(stream_name=stream.name, batch_data=batch_data)
        if nonempty_batch_count == 0:
            if self._on_empty_stream == 'warn':
                logging.warning('Stream `%s` appears to be empty. Set `main_loop.on_empty_stream` to `ignore` in order '
                                'to suppress this warning.', stream.name)
            elif self._on_empty_stream == 'error':
                raise ValueError('Stream `{}` appears to be empty. Set '
                                 '`main_loop.on_empty_stream` to `warn` in order to change this error into warning; '
                                 'set to `ignore` to remove it.'.format(stream.name))


    def train_by_stream(self, stream: StreamWrapper) -> None:
        """
        Train the model with the given stream.

        :param stream: stream to train with
        """
        self._run_epoch(stream=stream, train=True)

    def evaluate_stream(self, stream: StreamWrapper) -> None:
        """
        Evaluate the given stream.

        :param stream: stream to be evaluated
        :param stream_name: stream name
        """
        self._run_epoch(stream=stream, train=False)

    def get_stream(self, stream_name: str) -> StreamWrapper:
        """
        Get a :py:class:`StreamWrapper` with the given name.

        :param stream_name: stream name
        :return: dataset function name providing the respective stream
        :raise AttributeError: if the dataset does not provide the function creating the stream
        """
        if stream_name not in self._streams:
            stream_fn_name = '{}_stream'.format(stream_name)
            try:
                stream_fn = getattr(self._dataset, stream_fn_name)
                stream_epoch_limit = -1
                if self._fixed_epoch_size is not None and stream_name == self._train_stream_name:
                    stream_epoch_limit = self._fixed_epoch_size
                self._streams[stream_name] = StreamWrapper(stream_fn, buffer_size=self._buffer,
                                                           epoch_size=stream_epoch_limit, name=stream_name,
                                                           profile=self._epoch_profile)
            except AttributeError as ex:
                raise AttributeError('The dataset does not have a function for creating a stream named `{}`. '
                                     'The function has to be named `{}`.'.format(stream_name, stream_fn_name)) from ex
        return self._streams[stream_name]

    def _run_zeroth_epoch(self, streams: Iterable[str]) -> None:
        """
        Run zeroth epoch on the specified streams.

        Calls
            - :py:meth:`cxflow.hooks.AbstractHook.after_epoch`

        :param streams: stream names to be evaluated
        """
        for stream_name in streams:
            with self.get_stream(stream_name) as stream:
                self.evaluate_stream(stream)

        epoch_data = self._create_epoch_data(streams)
        for hook in self._hooks:
            hook.after_epoch(epoch_id=0, epoch_data=epoch_data)

    def _try_run(self, run_func: Callable[[], None]) -> None:
        """
        Try running the given function (training/prediction).

        Calls
            - :py:meth:`cxflow.hooks.AbstractHook.before_training`
            - :py:meth:`cxflow.hooks.AbstractHook.after_training`

        :param run_func: function to be run
        """
        # Initialization: before_training
        for hook in self._hooks:
            hook.before_training()

        try:
            run_func()
        except TrainingTerminated as ex:
            logging.info('Training terminated: %s', ex)

        # After training: after_training
        for hook in self._hooks:
            hook.after_training()

    def run_training(self, trace: Optional[TrainingTrace]=None) -> None:
        """
        Run the main loop in the training mode.

        Calls
            - :py:meth:`cxflow.hooks.AbstractHook.after_epoch`
            - :py:meth:`cxflow.hooks.AbstractHook.after_epoch_profile`
        """
        for stream_name in [self._train_stream_name] + self._extra_streams:
            self.get_stream(stream_name)

        def training():
            logging.debug('Training started')
            self._epochs_done = 0

            # Zeroth epoch: after_epoch
            if not self._skip_zeroth_epoch:
                logging.info('Evaluating 0th epoch')
                self._run_zeroth_epoch([self._train_stream_name] + self._extra_streams)
                logging.info('0th epoch done\n\n')

            # Training loop: after_epoch, after_epoch_profile
            while True:
                epoch_id = self._epochs_done + 1
                logging.info('Training epoch %s', epoch_id)
                self._epoch_profile.clear()
                epoch_data = self._create_epoch_data()

                with self.get_stream(self._train_stream_name) as stream:
                    self.train_by_stream(stream)

                for stream_name in self._extra_streams:
                    with self.get_stream(stream_name) as stream:
                        self.evaluate_stream(stream)

                with Timer('after_epoch_hooks', self._epoch_profile):
                    for hook in self._hooks:
                        hook.after_epoch(epoch_id=epoch_id, epoch_data=epoch_data)

                for hook in self._hooks:
                    hook.after_epoch_profile(epoch_id=epoch_id, profile=self._epoch_profile,
                                             train_stream_name=self._train_stream_name,
                                             extra_streams=self._extra_streams)
                self._epochs_done = epoch_id
                if trace is not None:
                    trace[TrainingTraceKeys.EPOCHS_DONE] = self._epochs_done
                logging.info('Epoch %s done\n\n', epoch_id)

        self._try_run(training)

    def run_evaluation(self, stream_name: str) -> None:
        """
        Run the main loop with the given stream in the prediction mode.

        :param stream_name: name of the stream to be evaluated
        """
        def prediction():
            logging.info('Running prediction')
            self._run_zeroth_epoch([stream_name])
            logging.info('Prediction done\n\n')
        self._try_run(prediction)
