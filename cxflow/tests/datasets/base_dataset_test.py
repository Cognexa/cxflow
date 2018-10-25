from typing import Optional, Tuple
from collections import OrderedDict
import logging

import numpy as np
import tabulate

import cxflow.datasets.base_dataset
from cxflow.datasets.base_dataset import BaseDataset
from cxflow.types import Stream


class MockDataset(BaseDataset):
    """Create testing dataset."""

    def __init__(self, config_str: str):
        """Initialize data variables."""
        self.empty = {}  # no data
        self.ragged = {}
        self.regular = {}
        self._configure_dataset(None)

    def _configure_dataset(self, output_dir: Optional[str], **kwargs):
        """Define testing data."""
        self.ragged = {'ragged': [[1, 2, 3], [4, 5]]} # non-rectangular
        self.regular = OrderedDict([
            ('reg1', True), # bool, dim = 0
            ('reg2', [[0, 3.5], [-2, 1.2]]), # float
            ('reg3', 'string') # not bui or float
        ])

    def empty_stream(self) -> Stream:
        """No data in datastream."""
        yield self.empty

    def ragged_stream(self) -> Stream:
        """Data is non-rectangular, raises warning."""
        yield self.ragged

    def regular_stream(self) -> Stream:
        """Data is regular."""
        yield self.regular

    def undefined_stream(self) -> Stream:
        """Undefined stream raises exception."""

    def make_table(self, stream_name: str) -> Tuple:
        """Tabulate necessary parts of log to get correct formatting."""
        if stream_name is 'empty':
            tab_empty = tabulate.tabulate([[]],
                headers=['name', 'dtype', 'shape', 'range'], tablefmt='grid').split('\n')
            return tab_empty
        elif stream_name is 'ragged':
            val = np.array(self.ragged['ragged'])
            tab_ragged = tabulate.tabulate([['ragged', val.dtype, val.shape]],
                headers=['name', 'dtype', 'shape', 'range'], tablefmt='grid').split('\n')
            return tab_ragged
        elif stream_name is 'regular':
            val1 = np.array(self.regular['reg1'])
            val2 = np.array(self.regular['reg2'])
            val3 = np.array(self.regular['reg3'])
            tab_regular = tabulate.tabulate([
                ['reg1', val1.dtype, val1.shape, '{} - {}'.format(val1.min(), val1.max())],
                ['reg2', val2.dtype, val2.shape, '{0:.2f} - {1:.2f}'.format(val2.min(), val2.max())],
                ['reg3', val3.dtype, val3.shape]],
                headers=['name', 'dtype', 'shape', 'range'], tablefmt='grid').split('\n')
            return tab_regular


def test_check_dataset(caplog):
    """Test logging of source names, dtypes and shapes of all the streams available in given dataset."""
    empty_table_logging = tuple(map(lambda line: ('root', logging.INFO, line), MockDataset(None).make_table('empty')))
    ragged_table_logging = tuple(map(lambda line: ('root', logging.INFO, line), MockDataset(None).make_table('ragged')))
    regular_table_logging = tuple(map(lambda line: ('root', logging.INFO, line), MockDataset(None).make_table('regular')))

    complete_logging = (
        (('root', logging.INFO, "Found 4 stream candidates: ['empty_stream', "
            "'ragged_stream', 'regular_stream', 'undefined_stream']"),)
        + (('root', logging.INFO, 'empty_stream'),)
        + empty_table_logging
        + (('root', logging.INFO, 'ragged_stream'),)
        + (('root', logging.WARNING,
            '*** stream source `ragged` appears to be ragged (non-rectangular) ***'),)
        + ragged_table_logging
        + (('root', logging.INFO, 'regular_stream'),)
        + regular_table_logging
        + (('root', logging.INFO, 'undefined_stream'),)
        + (('root', logging.WARNING, 'Exception was raised during checking stream '
            '`undefined_stream`, (stack trace is displayed only with --verbose flag)'),)
        + (('root', logging.DEBUG, 'Traceback (most recent call last):\n'
           f'  File "{cxflow.datasets.base_dataset.__file__}", line 61, in '
            'stream_info\n'
            '    batch = next(iter(stream_fn()))\n'
            "TypeError: 'NoneType' object is not iterable\n"),)
    )

    caplog.set_level(logging.DEBUG)
    MockDataset(None).stream_info()
    assert caplog.record_tuples == list(complete_logging)
