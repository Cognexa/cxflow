"""
Test module for csv hook (:py:class:`cxflow.hooks.WriteCSV).
"""

import numpy as np
import collections
import os
import tempfile

from cxflow.tests.test_core import CXTestCase
from cxflow.hooks.write_csv import WriteCSV


_EXAMPLES = 5
_VARIABLES = ['accuracy', 'precision', 'loss']


def _get_epoch_data():
    epoch_data = collections.OrderedDict([
        ('train', collections.OrderedDict([
            ('accuracy', 1),
            ('precision', np.ones(_EXAMPLES)),
            ('loss', collections.OrderedDict([('mean', 1)])),
            ('omitted', 0)])),
        ('test', collections.OrderedDict([
            ('accuracy', 2),
            ('precision', 2 * np.ones(_EXAMPLES)),
            ('loss', collections.OrderedDict([('mean', 2)])),
            ('omitted', 0)])),
        ('valid', collections.OrderedDict([
            ('accuracy', 3),
            ('precision', 3 * np.ones(_EXAMPLES)),
            ('loss', collections.OrderedDict([('mean', 3)])),
            ('omitted', 0)]))
    ])
    return epoch_data


class WriteCSVTest(CXTestCase):
    """Test case for :py:class:`cxflow.hooks.WriteCSV hook."""

    @staticmethod
    def get_tmp_filename():
        file = tempfile.NamedTemporaryFile()
        file.close()
        return file.name

    def test_init_hook(self):
        """Test correct hook initialization."""
        output_file = self.get_tmp_filename()

        hook = WriteCSV(output_dir="", output_file=output_file,
                        variables=_VARIABLES)

        self.assertEqual(hook._variables, _VARIABLES)

        with self.assertRaises(AssertionError):
            WriteCSV(output_dir="", output_file=output_file,
                     on_unknown_type='raise')

        with self.assertRaises(AssertionError):
            WriteCSV(output_dir="", output_file=output_file,
                     on_missing_variable='raise')

    def test_write_header(self):
        """Test writing a correct header to csv file."""
        output_file = self.get_tmp_filename()
        delimiter = ";"
        hook = WriteCSV(output_dir="", output_file=output_file,
                        variables=_VARIABLES, delimiter=delimiter)
        epoch_data = _get_epoch_data()
        hook._write_header(epoch_data)

        with open(output_file, 'r') as file:
            header = file.read()

        # header must ends with a newline symbol
        self.assertEqual(header[-1], "\n")
        header = header[:-1]

        tested_header_columns = header.split(delimiter)

        # epoch_id column must be first
        self.assertEqual(tested_header_columns[0], '"epoch_id"')
        tested_header_columns = tested_header_columns[1:]

        valid_header_columns = ['train_accuracy', 'train_precision',
                                'train_loss',
                                'test_accuracy', 'test_precision',
                                'test_loss',
                                'valid_accuracy', 'valid_precision',
                                'valid_loss']

        self.assertEqual(valid_header_columns,
                         tested_header_columns)

    def test_write_row(self):
        """Test writing one row to csv file."""
        output_file = self.get_tmp_filename()
        delimiter = ";"
        default_value = '?'
        hook = WriteCSV(output_dir="", output_file=output_file,
                        variables=_VARIABLES, delimiter=delimiter,
                        default_value=default_value)
        epoch_data = _get_epoch_data()
        hook._write_header(epoch_data)

        epoch_id = 6
        hook._write_row(epoch_id, epoch_data)

        with open(output_file, 'r') as file:
            content = file.readlines()
        row = content[1]

        # each line must ends with newline symbol
        self.assertEqual(row[-1], "\n")
        row = row[:-1]

        valid_row = ['6', '1', '?', '1', '2', '?', '2', '3', '?', '3']
        self.assertEqual(valid_row, row.split(delimiter))

    def test_raise_missing_variable(self):
        """
        Test raising error when selected variable is missing and
        on_missing_variable option is set to 'error'.
        """

        variables = _VARIABLES + ['missing']
        output_file = self.get_tmp_filename()
        hook = WriteCSV(output_dir="", output_file=output_file,
                        variables=variables, on_missing_variable='error')
        epoch_data = _get_epoch_data()
        hook._write_header(epoch_data)
        epoch_id = 6
        with self.assertRaises(KeyError):
            hook._write_row(epoch_id, epoch_data)

    def test_raise_unknown_type(self):
        """
        Test raising error when on_unknown_type option is set to 'error' and
        value of selected variable is not scalar.
        """
        output_file = self.get_tmp_filename()
        hook = WriteCSV(output_dir="", output_file=output_file,
                        on_unknown_type='error')
        epoch_data = _get_epoch_data()
        hook._write_header(epoch_data)
        epoch_id = 6
        with self.assertRaises(TypeError):
            hook._write_row(epoch_id, epoch_data)

    def test_after_epoch_one_header(self):
        """
        Tests result of after_epoch method and whether it writes
        only one header at the beginning of csv file.
        """
        output_file = self.get_tmp_filename()
        delimiter = "|"
        hook = WriteCSV(output_dir="", output_file=output_file,
                        delimiter=delimiter, variables=_VARIABLES)
        epoch_data = _get_epoch_data()

        hook.after_epoch(6, epoch_data)
        hook.after_epoch(7, epoch_data)

        with open(output_file) as file:
            content = file.readlines()

        self.assertEqual(len(content), 3)
        header = content[0]

        for i, epoch_id in enumerate(['6', '7']):
            row = content[i + 1]
            self.assertNotEqual(header, row)
            self.assertEqual(row[-1], "\n")
            row = row[:-1]
            valid_row = [epoch_id, '1', '', '1', '2', '', '2', '3', '', '3']
            self.assertEqual(valid_row, row.split(delimiter))
