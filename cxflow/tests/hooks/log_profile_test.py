"""
Module with profile hook test case (see :py:class:`cxflow.hooks.LogProfile`).
"""
import logging
from cxflow.constants import CXF_DEFAULT_TRAIN_STREAM
from cxflow.hooks import LogProfile


_TRAIN_ONLY_PROFILE = {'read_batch_train': [1.12, 2, 3],
                       'after_batch_hooks_train': [4, 0, 0.4],
                       'eval_batch_train': [6, 7.9, 8],
                       'after_epoch_hooks': [5.4]}

_TRAIN_AND_VALID_PROFILE = {'read_batch_train': [1.001, 2, 3],
                            'after_batch_hooks_train': [4, 0, 0],
                            'eval_batch_train': [6, 7, 8.54],
                            'read_batch_valid': [3, 5, 6],
                            'after_batch_hooks_valid': [1, 1.11, 1],
                            'eval_batch_valid': [0, 1, 2],
                            'after_epoch_hooks': [5.3]}

_TRAIN_TEST_AND_VALID_PROFILE = {'read_batch_train': [1.001, 2, 3],
                                 'after_batch_hooks_train': [4, 0, 0],
                                 'eval_batch_train': [6, 7, 8.54],
                                 'read_batch_valid': [3, 5, 6],
                                 'after_batch_hooks_valid': [1, 1.11, 1],
                                 'eval_batch_valid': [0, 1, 2],
                                 'read_batch_test': [1.1, 1.11, 1.111],
                                 'after_batch_hooks_test': [2.2, 2.22, 2.222],
                                 'eval_batch_test': [3.3, 3.33, 3.333],
                                 'after_epoch_hooks': [5.3]}


class TestLogProfile:
    """
    Test case for :py:class:`cxflow.hooks.LogProfile` self._hook.

    Hereby, we test only proper handling (logging) of the profile.
    The profiling itself is tested in the main_loop test (see cxflow.tests-unittest.main_loop_test.py).
    """

    def setup_method(self):
        self._hook = LogProfile()

    def test_missing_train(self, caplog):
        """Test KeyError raised on missing profile entries."""
        caplog.set_level(logging.INFO)

        self._hook.after_epoch_profile(0, {}, CXF_DEFAULT_TRAIN_STREAM, [])
        assert caplog.record_tuples == [
            ('root', logging.INFO, '\tT read data:\t0.000000'),
            ('root', logging.INFO, '\tT train:\t0.000000'),
            ('root', logging.INFO, '\tT eval:\t0.000000'),
            ('root', logging.INFO, '\tT hooks:\t0.000000')
        ]

        caplog.clear()
        self._hook.after_epoch_profile(0, {'some_contents': 1}, CXF_DEFAULT_TRAIN_STREAM, [])
        assert caplog.record_tuples == [
            ('root', logging.INFO, '\tT read data:\t0.000000'),
            ('root', logging.INFO, '\tT train:\t0.000000'),
            ('root', logging.INFO, '\tT eval:\t0.000000'),
            ('root', logging.INFO, '\tT hooks:\t0.000000')
        ]

    def test_train_only(self, caplog):
        """Test profile handling with only train stream."""
        caplog.set_level(logging.INFO)

        self._hook.after_epoch_profile(1, _TRAIN_ONLY_PROFILE, CXF_DEFAULT_TRAIN_STREAM, [])
        assert caplog.record_tuples == [
            ('root', logging.INFO, '\tT read data:\t6.120000'),
            ('root', logging.INFO, '\tT train:\t21.900000'),
            ('root', logging.INFO, '\tT eval:\t0.000000'),
            ('root', logging.INFO, '\tT hooks:\t9.800000'),
        ]

    def test_extra_streams(self, caplog):
        """Test extra streams handling."""
        caplog.set_level(logging.INFO)

        self._hook.after_epoch_profile(0, _TRAIN_ONLY_PROFILE, CXF_DEFAULT_TRAIN_STREAM, ['valid'])
        assert caplog.record_tuples == [
            ('root', logging.INFO, '\tT read data:\t6.120000'),
            ('root', logging.INFO, '\tT train:\t21.900000'),
            ('root', logging.INFO, '\tT eval:\t0.000000'),
            ('root', logging.INFO, '\tT hooks:\t9.800000'),
        ]

        # test additional entries being ignored if the extra stream was not specified
        caplog.clear()
        self._hook.after_epoch_profile(1, _TRAIN_AND_VALID_PROFILE, CXF_DEFAULT_TRAIN_STREAM, [])
        assert caplog.record_tuples == [
            ('root', logging.INFO, '\tT read data:\t6.001000'),
            ('root', logging.INFO, '\tT train:\t21.540000'),
            ('root', logging.INFO, '\tT eval:\t0.000000'),
            ('root', logging.INFO, '\tT hooks:\t9.300000'),
        ]

        # test one additional stream
        caplog.clear()
        self._hook.after_epoch_profile(1, _TRAIN_AND_VALID_PROFILE, CXF_DEFAULT_TRAIN_STREAM, ['valid'])
        assert caplog.record_tuples == [
            ('root', logging.INFO, '\tT read data:\t20.001000'),
            ('root', logging.INFO, '\tT train:\t21.540000'),
            ('root', logging.INFO, '\tT eval:\t3.000000'),
            ('root', logging.INFO, '\tT hooks:\t12.410000'),
        ]

        # test two additional streams
        caplog.clear()
        self._hook.after_epoch_profile(1, _TRAIN_TEST_AND_VALID_PROFILE, CXF_DEFAULT_TRAIN_STREAM, ['valid', 'test'])
        assert caplog.record_tuples == [
            ('root', logging.INFO, '\tT read data:\t23.322000'),
            ('root', logging.INFO, '\tT train:\t21.540000'),
            ('root', logging.INFO, '\tT eval:\t12.963000'),
            ('root', logging.INFO, '\tT hooks:\t19.052000'),
        ]
