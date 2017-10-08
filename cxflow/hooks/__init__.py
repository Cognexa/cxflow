"""
Module with official **cxflow** hooks.


.. tip::

    Hooks listed here may be configured without specifying their fully qualified names. E.g.:

    .. code-block:: yaml

        hooks:
          - SaveBest

"""
from .abstract_hook import AbstractHook, TrainingTerminated
from .every_n_epoch import EveryNEpoch
from .accumulate_variables import AccumulateVariables
from .write_csv import WriteCSV
from .stop_after import StopAfter
from .log_variables import LogVariables
from .log_profile import LogProfile
from .log_dir import LogDir
from .save import SaveEvery, SaveBest
from .compute_stats import ComputeStats
from .check import Check
from .show_progress import ShowProgress
from .on_plateau import OnPlateau
from .stop_on_plateau import StopOnPlateau

AbstractHook.__module__ = '.hooks'

__all__ = ['AbstractHook', 'TrainingTerminated', 'AccumulateVariables', 'WriteCSV', 'StopAfter', 'LogVariables',
           'LogProfile', 'SaveEvery', 'SaveBest', 'CatchSigint', 'ComputeStats', 'Check', 'ShowProgress', 'EveryNEpoch',
           'OnPlateau', 'StopOnPlateau']
