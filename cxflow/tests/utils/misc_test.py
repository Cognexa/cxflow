"""
Test module for :py:class:`cxflow.utils.misc.CaughtInterrupts`.
"""
import os
import signal
import threading
import platform
import pytest

from cxflow.hooks import TrainingTerminated
from cxflow.utils.misc import CaughtInterrupts


kill = os.kill

if 'Windows' in platform.system():
    def kill_windows(pid, signum):
        """
        Adapt os.kill for Windows platform.

        Reference: <https://stackoverflow.com/questions/35772001/how-to-handle-the-signal-in-python-on-windows-machine>
        """
        sigmap = {signal.SIGINT: signal.CTRL_C_EVENT, signal.SIGBREAK: signal.CTRL_BREAK_EVENT}
        if signum in sigmap and pid == os.getpid():
            # we don't know if the current process is a
            # process group leader, so just broadcast
            # to all processes attached to this console.
            pid = 0
        thread = threading.current_thread()
        handler = signal.getsignal(signum)
        # work around the synchronization problem when calling
        # kill from the main thread.
        if (signum in sigmap and
            thread.name == 'MainThread' and
            callable(handler) and
            pid == 0):
            event = threading.Event()

            def handler_set_event(signum, frame):
                event.set()
                return handler(signum, frame)
            signal.signal(signum, handler_set_event)
            try:
                os.kill(pid, sigmap[signum])
                # busy wait because we can't block in the main
                # thread, else the signal handler can't execute.
                while not event.is_set():
                    pass
            finally:
                signal.signal(signum, handler)
        else:
            os.kill(pid, sigmap.get(signum, signum))
    kill = kill_windows


class TestCaughtInterrupts:
    """Test case for ``CaughtInterrupts`` with-resource class."""

    def _interrupt_handler(self, *_):
        self._signal_unhandled = True

    def setup_method(self):
        """Register interrupt signal handlers."""
        for sig in CaughtInterrupts.INTERRUPT_SIGNALS:
            signal.signal(sig, self._interrupt_handler)
        self._signal_unhandled = False

    def test_catching_outside(self):
        """Test ``CaughtInterrupts`` does not handle interrupt signals outside with-resource environment."""
        for sig in CaughtInterrupts.INTERRUPT_SIGNALS:
            # without with-resource
            CaughtInterrupts()
            kill(os.getpid(), sig)
            assert self._signal_unhandled, \
                            'CaughtInterrupts handles signal `{}` outside with-resource environment'.format(sig)
            self._signal_unhandled = False

            # with with-resource
            with CaughtInterrupts():
                pass
            kill(os.getpid(), sig)
            assert self._signal_unhandled, \
                            'CaughtInterrupts handles signal `{}` outside with-resource environment'.format(sig)
            self._signal_unhandled = False

    def test_catching_inside(self):
        """Test ``CaughtInterrupts`` does handle interrupt signals inside with-resource environment."""
        for sig in CaughtInterrupts.INTERRUPT_SIGNALS:
            with CaughtInterrupts():
                kill(os.getpid(), sig)
            assert not self._signal_unhandled, \
                             'CaughtInterrupts does not handle signal `{}` inside with-resource environment'.format(sig)

    def test_raising(self):
        """Test ``CaughtInterrupts`` does rise ``TrainingTerminated`` exception."""
        for sig in CaughtInterrupts.INTERRUPT_SIGNALS:
            with CaughtInterrupts() as catch:
                kill(os.getpid(), sig)
                with pytest.raises(TrainingTerminated):
                    catch.raise_check_interrupt()

    def test_exit(self):
        """Test ``CaughtInterrupts`` calls sys.exit after 2nd interrupt signal."""
        for sig in CaughtInterrupts.INTERRUPT_SIGNALS:
            with CaughtInterrupts():
                kill(os.getpid(), sig)
                with pytest.raises(SystemExit):
                    kill(os.getpid(), sig)
