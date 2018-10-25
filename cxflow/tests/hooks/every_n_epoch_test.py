"""
Test module for EveryNEpoch hook (cxflow.hooks.every_n_epoch).
"""

from cxflow.hooks.every_n_epoch import EveryNEpoch


class DummyHook(EveryNEpoch):
    """Dummy hook which inherits from :py:class:`cxflow.hooks.EveryNEpoch` hook."""

    def __init__(self, **kwargs):
        """Create new DummyHook."""
        super().__init__(**kwargs)
        self._epoch_ids = []

    def _after_n_epoch(self, epoch_id):
        """After every n epoch append ``epoch_id`` to the ``_epoch_ids`` list."""
        self._epoch_ids.append(epoch_id)


def test_after_n_epoch():
    """Test whether ``_after_n_epoch`` method is called every third epoch."""

    hook = DummyHook(n_epochs=3)

    for i in range(9):
        hook.after_epoch(i)

    assert hook._epoch_ids == [0, 3, 6]
