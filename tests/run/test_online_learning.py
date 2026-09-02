"""Tests for run.online_learning's pre-loop test gate."""
import os

from predicators import utils
from predicators.run import checkpoints
from predicators.run.online_learning import initial_test_due


def test_initial_test_due(tmp_path) -> None:
    """The pre-loop test runs on a fresh start, is skipped by its flag, and on
    an --auto_resume relaunch at cycle 0 runs only when no pre-loop result was
    saved."""
    base = {
        "results_dir": str(tmp_path),
        "env": "cover",
        "approach": "random_actions",
        "seed": 0
    }
    utils.reset_config(base)
    assert initial_test_due()
    utils.reset_config({**base, "skip_initial_test": True})
    assert not initial_test_due()
    utils.reset_config({
        **base, "skip_test_until_last_ite_or_early_stopping":
        True
    })
    assert initial_test_due()  # only the per-cycle tests are deferred
    # A plain --load_approach --skip_until_cycle 0 still skips it.
    utils.reset_config({**base, "skip_until_cycle": 0})
    assert not initial_test_due()
    # A resume that found only the post-offline checkpoint reruns it ...
    utils.reset_config({**base, "skip_until_cycle": 0, "auto_resume": True})
    assert initial_test_due()
    # ... unless the pre-loop result is already on disk.
    with open(checkpoints.test_results_path(None), "wb") as f:
        f.write(b"")
    assert os.path.isfile(checkpoints.test_results_path(None))
    assert not initial_test_due()
    # ... and a result older than the checkpoint is an earlier run's.
    written = os.path.getmtime(checkpoints.test_results_path(None))
    assert initial_test_due(checkpoint_mtime=written + 10.0)
    assert not initial_test_due(checkpoint_mtime=written - 10.0)
    # Past cycle 0 the loop owns testing.
    utils.reset_config({**base, "skip_until_cycle": 1, "auto_resume": True})
    assert not initial_test_due()
