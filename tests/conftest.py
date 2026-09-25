"""Shared configurations for pytest.

See https://docs.pytest.org/en/6.2.x/fixture.html.
"""

from typing import Iterator

import pytest

from predicators.envs import _MOST_RECENT_ENV_INSTANCE


def pytest_addoption(parser):
    """Enable a command line flag for running tests decorated with @longrun."""
    parser.addoption('--longrun',
                     action='store_true',
                     dest="longrun",
                     default=False,
                     help="enable tests decorated with @longrun")


def _restore_env_cache() -> Iterator[None]:
    """Yield, then put the env cache back as it was before.

    get_or_create_env() returns the cached env of a name, and
    get_gt_options() builds its skills from that env's types, so an env
    one test caches would otherwise reach every later test, even one
    built under another config (a partially observable Bridge env has a
    different block type) or one the test has since disconnected.
    """
    saved = dict(_MOST_RECENT_ENV_INSTANCE)
    yield
    _MOST_RECENT_ENV_INSTANCE.clear()
    _MOST_RECENT_ENV_INSTANCE.update(saved)


@pytest.fixture(autouse=True)
def _env_cache_per_test() -> Iterator[None]:
    """Undo a test's changes to the env cache when the test ends."""
    yield from _restore_env_cache()


@pytest.fixture(autouse=True, scope="module")
def _env_cache_per_module() -> Iterator[None]:
    """Undo a module's changes to the env cache when the module ends.

    pytest sets up module-scoped fixtures before function-scoped ones,
    so the snapshot of _env_cache_per_test already holds an env that a
    module-scoped fixture caches. Only this fixture drops that env.
    """
    yield from _restore_env_cache()
