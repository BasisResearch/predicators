"""Root pytest configuration."""

# submodules/ holds git submodules. Their tests belong to their own repos and
# their code is not written against this repo's pylint config, so `pytest .`
# (the lint gate) must not descend into them.
collect_ignore_glob = ["logs/*", "submodules/*"]


def pytest_configure(config):
    """Register the markers the tests use, so pytest does not warn about
    them."""
    config.addinivalue_line(
        "markers", "slow: runs for minutes; deselect with -m 'not slow'")
