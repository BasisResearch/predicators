"""Root pytest configuration."""

# third_party/ holds git submodules. Their tests belong to their own repos and
# their code is not written against this repo's pylint config, so `pytest .`
# (the lint gate) must not descend into them.
collect_ignore_glob = ["logs/*", "third_party/*"]
