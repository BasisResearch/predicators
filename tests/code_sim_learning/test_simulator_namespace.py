"""The simulator-namespace readers accept the current export names and the
names they replaced, so files written under an old name still load."""
from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.code_sim_learning.utils import read_physical_param_specs


def test_physical_param_specs_read_under_both_names() -> None:
    """``PHYSICAL_PARAM_SPECS`` is the export; ``PHYSICAL_PARAMS`` (its name
    until 2026-09-05) is still read; a callable and an absent export follow the
    ``PARAM_SPECS`` conventions."""
    spec = ParamSpec("mu", 0.5, lo=0.0, hi=1.0)
    assert read_physical_param_specs({"PHYSICAL_PARAM_SPECS": [spec]}) == \
        [spec]
    assert read_physical_param_specs({"PHYSICAL_PARAMS": [spec]}) == [spec]
    assert read_physical_param_specs({"PHYSICAL_PARAM_SPECS":
                                      lambda: [spec]}) == [spec]
    assert read_physical_param_specs({"PHYSICAL_PARAM_SPECS": []}) is None
    assert read_physical_param_specs({}) is None
