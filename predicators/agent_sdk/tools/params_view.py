"""Read-only view of fitted simulator parameters for synthesis tools."""
from typing import Any, Dict

# ── Predicate-invention tools ─────────────────────────────────────


class _ParamsView:
    """Read-through view onto a fitted-parameters dict.

    Holds the dict directly (not the approach) so predicate classifiers
    that close over this view do not transitively reference the
    approach. The approach must mutate the same dict object in place on
    each re-fit (clear + update) so the view picks up new values
    automatically; replacing the dict would break the live link.
    """

    def __init__(self, params: Dict[str, float]) -> None:
        self._params = params

    def __getitem__(self, key: str) -> float:
        if key not in self._params:
            # pylint: disable-next=import-outside-toplevel
            from predicators.settings import CFG
            known = ", ".join(sorted(self._params)) or "<none loaded yet>"
            refresh = ("call" if CFG.agent_sim_learn_fit_invisible else
                       "call (run sim.fit() to fit them)")
            raise KeyError(
                f"params[{key!r}] is not among the current simulator "
                f"parameters (available: {known}). Declare "
                f"ParamSpec({key!r}, ...) in simulator.py, or update "
                "the classifier to use a declared parameter; after a "
                "structural edit the values refresh at the next sim "
                f"{refresh}.")
        return self._params[key]

    def __contains__(self, key: object) -> bool:
        return key in self._params

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-style fallback lookup; mirrors ``dict.get``."""
        return self._params.get(key, default)

    def __repr__(self) -> str:
        return f"_ParamsView({self._params!r})"
