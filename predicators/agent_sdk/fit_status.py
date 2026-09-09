"""Consistent descriptions of the evidence behind a published model."""
from typing import Any, Mapping


def format_fit_status(state: Mapping[str, Any]) -> str:
    """Describe calibration without turning a rejected fit into evidence."""
    version = state.get("version")
    coverage = state.get("coverage")
    if state.get("pinned", False):
        status = (f"UNVALIDATED ({version}): fit rejected; using declared "
                  "rule values and unchanged planning physics")
    elif coverage is not None and coverage[0] < coverage[1]:
        status = f"PARTIAL FIT ({version})"
    else:
        status = f"fitted ({version})"
    if coverage is not None:
        status += (f"; {coverage[0]}/{coverage[1]} recorded motion segments "
                   "accepted by the fit (not held-out validation)")
    rejection = state.get("last_rejection")
    if rejection is not None:
        status = (f"FIT REJECTED ({rejection}); retaining earlier {status}. "
                  "The latest data has not validated these parameters")
    return status
