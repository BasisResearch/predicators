"""Agent-harness robot control experiment (see PLAN.md).

Importing this package pins OpenBLAS to one thread if the caller has not
chosen otherwise: the OpenBLAS bundled with numpy 1.23.5 produces wrong
results for tall-skinny matrix products when multithreaded on the AVX-512
Xeons we run on (see DEBUG_LOG.md, entry 1). Entry points should set the
variable before importing numpy; this is a fallback for interactive use.

It also imports ``predicators.utils`` first: predicators has an import-order
sensitive cycle (``structs`` <-> ``image_patch_wrapper``) that is only safe
when the package is entered through ``predicators.utils`` or an env module.
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import predicators.utils  # noqa: E402,F401  (import-order guard, see above)
