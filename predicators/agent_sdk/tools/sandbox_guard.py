"""Sandbox-escape screening for agent-supplied text."""
import os
import re
import sys
import traceback
from typing import List, Optional, Sequence, Set

# Filesystem roots that, when they prefix an absolute path outside the
# sandbox, mark it as a real escape (vs. data like "/done" printed by code).
SANDBOX_SYSTEM_ROOTS = (
    "/Users",
    "/home",
    "/root",
    "/etc",
    "/usr",
    "/opt",
    "/var",
    "/private",
    "/tmp",
    "/bin",
    "/sbin",
    "/lib",
    "/sys",
    "/proc",
    "/dev",
    "/mnt",
    "/srv",
)
# Predicators-source introspection that reaches outside the sandbox. The
# bare ``getsource`` substring also covers ``getsourcefile`` /
# ``getsourcelines``; ``inspect.getfile`` is matched explicitly so the
# generic ``getfilesystemencoding`` etc. don't false-positive.
SANDBOX_INTROSPECTION = ("getsource", "inspect.getfile", "site-packages")
# Hidden-ground-truth handles reachable from the in-process exec
# namespace: ``State.privileged`` (the env-only channel carrying hidden
# dynamics bookkeeping), the probe's ``_ctx`` (whose ``env`` is the LIVE
# real env, attributes and all), and the env's hidden-dynamics kill
# switch. Observed exploits: run_20260819_183137 seed1 read the cure
# counters off ``s.privileged``; run_20260819_183252 seed2 did
# ``sim._ctx.env`` to read ``cure_threshold`` and flipped
# ``_skip_domain_specific_dynamics`` to run true dynamics in its belief
# sim. A ``sim`` that is a facade keeps the full probe in ``_probe``,
# with every call the facade leaves out, so that name is screened too.
# Word-boundary regexes so e.g. ``budget_ctx`` never matches.
_SANDBOX_HIDDEN_STATE_RE = re.compile(r"\bprivileged\b|\b_ctx\b|\b_probe\b|"
                                      r"\b_skip_domain_specific_dynamics\b")
# Hidden-implementation predicators imports inside agent-executed
# Python: the exec runs in-process, so ``from predicators.envs... import
# ...`` would hand the agent the real env classes and ground-truth
# constants the sandbox deliberately hides (observed in
# run_20260717_182040 seed1 turn 22). Public authoring surfaces
# (``predicators.structs``, ``predicators.utils``) stay importable -
# agent-written simulator.py code depends on them.
# The module alternation is shared with the sandbox hook script (see
# sandbox_setup.VALIDATE_SANDBOX_SCRIPT), which anchors it differently
# for shell-command strings; sharing the alternation keeps the two
# guards covering the same modules by construction.
# The injected model base is the public surface. Its loader module and the
# supplied Oracle artifacts carry privileged constructors/mechanisms too.
# Include the package-level ``from ... import module`` spelling in text
# screens; the runtime import hook sees the equivalent dotted module name.
SANDBOX_HIDDEN_MODULES_PATTERN = (
    r"predicators\.(?:envs|ground_truth_models|code_sim_learning"
    r"(?:\.|\s+import\s+)(?:base_simulator|(?:continual|boil|bridge)_oracle))\b"
)
# The predicators modules agent code may import, the same in every arm:
# the public authoring surface (SANDBOX_PUBLIC_MODULES) and the modules
# that loading it loads. Everything else an arm is given reaches it
# through the names the harness injects, so no other module of the
# package is importable. The names are exact: a listed package does not
# admit its other modules. tests/agent_sdk/test_sandbox_setup.py
# recomputes the set, so a new import in the public modules fails there
# until it is listed here.
SANDBOX_PUBLIC_MODULES = ("predicators.structs", "predicators.utils")
SANDBOX_ALLOWED_MODULES = (
    "predicators",
    "predicators.args",
    "predicators.image_patch_wrapper",
    "predicators.pretrained_model_interface",
    "predicators.pybullet_helpers",
    "predicators.pybullet_helpers.geometry",
    "predicators.pybullet_helpers.joint",
    "predicators.settings",
    "predicators.structs",
    "predicators.utils",
)
# Packages allowed with every module under them (vendored code).
SANDBOX_ALLOWED_PACKAGE_TREES = ("predicators.third_party", )
# Modules allowed beside that set. None holds anything an arm lacks:
# - the policy worker, the harness's own entry point inside a guarded
#   python (primitive_policy.open_primitive_policy), and its package;
# - the ``code_sim_learning`` package itself, which holds a docstring.
#   With it importable, a refusal names the module the code asked for,
#   and the hidden modules under it keep their own message.
SANDBOX_EXTRA_MODULES = (
    "predicators.agent_sdk",
    "predicators.agent_sdk.primitive_policy_worker",
    "predicators.code_sim_learning",
)
# The packages among the names above. A name read from a package may be
# one of its other modules, so the text screen refuses it; a name read
# from a module is an attribute of that module.
SANDBOX_ALLOWED_PACKAGES = (
    "predicators",
    "predicators.agent_sdk",
    "predicators.code_sim_learning",
    "predicators.pybullet_helpers",
)


def allowed_modules_pattern(extra: Sequence[str] = ()) -> str:
    """The pattern of the module names agent code may import
    (``re.fullmatch``), with the modules ``extra`` admitted beside the shared
    list."""
    exact = SANDBOX_ALLOWED_MODULES + SANDBOX_EXTRA_MODULES + tuple(extra)
    return "(?:" + "|".join(
        [re.escape(m) for m in exact] +
        [re.escape(t) + r"(?:\.\w+)*"
         for t in SANDBOX_ALLOWED_PACKAGE_TREES]) + ")"


# One pattern for both guards: the sandbox python guard
# (sandbox_setup.write_pyguard) and the text screen below.
SANDBOX_ALLOWED_MODULES_PATTERN = allowed_modules_pattern()
# What a model file (``simulator.py``) may import beside the modules
# above. The harness loads that file in its own process, and the agent
# is shown these two imports: the model contract's example imports
# ``ParamSpec`` from ``fit_space`` (prompts/subclass_model.md), and the
# engine wrapper in ``reference/base_sim/`` names ``commands`` as the
# vocabulary of ``queue_residual_commands``.
SANDBOX_MODEL_FILE_MODULES = (
    "predicators.code_sim_learning.commands",
    "predicators.code_sim_learning.fit_space",
)
# What the agent reads about a module off the list. It names the module
# and nothing else: not what the module is for, and not which arm it
# belongs to.
SANDBOX_UNAVAILABLE_SUFFIX = (
    " is not available in the sandbox; use ./reference/ and the tools")
_HIDDEN_IMPORT_REASON = ("importing predicators env/ground-truth modules "
                         "would expose implementation the sandbox hides; "
                         "use the provided namespace and the MCP tools / "
                         "./reference/ files")
# Shell / interpreter forms that would start a python without the
# sandbox's sitecustomize guard (PYTHONPATH edits, ``env -i``, and the
# -S / -I / -E interpreter flags). Screened by the Bash hook and by the
# guard's own subprocess check, never by ``run_python`` (in-process).
SANDBOX_PYTHON_BYPASS_PATTERN = (
    r"(?:^|[\s;&|(`])(?:PYTHONPATH=|unset\s+PYTHONPATH|env\s+-i\b"
    r"|(?:\S*/)?python[0-9.]*\s+(?:-[a-zA-Z]+\s+(?:\S+\s+)?)*-[a-zA-Z]*[SIE])")
_SANDBOX_PREDICATORS_IMPORT_RE = re.compile(
    r"^\s*(?:from|import)\s+" + SANDBOX_HIDDEN_MODULES_PATTERN, re.MULTILINE)
# A dotted predicators name wherever agent code spells one: an import, a
# string handed to ``importlib.import_module`` / ``__import__``, a
# ``sys.modules[...]`` key, or an attribute walk from the package. A
# name inside a file path (``./predicators.txt``) is not one.
_PREDICATORS_NAME_RE = re.compile(r"(?<![\w./])predicators(?:\.[A-Za-z_]\w*)+")
# ``from <package> import a, b``: the names may be modules of the package.
_PREDICATORS_FROM_RE = re.compile(
    r"(?<![\w.])from\s+(predicators(?:\.\w+)*)\s+import\s+"
    r"(\([^)]*\)|(?:[^\n;#\\]|\\\n)*)")
# Host filesystem prefixes scrubbed from tracebacks surfaced to agents:
# absolute source paths both leak the layout and invite out-of-sandbox
# probing (an audited run responded to one with ``find /``).
_HOST_REPO_ROOT = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", ".."))


def _scrub_host_paths(text: str) -> str:
    """Strip host-absolute path prefixes from traceback text."""
    text = text.replace(_HOST_REPO_ROOT + os.sep, "")
    home = os.path.expanduser("~")
    if home and home != "/":
        text = text.replace(home + os.sep, "~" + os.sep)
    return text


def agent_traceback() -> str:
    """The traceback of the exception being handled, as an agent is shown it:
    host path prefixes scrubbed.

    Under ``agent_sim_learn_fit_invisible`` the harness's own frames are
    left out, since their file and function names and source lines name
    what that arm's agent must not read. The agent's frames, the
    libraries' and the error itself stay.

    ``continual_point_values_only`` leaves the harness's frames out too.
    """
    # pylint: disable-next=import-outside-toplevel
    from predicators.settings import CFG
    if not (CFG.agent_sim_learn_fit_invisible
            or CFG.continual_point_values_only):
        return _scrub_host_paths(traceback.format_exc())
    # _HOST_REPO_ROOT is the harness's package directory.
    package = _HOST_REPO_ROOT + os.sep
    error = traceback.TracebackException(*sys.exc_info())
    pending: List[Optional[traceback.TracebackException]] = [error]
    seen: Set[int] = set()
    while pending:
        link = pending.pop()
        if link is None or id(link) in seen:
            continue
        seen.add(id(link))
        link.stack = traceback.StackSummary.from_list([
            frame for frame in link.stack
            if not os.path.realpath(frame.filename).startswith(package)
        ])
        pending.extend([link.__cause__, link.__context__])
    return _scrub_host_paths("".join(error.format()))


def _reference_allowed(name: str, extra: Sequence[str]) -> bool:
    """Whether the dotted name ``name`` stays inside the allowed modules.

    It does when it is an allowed module, or a name read from an allowed
    module that is no package. A name read from an allowed package may
    be one of the package's other modules, so it is refused.
    """
    known = SANDBOX_ALLOWED_MODULES + SANDBOX_EXTRA_MODULES + tuple(extra)
    parts = name.split(".")
    for end in range(len(parts), 0, -1):
        prefix = ".".join(parts[:end])
        if any(prefix == tree or prefix.startswith(tree + ".")
               for tree in SANDBOX_ALLOWED_PACKAGE_TREES):
            return True
        if prefix in known:
            return end == len(parts) or \
                prefix not in SANDBOX_ALLOWED_PACKAGES
    return False


def agent_code_import_refusal(
    text: str, extra: Sequence[str] = ()) -> Optional[str]:
    """Why agent-written Python may not run, for the predicators modules it
    names; ``None`` when it names only allowed ones.

    The allowed modules are ``SANDBOX_ALLOWED_MODULES_PATTERN`` and
    ``extra`` (``SANDBOX_MODEL_FILE_MODULES`` for a model file). The
    screen reads text: it covers the import statements, the
    ``importlib.import_module`` / ``__import__`` / ``sys.modules``
    spellings that name a module in a string, and attribute walks from
    the ``predicators`` package. A name the code computes, or reaches
    from an object it already holds, passes.
    """
    if _SANDBOX_PREDICATORS_IMPORT_RE.search(text):
        return _HIDDEN_IMPORT_REASON
    names = [match.group(0) for match in _PREDICATORS_NAME_RE.finditer(text)]
    for match in _PREDICATORS_FROM_RE.finditer(text):
        listed = match.group(2).replace("\\", " ").strip("() \n")
        for item in listed.split(","):
            words = item.split()
            if words and words[0].isidentifier():
                names.append(f"{match.group(1)}.{words[0]}")
    for name in names:
        if _reference_allowed(name, extra):
            continue
        if re.match(SANDBOX_HIDDEN_MODULES_PATTERN, name):
            return _HIDDEN_IMPORT_REASON
        return name + SANDBOX_UNAVAILABLE_SUFFIX
    return None


def model_file_import_refusal(text: str) -> Optional[str]:
    """Why the harness may not load an agent-written ``simulator.py``, for the
    predicators modules it names; ``None`` when it may.

    The file runs in the harness process, where every module is loaded
    and no import hook stands, so this screen of its text is the
    barrier.
    """
    return agent_code_import_refusal(text, SANDBOX_MODEL_FILE_MODULES)


# Path-like tokens: absolute (``/foo``) or parent-traversal (``..``/``../foo``),
# anchored at a boundary (start, whitespace, quote, ``(`` or ``=``) so we skip
# ``/`` inside URLs (preceded by ``:``), division, and ``./relative`` paths
# (which stay inside the sandbox).
_SANDBOX_PATH_RE = re.compile(
    r"""(?:^|(?<=[\s'"`(=]))((?:/|\.\.)[^\s'"`)<>|;:,]*)""")


def _screen_text_for_sandbox_escape(
        text: str, sandbox_dir: Optional[str]) -> Optional[str]:
    """Best-effort screen of a Bash command / ``run_python`` code string.

    Returns a short deny reason if ``text`` looks like it reads outside
    ``sandbox_dir`` - an absolute or ``..`` path resolving out of the
    sandbox, or predicators-source introspection - else ``None``.

    ``sandbox_dir`` may be None (an in-process session with no sandbox
    copy on disk): the path screen has no boundary to enforce then and
    is skipped, but the source-introspection, hidden-state, and module-
    import screens (``agent_code_import_refusal``) still apply - the exec
    runs in-process either way, and those leaks do not depend on a
    sandbox directory existing.

    This is a heuristic: a determined script can still escape (env vars,
    ``subprocess``, computed paths), so OS-level isolation (the docker
    sandbox) remains the only hard boundary. The equivalent self-contained
    logic for Bash lives in ``sandbox_setup.VALIDATE_SANDBOX_SCRIPT``;
    keep the two in sync.
    """
    for needle in SANDBOX_INTROSPECTION:
        if needle in text:
            return (f"'{needle}' may read predicators source outside the "
                    "sandbox; use the MCP tools and ./reference/ files")
    hidden = _SANDBOX_HIDDEN_STATE_RE.search(text)
    if hidden is not None:
        # The text names no element an arm may lack.
        return (f"'{hidden.group(0)}' accesses harness-internal state "
                "that carries hidden environment ground truth; use the "
                "observable features and the documented tools only")
    unavailable = agent_code_import_refusal(text)
    if unavailable is not None:
        return unavailable
    if sandbox_dir is None:
        return None
    sandbox = os.path.realpath(sandbox_dir)
    for match in _SANDBOX_PATH_RE.finditer(text):
        token = match.group(1)
        resolved = os.path.realpath(
            token if os.path.isabs(token) else os.path.join(sandbox, token))
        if resolved == sandbox or resolved.startswith(sandbox + os.sep):
            continue
        if token.startswith("/") and not any(
                token == root or token.startswith(root + "/")
                for root in SANDBOX_SYSTEM_ROOTS):
            # Absolute but not a real filesystem path (e.g. printed data
            # like "/done") - don't flag.
            continue
        return f"path '{token}' resolves outside the sandbox directory"
    return None
