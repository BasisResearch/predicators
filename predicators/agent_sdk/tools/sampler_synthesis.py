"""The ``sim.samplers()`` loader for sampler-synthesis sessions."""
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from predicators.agent_sdk.proposal_exec import build_exec_context, \
    load_learned_samplers
from predicators.agent_sdk.synthesis_backend import SamplerSynthesisBackend
from predicators.agent_sdk.tools.params_view import _ParamsView
from predicators.agent_sdk.tools.sandbox_guard import _scrub_host_paths
from predicators.agent_sdk.tools.snapshots import _ArtifactSnapshotter
from predicators.settings import CFG
from predicators.structs import Object


def make_sampler_loader(
    samplers_file: str,
    samplers_versions_dir: str,
    approach: SamplerSynthesisBackend,
    cycle_index_provider: Optional[Callable[[], int]] = None,
) -> Callable[[], str]:
    """Build the ``sim.samplers()`` loader for one synthesis session.

    On each call the loader loads
    ``samplers.py`` fresh (snapshotting into ``samplers_versions_dir``),
    validates the ``LEARNED_SAMPLERS`` dict (option name -> callable),
    installs it into ``approach._synthesized_samplers`` so refinement
    uses it, and reports a per-option shape/in-box sanity check.

    Args:
        samplers_file: Host path to the agent-edited ``samplers.py``.
        samplers_versions_dir: Directory for per-call snapshots.
        approach: The ``AgentSimLearningApproach`` instance.
        cycle_index_provider: Returns the current 0-based cycle
            (negative = the offline pass, rendered as ``offline``).
    """
    # pylint: disable=import-outside-toplevel
    import traceback  # pylint: disable=redefined-outer-name,reimported

    from predicators.code_sim_learning.fit_space import ParamSpec

    # pylint: enable=import-outside-toplevel
    _snapshotter = _ArtifactSnapshotter(
        live_file=samplers_file,
        versions_dir=samplers_versions_dir,
        artifact_name="samplers",
        cycle_index_provider=cycle_index_provider,
        missing_file_hint=("Use Write to create it with "
                           "LEARNED_SAMPLERS = {\"OptionName\": fn, ...}."),
    )
    params_view = _ParamsView(approach._fitted_params)  # pylint: disable=protected-access

    def _snapshot_and_load_samplers(
        path: str,
    ) -> Tuple[Dict[str, Any], Optional[str], Optional[str], List[str]]:
        """Snapshot ``path`` then exec it into a fresh namespace.

        Returns ``(samplers, version_tag, error_msg, warnings)``.
        Entries keyed by an unknown option name, or whose value is not
        callable, are skipped and described in ``warnings``. On success,
        mutates ``approach._synthesized_samplers`` to the validated
        dict.
        """
        raw, version_tag, err = _snapshotter.snapshot(path)
        if err is not None:
            return {}, None, err, []
        assert raw is not None and version_tag is not None

        ctx = build_exec_context(
            types=approach._types,  # pylint: disable=protected-access
            predicates=approach._get_all_predicates(),  # pylint: disable=protected-access
            options=approach._get_all_options(),  # pylint: disable=protected-access
            extra_context={
                "params": params_view,
                "ParamSpec": ParamSpec,
            })
        option_names = {o.name for o in approach._get_all_options()}  # pylint: disable=protected-access
        valid, warnings, err = load_learned_samplers(raw.decode("utf-8"), ctx,
                                                     option_names)
        if err is not None:
            return {}, version_tag, (f"[{version_tag}] Error executing "
                                     f"{path}:\n{err}"), []

        # Mutate approach state so sim.refine / test-time
        # refinement draw from the agent's draft samplers.
        approach._synthesized_samplers = valid  # pylint: disable=protected-access
        return valid, version_tag, None, warnings

    def _sanity_check(name: str, fn: Any) -> str:
        """Draw a few params from a representative state; report shape/box."""
        # pylint: disable=protected-access
        options_by_name = {o.name: o for o in approach._get_all_options()}
        opt = options_by_name[name]
        train_tasks = approach._train_tasks
        if not train_tasks:
            return f"  {name}: no train task to sanity-check against."
        state = train_tasks[0].init
        # Pick the first object of each option-arg type present in the state.
        objs: List[Object] = []
        for t in opt.types:
            match = next((o for o in state if o.type.name == t.name), None)
            if match is None:
                return (f"  {name}: no object of type '{t.name}' in the "
                        "train-task state to sanity-check against.")
            objs.append(match)
        box = opt.params_space
        expected = box.shape[0]
        rng = np.random.default_rng(CFG.seed)
        in_box = 0
        n_draws = 3
        for _ in range(n_draws):
            try:
                raw = fn(state, set(), rng, objs)
                arr = np.asarray(raw, dtype=np.float32).reshape(-1)
            except Exception:  # pylint: disable=broad-except
                last = traceback.format_exc().strip().splitlines()[-1]
                return (f"  {name}: ERROR — sampler raised: {last} "
                        "(note: this check, and refinement at steps with "
                        "no subgoal annotation, call the sampler with "
                        "subgoal_atoms=set(); it must not crash on an "
                        "empty set — fall back to a default or uniform "
                        "draw).")
            if arr.shape != (expected, ):
                return (f"  {name}: ERROR — returned shape {arr.shape}, "
                        f"expected ({expected},).")
            if bool(np.all(arr >= box.low - 1e-6)) and \
                    bool(np.all(arr <= box.high + 1e-6)):
                in_box += 1
        return (f"  {name}: OK — {n_draws} draws, {in_box}/{n_draws} "
                f"within the params box.")

    def sampler_report() -> str:
        """Reload samplers.py, install LEARNED_SAMPLERS, and report the per-
        option sanity check."""
        try:
            samplers, version_tag, err, warnings = (
                _snapshot_and_load_samplers(samplers_file))
        except Exception:  # pylint: disable=broad-except
            return (f"Error loading samplers.py:\n"
                    f"{_scrub_host_paths(traceback.format_exc())}")

        if err is not None:
            return err

        prefix = f"[{version_tag}]"
        lines = [
            f"{prefix} Sampler report — {len(samplers)} per-skill "
            f"sampler(s) installed.",
        ]
        if warnings:
            lines.append("")
            lines.append("Warnings (entries skipped during load):")
            for w in warnings:
                lines.append(f"  - {w}")

        if not samplers:
            lines.append("")
            lines.append("LEARNED_SAMPLERS is empty — add "
                         "{\"OptionName\": fn} entries to samplers.py.")
            return "\n".join(lines)

        lines.append("")
        lines.append("Sanity check (representative train-task state):")
        for name in sorted(samplers):
            lines.append(_sanity_check(name, samplers[name]))
        lines.append("")
        lines.append("Now call sim.refine with a sketch that "
                     "uses these options to measure the samples-to-refine "
                     "improvement.")
        return "\n".join(lines)

    return sampler_report
