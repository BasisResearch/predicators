"""Digest renderers over ToolContext state.

The single formatting source for the type / option / task / trajectory
summaries: the solve and synthesis prompt builders inject them directly,
and the probe serves the same text from ``sim.task()`` and
``describe_trajectory`` (so the wording cannot drift between surfaces).
"""
from typing import Any, Collection, Iterable, List, Optional, Union

from predicators import utils


def render_types_digest(types: Iterable[Any]) -> str:
    """One line per type: name, parent, feature names."""
    lines = []
    for t in sorted(types, key=lambda t: t.name):
        features = ", ".join(
            t.feature_names) if t.feature_names else "(no features)"
        parent_str = f" (parent: {t.parent.name})" if t.parent else ""
        lines.append(f"- {t.name}{parent_str}: [{features}]")
    if not lines:
        return "No types defined."
    return "\n".join(lines)


def render_options_digest(options: Iterable[Any],
                          gt_options_ref_path: Optional[str] = None) -> str:
    """One line per option: typed signature plus parameter box/descriptions.

    The ``obj:type`` signature and parameter listing here are what the
    strict plan parser expects plans to match, so this exact rendering
    is shared by every prompt surface.
    """
    lines = []
    for opt in sorted(options, key=lambda o: o.name):
        type_sig = ", ".join(t.name for t in opt.types)
        params_dim = opt.params_space.shape[0] if opt.params_space.shape else 0
        if params_dim > 0:
            low = opt.params_space.low.tolist()
            high = opt.params_space.high.tolist()
            if opt.params_description:
                desc = ", ".join(opt.params_description)
                param_info = f", params=[{desc}], low={low}, high={high}"
            else:
                param_info = (f", params_dim={params_dim}, "
                              f"low={low}, high={high}")
        else:
            param_info = ""
        lines.append(f"  {opt.name}({type_sig}{param_info})")
    if not lines:
        return "No options defined."
    if gt_options_ref_path:
        lines.append(f"\nOption definition source code: "
                     f"`{gt_options_ref_path}` (search for the option name).")
    return "\n".join(lines)


def render_task_digest(task: Any,
                       task_idx: Union[int, str],
                       predicates: Collection[Any],
                       include_goal_query_hint: bool = False) -> str:
    """Goal (NL preferred), initial atoms, objects, and init-state details for
    one task.

    ``task_idx`` is the header label; a string (e.g. ``"(current solve
    task)"``) is allowed for tasks outside the train list.
    ``include_goal_query_hint`` adds the ``is_goal_state`` /
    ``goal_holds`` pointer, which only makes sense (and is only
    numerically valid) in sessions whose exec namespace binds those
    names (synthesis ``run_python``), so pass it only with an integer
    ``task_idx``.
    """
    if task.goal_nl:
        goal_line = f"  Goal (natural language): {task.goal_nl}"
    else:
        goal_str = ", ".join(str(g) for g in sorted(task.goal))
        goal_line = f"  Goal: {{{goal_str}}}"
    init_atoms = utils.abstract(task.init, predicates)
    atoms_str = ", ".join(str(a) for a in sorted(init_atoms))
    objects = sorted(task.init, key=str)
    obj_str = ", ".join(f"{o.name}:{o.type.name}" for o in objects)
    state_str = task.init.pretty_str()
    hint_line = ""
    if include_goal_query_hint:
        hint_line = (f"  Goal achievement: query "
                     f"`is_goal_state(state, {task_idx})` or "
                     f"`train_tasks[{task_idx}].goal_holds(state)`.\n")
    return (f"Task {task_idx}:\n"
            f"{goal_line}\n"
            f"{hint_line}"
            f"  Initial atoms: {{{atoms_str}}}\n"
            f"  Objects: [{obj_str}]\n\n"
            f"Initial state details:\n{state_str}")


def render_trajectory_digest(trajectories: List[Any],
                             train_tasks: List[Any],
                             predicates: Collection[Any],
                             traj_idx: int,
                             include_states: bool = True,
                             include_atoms: bool = False,
                             max_timesteps: int = 10) -> str:
    """Header (provenance, goal, reached_goal) plus per-timestep state / atoms.

    / action for one trajectory.

    Raises ``ValueError`` on an out-of-range ``traj_idx``.
    """
    if not trajectories:
        raise ValueError("No trajectories available yet.")
    if traj_idx < 0 or traj_idx >= len(trajectories):
        raise ValueError(f"Invalid traj_idx {traj_idx}. "
                         f"Available: 0-{len(trajectories) - 1}")
    traj = trajectories[traj_idx]
    provenance = "demo" if traj.is_demo else "interaction"
    task_idx = traj._train_task_idx  # pylint: disable=protected-access
    header = (f"Trajectory {traj_idx}: {len(traj.states)} states, "
              f"{len(traj.actions)} actions  "
              f"[provenance={provenance}, task={task_idx}")
    if task_idx is not None and 0 <= task_idx < len(train_tasks):
        task = train_tasks[task_idx]
        reached = task.goal_holds(traj.states[-1])
        goal_str = ", ".join(str(g) for g in sorted(task.goal))
        header += f", reached_goal={reached}]"
        lines = [header, f"Goal: {{{goal_str}}}"]
    else:
        header += "]"
        lines = [header]

    for t_step, state in enumerate(traj.states[:max_timesteps]):
        lines.append(f"\n--- Timestep {t_step} ---")
        if include_states:
            lines.append("State:")
            lines.append(state.dict_str(indent=2, num_decimal_points=4))
        if include_atoms:
            atoms = utils.abstract(state, predicates)
            atoms_str = ", ".join(str(a) for a in sorted(atoms))
            lines.append(f"Atoms: {{{atoms_str}}}")
        if t_step < len(traj.actions):
            act = traj.actions[t_step]
            opt = act.get_option()
            lines.append(f"Action: {opt.name}({opt.objects})")

    if len(traj.states) > max_timesteps:
        lines.append(
            f"\n... ({len(traj.states) - max_timesteps} more timesteps)")
    return "\n".join(lines)
