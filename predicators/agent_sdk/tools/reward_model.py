"""A level's public reward model, as every arm's sandbox helper scores it.

One implementation scores a state sequence for all arms. The EMPIRIC
family passes its belief simulator, so a rule that replays physics runs
on that model. The baseline arms have no supplied simulator. They pass
none, a replay rule is then not checked, and the verdict's note says so.
Only the public triple (reward, solved, note) leaves this module.
"""
import json
import weakref
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, \
    Tuple

from predicators.agent_sdk.tools.scene import apply_state_modifications
from predicators.agent_sdk.tools.verdicts import evaluate_states_with, \
    task_evaluator
from predicators.structs import Action, State, Task, step_option_labels

# The note of a verdict whose replay rule had no simulator to run on.
REPLAY_UNCHECKED_NOTE = (
    "This level's reward model has a rule that replays physics, and this "
    "helper has no simulator to replay on, so that rule was not checked. "
    "The goal holds at the end of the sequence and no rule that needs no "
    "replay was broken. An unchecked rule earns no success credit here: "
    "solved is False and the reward has no success bonus. Only the real "
    "environment checks that rule.")

# Where the tool's scored sequence can start.
PREDICTION_STARTS = ("observation", "episode", "initial")

# The sequence each evaluator scored last on the engine-free path. A
# certificate may memoise its verdict by the identity of the scored
# objects (the Domino one does). A sequence built for one call dies with
# it, the next call's objects can take the same addresses, and the memo
# would then answer with the old verdict. Objects kept alive until the
# next call cannot share an address with that call's objects.
_LAST_SCORED: "weakref.WeakKeyDictionary[Any, Tuple[Any, Any]]" = \
    weakref.WeakKeyDictionary()


def level_task(train_tasks: Sequence[Task], current_task: Optional[Task],
               task_idx: Optional[int]) -> Task:
    """The task a reward-model query scores.

    ``task_idx`` indexes ``train_tasks``, the levels reached so far.
    None is the level in progress, or the first task when no level is
    bound.
    """
    if task_idx is None:
        if current_task is not None:
            return current_task
        task_idx = 0
    if not 0 <= task_idx < len(train_tasks):
        raise ValueError(f"task_idx {task_idx} out of range "
                         f"(0-{len(train_tasks) - 1}, the levels reached so "
                         "far; omit it for the current level).")
    return train_tasks[task_idx]


def trajectory_step_options(
        actions: Optional[Sequence[Any]]) -> Optional[Sequence[Any]]:
    """The per-transition labels an evaluator reads.

    ``Action`` objects are labeled by their producing options. Labels
    that are already built pass through. None means no labels.
    """
    if actions is None:
        return None
    acts = list(actions)
    if acts and isinstance(acts[0], Action):
        return step_option_labels(acts)
    return acts


def score_trajectory(task: Task,
                     states: Sequence[State],
                     actions: Optional[Sequence[Any]] = None,
                     *,
                     sim_env: Optional[Any] = None,
                     replay_engine: bool = True,
                     which: Any = "(current level)") -> Dict[str, Any]:
    """Score ``states`` with ``task``'s reward model: the public triple.

    ``sim_env`` is the caller's simulator for a rule that replays
    physics. ``replay_engine=False`` is the arm with no supplied
    simulator: the rule gets none, never the true environment, and
    ``note`` says when the verdict left it unchecked. ``which`` names
    the task in the error for one with no goal atoms.
    """
    evaluator = task_evaluator(task)
    if evaluator is None:
        raise ValueError(f"Task {which} has no goal atoms to judge.")
    if not states:
        raise ValueError("`states` must be a non-empty sequence.")
    state_list = list(states)
    step_options = trajectory_step_options(actions)
    if not replay_engine:
        sim_env = None
    verdict = evaluate_states_with(evaluator,
                                   state_list,
                                   step_options,
                                   sim_env=sim_env)
    note = verdict.get("note", "")
    if not replay_engine:
        unchecked = getattr(evaluator, "replay_unchecked", None)
        if unchecked is not None and unchecked(state_list, step_options):
            note = REPLAY_UNCHECKED_NOTE
        try:
            _LAST_SCORED[evaluator] = (state_list, step_options)
        except TypeError:
            # An evaluator that takes no weak reference is scored
            # without this protection.
            pass
    return {
        "reward": verdict["reward"],
        "solved": verdict["solved"],
        "note": note,
    }


def with_features(state: State, changes: Mapping[str, Mapping[str,
                                                              Any]]) -> State:
    """A copy of ``state`` with the named objects' named features changed.

    ``changes`` maps an object name (``"jug0"`` or ``"jug0:jug"``) to
    ``{feature: value}``. ``state`` itself is left as it is.
    """
    if not isinstance(changes, Mapping):
        raise ValueError("Feature changes must map an object name to "
                         "{feature: value}.")
    names = {obj.name for obj in state}
    mods: List[Dict[str, Any]] = []
    for name, features in changes.items():
        if not isinstance(features, Mapping):
            raise ValueError(f"The changes of '{name}' must be a "
                             "{feature: value} mapping.")
        if name not in names and isinstance(name, str):
            name = name.split(":", 1)[0]
        mods.append({"object": name, "features": features})
    changed, _, error = apply_state_modifications(state, mods)
    if error:
        raise ValueError(error)
    return changed


def make_engine_free_evaluate_trajectory_fn(
        task_of: Callable[[Optional[int]], Task]) -> Callable[..., Any]:
    """Build ``evaluate_trajectory`` for an arm with no supplied simulator.

    The EMPIRIC helper with no engine behind it and no physics sweep.
    ``task_of`` resolves a ``task_idx`` to its task
    (:func:`level_task`).
    """

    def evaluate_trajectory(states: Sequence[State],
                            actions: Optional[Sequence[Any]] = None,
                            task_idx: Optional[int] = None) -> Dict[str, Any]:
        """Score ``states`` with a level's reward model.

        ``task_idx`` indexes ``train_tasks``, the levels reached so
        far; omit it for the current level.
        ``states``: the sequence, ``states[t]`` before action ``t``.
        ``actions``: the recorded ``Action`` objects, or one label per
        transition, ``(option_name, (object_name, ...), (param, ...))``,
        ``None`` for a transition you do not attribute to a skill; omit
        for no labels. Returns ``{"reward", "solved", "note"}``. No
        simulator backs this helper. A rule that replays physics is not
        checked: the verdict then has ``solved`` False and no success
        bonus, and ``note`` says so ("" otherwise).
        """
        return score_trajectory(
            task_of(task_idx),
            states,
            actions,
            replay_engine=False,
            which="(current level)" if task_idx is None else task_idx)

    return evaluate_trajectory


def predicted_states(start: State, changes: Sequence[Any]) -> List[State]:
    """``start`` followed by one state per entry of ``changes``.

    Each entry maps object names to ``{feature: value}`` and changes the
    state before it; ``{}`` repeats that state.
    """
    states = [start]
    for index, entry in enumerate(changes):
        try:
            states.append(with_features(states[-1], entry))
        except ValueError as e:
            raise ValueError(f"states[{index}]: {e}") from e
    return states


def _label(index: int, label: Any) -> Any:
    """One JSON action label as the ``StepOption`` an evaluator reads."""
    if label is None:
        return None
    if (not isinstance(label, (list, tuple)) or not 1 <= len(label) <= 3
            or not isinstance(label[0], str)):
        raise ValueError(
            f"actions[{index}] must be null or [\"Skill\", [\"object\", "
            "...], [param, ...]].")
    objects = label[1] if len(label) > 1 else []
    params = label[2] if len(label) > 2 else []
    if not isinstance(objects,
                      (list, tuple)) or not isinstance(params, (list, tuple)):
        raise ValueError(f"actions[{index}]: the objects and the "
                         "parameters must be lists.")
    try:
        return (label[0], tuple(str(o).split(":", 1)[0] for o in objects),
                tuple(float(p) for p in params))
    except (TypeError, ValueError) as e:
        raise ValueError(f"actions[{index}]: parameters must be "
                         "numbers.") from e


def score_prediction(
    task: Task,
    start_states: Sequence[State],
    start_labels: Sequence[Any],
    changes: Any,
    actions: Any = None,
    which: Any = "(current level)",
) -> Dict[str, Any]:
    """Score a prediction given as JSON, for an arm with no Python workbench.

    The scored sequence is ``start_states`` (with ``start_labels`` on
    their transitions) followed by one state per entry of ``changes``
    (:func:`predicted_states`). ``actions`` holds one label per
    predicted state, or None for no labels. No simulator is passed, as
    in :func:`make_engine_free_evaluate_trajectory_fn`.
    """
    if not isinstance(changes, (list, tuple)):
        raise ValueError("`states` must be a list with one {object: "
                         "{feature: value}} entry per predicted state.")
    if not start_states:
        raise ValueError("No start state to predict from.")
    states = list(start_states[:-1]) + predicted_states(
        start_states[-1], changes)
    labels: Optional[List[Any]] = None
    if actions is not None:
        if not isinstance(actions,
                          (list, tuple)) or len(actions) != len(changes):
            raise ValueError("`actions` must hold one label per predicted "
                             f"state ({len(changes)}), null for a "
                             "transition with no skill.")
        labels = list(start_labels) + [
            _label(i, label) for i, label in enumerate(actions)
        ]
    elif any(label is not None for label in start_labels):
        # The recorded part keeps its labels when the prediction has none.
        labels = list(start_labels) + [None] * len(changes)
    verdict = score_trajectory(task,
                               states,
                               labels,
                               replay_engine=False,
                               which=which)
    # Plain JSON values: an evaluator may return numpy scalars.
    return {
        "reward": float(verdict["reward"]),
        "solved": bool(verdict["solved"]),
        "note": str(verdict["note"]),
    }


def load_prediction_file(path: str) -> Tuple[Any, Any]:
    """Read ``(states, actions)`` from a JSON file the agent wrote.

    The file holds ``{"states": [...], "actions": [...]}``, with
    ``actions`` optional. JSON only: the harness never unpickles a file
    from the sandbox.
    """
    try:
        with open(path, encoding="utf-8") as f:
            content = json.load(f)
    except (OSError, ValueError) as e:
        raise ValueError(f"`path` is not a readable JSON file: {e}") from e
    if not isinstance(content, dict) or "states" not in content:
        raise ValueError("The file must hold {\"states\": [...], "
                         "\"actions\": [...]} with `actions` optional.")
    return content["states"], content.get("actions")
