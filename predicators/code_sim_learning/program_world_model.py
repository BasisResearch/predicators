"""Option-level program world models (paper arm C4).

A code world model with no engine underneath, in the form of Pinductor
/ POMDP Coder ported to parameterized skills: the agent writes
``world_model.py`` exporting

.. code-block:: python

    LATENT_FEATURES: Dict[str, List[str]]   # {type: [hidden feature names]}

    def initial_latent(obs: State, rng: np.random.Generator) -> Dict[str, Any]:
        ...  # a draw of the hidden state consistent with the first observation

    def transition(obs: State, latent: Dict[str, Any], option: _Option,
                   rng: np.random.Generator
                   ) -> Tuple[State, Dict[str, Any], int]:
        ...  # (next observed state, next latent, low-level steps consumed)

The observation is the environment's ``State`` (observable features
only); the latent is the program's own hidden block, carried on
``State.latent`` between options so backtracking search restores it per
node. :class:`ProgramOptionModel` wraps the program as an option model,
so the sketch / refine / validation harness runs against it unchanged.
:func:`score_program` is the Pinductor scorer: a particle-filtered
kernel pseudo-likelihood of the program on the recorded option-level
trajectories (paper Eq. 7-8 of Pinductor), teacher-forced on the
observations with the particles carrying the latent.
"""
from __future__ import annotations

import copy
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from predicators import utils
from predicators.agent_sdk.proposal_exec import build_exec_context, \
    exec_code_safely
from predicators.code_sim_learning.utils import observation_view
from predicators.nsrt_learning.segmentation import segment_trajectory
from predicators.option_model import _OptionModelBase
from predicators.structs import LowLevelTrajectory, ParameterizedOption, \
    Predicate, State, Type, _Option

PROGRAM_EXPORTS = ("initial_latent", "transition")
# Distance charged to a particle whose transition raised: far outside
# the range of a sane prediction (distances are in feature-std units),
# so a crashing program scores badly instead of silently.
EXCEPTION_DISTANCE = 10.0
# Feature-std floor for the observation distance: a feature that never
# moved in the data still needs a finite scale to be compared on.
_SCALE_FLOOR = 1e-3


@dataclass
class ProgramWorldModel:
    """The loaded program: its two callables and the latent declaration."""
    initial_latent: Callable[..., Dict[str, Any]]
    transition: Callable[..., Any]
    latent_features: Dict[str, List[str]] = field(default_factory=dict)


def load_program_world_model(
    code: str,
    types: Set[Type],
    predicates: Set[Predicate],
    options: Set[ParameterizedOption],
) -> Tuple[Optional[ProgramWorldModel], Optional[str]]:
    """Exec ``code`` and validate the world-model exports.

    Returns ``(program, None)`` or ``(None, error text)``. The exec
    namespace is the same one predicates.py gets (types bound as
    ``<name>_type``, ``np``), so the two artifacts share a vocabulary.
    """
    ctx = build_exec_context(types=types,
                             predicates=predicates,
                             options=options,
                             extra_context={"np": np})
    _, err = exec_code_safely(code, ctx, "transition")
    if err is not None:
        return None, err
    missing = [n for n in PROGRAM_EXPORTS if not callable(ctx.get(n))]
    if missing:
        return None, (f"world_model.py must define the callables "
                      f"{list(PROGRAM_EXPORTS)}; missing or not callable: "
                      f"{missing}.")
    feats = ctx.get("LATENT_FEATURES", {})
    if not isinstance(feats, dict):
        return None, ("LATENT_FEATURES must be a dict "
                      "{type_name: [hidden feature names]}.")
    latent_features = {
        str(k): [str(f) for f in (v or [])]
        for k, v in feats.items()
    }
    return ProgramWorldModel(ctx["initial_latent"], ctx["transition"],
                             latent_features), None


def _unpack_transition(out: Any) -> Tuple[State, Dict[str, Any], int]:
    if not isinstance(out, tuple) or len(out) != 3:
        raise ValueError("transition must return a 3-tuple (next_obs, "
                         "next_latent, num_low_level_steps).")
    nxt, latent, num_steps = out
    if not isinstance(nxt, State):
        raise ValueError("transition's first return value must be a State "
                         f"(got {type(nxt).__name__}).")
    if not isinstance(latent, dict):
        raise ValueError("transition's second return value must be the "
                         f"latent dict (got {type(latent).__name__}).")
    return nxt, latent, int(num_steps)


def _check_next_state(prev: State, nxt: State) -> None:
    if set(nxt.data) != set(prev.data):
        raise ValueError("the next state must keep exactly the objects of "
                         "the current one.")
    for obj, arr in nxt.data.items():
        vals = np.asarray(arr, dtype=float)
        if vals.shape != np.asarray(prev.data[obj], dtype=float).shape:
            raise ValueError(f"{obj.name}: the feature vector changed shape.")
        if not np.all(np.isfinite(vals)):
            raise ValueError(f"{obj.name}: non-finite feature values.")


def _observation(state: State) -> State:
    """A private copy of the observable part of ``state`` (no latent)."""
    obs = observation_view(state).copy()
    obs.latent = None
    return obs


class ProgramOptionModel(_OptionModelBase):
    """An option model that steps an agent-written option-level program.

    The latent rides on ``State.latent``; a state without one (a task's
    initial state, a probe reset) is seeded from the program's
    ``initial_latent`` - or from ``initial_latent_override`` while the
    capture gate sweeps the belief particles. Program exceptions surface
    as :class:`utils.OptionExecutionFailure` with the traceback tail in
    ``last_execution_failure``, the same channel the engine-backed model
    uses, so every consumer reports them the same way.
    """

    def __init__(self, program: ProgramWorldModel, seed: int = 0) -> None:
        super().__init__()
        self._program = program
        self._rng = np.random.default_rng(seed)
        self.initial_latent_override: Optional[Dict[str, Any]] = None
        self.last_execution_failure: Optional[str] = None
        self.last_trajectory: Optional[LowLevelTrajectory] = None
        self.sim_env: Optional[Any] = None

    @property
    def program(self) -> ProgramWorldModel:
        """The wrapped program."""
        return self._program

    def initial_latent(
            self,
            state: State,
            rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
        """A latent for ``state``: the override if one is set, else a draw from
        the program's ``initial_latent`` (with ``rng`` or the model's own)."""
        if self.initial_latent_override is not None:
            return copy.deepcopy(self.initial_latent_override)
        latent = self._program.initial_latent(
            _observation(state), rng if rng is not None else self._rng)
        if not isinstance(latent, dict):
            raise utils.OptionExecutionFailure(
                "The world model's initial_latent must return a dict "
                f"(got {type(latent).__name__}).")
        return copy.deepcopy(latent)

    def get_next_state_and_num_actions(self, state: State,
                                       option: _Option) -> Tuple[State, int]:
        self.last_execution_failure = None
        self.last_trajectory = None
        try:
            latent = (copy.deepcopy(state.latent) if state.latent is not None
                      else self.initial_latent(state))
            nxt, next_latent, num_steps = _unpack_transition(
                self._program.transition(_observation(state), latent, option,
                                         self._rng))
            _check_next_state(state, nxt)
        except utils.OptionExecutionFailure:
            raise
        except Exception as e:  # pylint: disable=broad-except
            tail = traceback.format_exc().strip().splitlines()[-1]
            self.last_execution_failure = f"world model: {tail}"
            raise utils.OptionExecutionFailure(
                f"The world model's transition raised on {option.name}: "
                f"{tail}") from e
        nxt.latent = copy.deepcopy(next_latent)
        return nxt, max(1, num_steps)


# ── Data ─────────────────────────────────────────────────────


def option_transitions(
        traj: LowLevelTrajectory,
        predicates: Set[Predicate]) -> List[Tuple[State, _Option, State, int]]:
    """``(pre-state, option, post-state, num low-level steps)`` per executed
    option of ``traj``; empty when the actions carry no options."""
    if not traj.actions or not traj.actions[0].has_option():
        return []
    segments = segment_trajectory(traj, predicates)
    return [(seg.states[0], seg.get_option(), seg.states[-1], len(seg.actions))
            for seg in segments]


def feature_scales(
    trajectories: Sequence[LowLevelTrajectory]
) -> Dict[Tuple[str, str], float]:
    """Per ``(type, feature)`` standard deviation over every recorded state,
    floored, as the unit of the observation distance."""
    values: Dict[Tuple[str, str], List[float]] = {}
    for traj in trajectories:
        for state in traj.states:
            for obj in state:
                for feat, val in zip(obj.type.feature_names, state[obj]):
                    values.setdefault((obj.type.name, feat),
                                      []).append(float(val))
    return {
        key: max(float(np.std(vals)), _SCALE_FLOOR)
        for key, vals in values.items()
    }


def observation_distance(pred: State, actual: State,
                         scales: Dict[Tuple[str, str], float]) -> float:
    """Mean absolute feature error of ``pred`` against ``actual`` in feature-
    std units (the Pinductor distance kernel's argument)."""
    errs: List[float] = []
    for obj in actual:
        if obj not in pred.data:
            return EXCEPTION_DISTANCE
        for feat, a, p in zip(obj.type.feature_names, actual[obj], pred[obj]):
            scale = scales.get((obj.type.name, feat), 1.0)
            errs.append(abs(float(p) - float(a)) / scale)
    return float(np.mean(errs)) if errs else 0.0


# ── Scoring (particle-filtered kernel pseudo-likelihood) ─────


@dataclass
class ProgramScore:
    """What ``score_program`` measured, renderable for the agent."""
    total: float
    num_particles: int
    # (trajectory index, score, number of option transitions)
    per_trajectory: List[Tuple[int, float, int]]
    # "type.feature" -> (mean error, max error, count), in feature-std units
    feature_errors: Dict[str, Tuple[float, float, int]]
    worst_examples: List[str]
    exceptions: List[str]

    def render(self) -> str:
        """The score report."""
        n_steps = sum(n for _, _, n in self.per_trajectory)
        lines = [
            f"World-model score (particle-filter kernel pseudo-likelihood, "
            f"{self.num_particles} particles): {self.total:.3f} over "
            f"{n_steps} option transitions in {len(self.per_trajectory)} "
            "trajectories. 0 is a perfect model; each unit is one "
            "feature-std of mean error per transition.",
        ]
        for idx, score, n in self.per_trajectory:
            per = score / n if n else 0.0
            lines.append(f"  traj {idx}: {score:.3f} over {n} transitions "
                         f"({per:.3f} per transition)")
        if self.feature_errors:
            lines.append("")
            lines.append("Per-feature error of the most likely particle "
                         "(mean / max, in feature-std units; count):")
            ranked = sorted(self.feature_errors.items(),
                            key=lambda kv: -kv[1][0])
            for label, (mean, mx, count) in ranked:
                lines.append(f"  {label:<32} {mean:8.3f} / {mx:8.3f}  "
                             f"(n={count})")
        if self.worst_examples:
            lines.append("")
            lines.append("Worst transitions:")
            lines.extend(f"  {ex}" for ex in self.worst_examples)
        if self.exceptions:
            lines.append("")
            lines.append("The program RAISED on some transitions (each "
                         f"charged {EXCEPTION_DISTANCE:g} units):")
            lines.extend(f"  {ex}" for ex in self.exceptions)
        return "\n".join(lines)


def score_program(
    program: ProgramWorldModel,
    trajectories: Sequence[LowLevelTrajectory],
    predicates: Set[Predicate],
    *,
    num_particles: int,
    kernel_bandwidth: float,
    rng: np.random.Generator,
    scales: Optional[Dict[Tuple[str, str], float]] = None,
    max_examples: int = 3,
    traj_indices: Optional[Set[int]] = None,
) -> ProgramScore:
    """Score ``program`` on the recorded trajectories.

    Per trajectory: ``num_particles`` latents are drawn from
    ``initial_latent``; at every recorded option each particle predicts
    the next observation from the RECORDED current one (teacher forcing;
    only the latent is carried), the step score is minus the weight-
    averaged observation distance (Eq. 7), the weights are multiplied by
    ``exp(-distance / kernel_bandwidth)`` and resampled when the
    effective sample size halves, and the trajectory score is the sum
    over steps (Eq. 8). Per-feature errors and the worst transitions
    come from the highest-weight particle's prediction.
    """
    assert num_particles >= 1
    scales = scales if scales is not None else feature_scales(trajectories)
    per_traj: List[Tuple[int, float, int]] = []
    feat_acc: Dict[str, List[float]] = {}
    examples: List[Tuple[float, str]] = []
    exceptions: List[str] = []

    def _note_exception(where: str) -> None:
        if len(exceptions) < 3:
            tail = traceback.format_exc().strip().splitlines()[-1]
            exceptions.append(f"{where}: {tail}")

    for ti, traj in enumerate(trajectories):
        if traj_indices is not None and ti not in traj_indices:
            continue
        transitions = option_transitions(traj, predicates)
        if not transitions:
            per_traj.append((ti, 0.0, 0))
            continue
        obs0 = _observation(traj.states[0])
        particles: List[Dict[str, Any]] = []
        for _ in range(num_particles):
            try:
                latent = program.initial_latent(obs0.copy(), rng)
                if not isinstance(latent, dict):
                    raise ValueError("initial_latent must return a dict")
                particles.append(copy.deepcopy(latent))
            except Exception:  # pylint: disable=broad-except
                _note_exception(f"traj {ti} initial_latent")
                particles.append({})
        weights = np.full(num_particles, 1.0 / num_particles)
        traj_score = 0.0
        for step, (obs_t, option, obs_next, _) in enumerate(transitions):
            dists = np.full(num_particles, EXCEPTION_DISTANCE)
            new_latents: List[Dict[str, Any]] = []
            preds: List[Optional[State]] = []
            for k in range(num_particles):
                try:
                    nxt, latent_k, _ = _unpack_transition(
                        program.transition(_observation(obs_t),
                                           copy.deepcopy(particles[k]), option,
                                           rng))
                    _check_next_state(obs_t, nxt)
                    dists[k] = observation_distance(nxt, obs_next, scales)
                    new_latents.append(latent_k)
                    preds.append(nxt)
                except Exception:  # pylint: disable=broad-except
                    _note_exception(f"traj {ti} step {step} ({option.name})")
                    new_latents.append(particles[k])
                    preds.append(None)
            traj_score -= float(np.sum(weights * dists))
            best = int(np.argmax(weights))
            pred = preds[best]
            if pred is not None:
                for obj in obs_next:
                    for feat, a, p in zip(obj.type.feature_names,
                                          obs_next[obj], pred[obj]):
                        scale = scales.get((obj.type.name, feat), 1.0)
                        err = abs(float(p) - float(a)) / scale
                        feat_acc.setdefault(f"{obj.type.name}.{feat}",
                                            []).append(err)
                        if err > 0.5:
                            examples.append(
                                (err,
                                 (f"traj {ti} step {step} ({option.name}): "
                                  f"{obj.name}.{feat} predicted {float(p):.4g}"
                                  f", observed {float(a):.4g} "
                                  f"({err:.2f} std)")))
            log_w = np.log(weights + 1e-300) - dists / kernel_bandwidth
            log_w -= log_w.max()
            new_w = np.exp(log_w)
            new_w /= new_w.sum()
            ess = 1.0 / float(np.sum(new_w**2))
            if ess < num_particles / 2.0:
                picks = rng.choice(num_particles, size=num_particles, p=new_w)
                new_latents = [copy.deepcopy(new_latents[i]) for i in picks]
                new_w = np.full(num_particles, 1.0 / num_particles)
            particles = new_latents
            weights = new_w
        per_traj.append((ti, traj_score, len(transitions)))

    feature_errors = {
        label: (float(np.mean(errs)), float(np.max(errs)), len(errs))
        for label, errs in feat_acc.items()
    }
    examples.sort(key=lambda e: -e[0])
    return ProgramScore(
        total=float(sum(s for _, s, _ in per_traj)),
        num_particles=num_particles,
        per_trajectory=per_traj,
        feature_errors=feature_errors,
        worst_examples=[ex for _, ex in examples[:max_examples]],
        exceptions=exceptions)


def roll_program_latents(
    program: ProgramWorldModel,
    traj: LowLevelTrajectory,
    predicates: Set[Predicate],
    rng: np.random.Generator,
) -> List[Optional[Dict[str, Any]]]:
    """The program's latent at every state of ``traj`` (one draw).

    Entry ``i`` is the latent in force when state ``i`` is observed: the
    initial draw for the first option's states, the post-option latent
    after each option completes. ``None`` from the first state whose
    latent could not be computed (the program raised). Consumed by
    ``sim.predicates()`` so latent-reading predicates are scored on
    meaningful values.
    """
    n_states = len(traj.states)
    transitions = option_transitions(traj, predicates)
    if not transitions:
        return [None] * n_states
    try:
        latent = copy.deepcopy(
            program.initial_latent(_observation(traj.states[0]), rng))
    except Exception:  # pylint: disable=broad-except
        return [None] * n_states
    out: List[Optional[Dict[str, Any]]] = [copy.deepcopy(latent)]
    for obs_t, option, _, num_steps in transitions:
        out.extend(copy.deepcopy(latent) for _ in range(num_steps - 1))
        try:
            _, latent, _ = _unpack_transition(
                program.transition(_observation(obs_t), copy.deepcopy(latent),
                                   option, rng))
        except Exception:  # pylint: disable=broad-except
            out.extend([None] * (n_states - len(out)))
            return out
        out.append(copy.deepcopy(latent))
    return out[:n_states] + [None] * max(0, n_states - len(out))
