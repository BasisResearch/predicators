"""The play loop of the continual protocol's agent arms (docs/continual-
protocol.md, section 5).

``ContinualPlayMixin`` is the controller side of an agent arm: it plays
one level at a time through play sessions, sandbox sessions of the SDK
machinery whose tool surface is the protocol's env and skill tools
(``agent_sdk.tools.continual_tools``). Around each session it builds the
query from the level, the journal and the recorded episodes; after it,
it records the session in ``attempts.md``, refreshes the arm's data from
the recorded episodes, services what the session asked for, and
checkpoints. A requeue resumes the interrupted session's transcript
(section 6.6).

Why a mixin. The arms' learning and session machinery live in the
phased approach classes (``AgentModelFreeApproach`` and its
``AgentSimPredicateInventionApproach`` descendant), where the simulator
synthesis, the parameter fit, predicate invention, the sandbox and the
session managers are implemented. An arm keeps that class as its base
and mixes this loop in front of it, the way ``AgentSessionMixin`` and
``SamplerLearningMixin`` add their concerns; the phased loop's own entry
points (``_solve``, the explorers) are simply unused under the protocol.
The mixin has no base class of its own, so there is no diamond, and
what it needs from its host is declared below as the host contract.

The harness never chooses for the agent: whether to act, reset, learn
or end is decided inside the session; the loop only services what the
session asked for and enforces two operational guards, the per-session
wall clock and the idle-session limit.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, \
    Sequence, Set, Tuple

from predicators import utils
from predicators.agent_sdk import journal as journal_mod
from predicators.agent_sdk.play_prompts import build_play_query, \
    build_play_system_prompt, render_data_status
from predicators.agent_sdk.session_base import AgentSessionFatalError, \
    query_fatal_error
from predicators.agent_sdk.tools.continual_tools import CONTINUAL_TOOL_NAMES, \
    PlayState, build_continual_tools, format_observation, visible_goal
from predicators.agent_sdk.tools.digests import render_options_digest, \
    render_types_digest
from predicators.run.episode import EpisodeState
from predicators.settings import CFG
from predicators.structs import Dataset, LowLevelTrajectory, Predicate

if TYPE_CHECKING:  # pragma: no cover - the run package imports approaches
    from predicators.agent_sdk.session_manager import SessionManagerProtocol
    from predicators.agent_sdk.tools.context import ToolContext
    from predicators.run.continual import ProtocolSession

SESSION_KIND = "play"


def _run_ended(reason: str, note: str = "") -> Exception:
    """``RunEnded`` without a module-level import of the run package."""
    # pylint: disable-next=import-outside-toplevel
    from predicators.run.continual import RunEnded
    return RunEnded(reason, note)


def env_predicate_names(session: ProtocolSession) -> Set[str]:
    """The env's own predicate names, which an observation lists first."""
    return {p.name for p in session.env_predicates}


class ContinualPlayMixin:
    """The play loop, mixed in front of an agent arm's phased base class.

    The arm declares its tool surface through :meth:`_continual_tool_names`
    and its learning through :meth:`_learn_callable` and
    :meth:`_learning_status`; it wires the parent-specific overrides
    (system prompt, checkpoint state) with :meth:`_play_system_prompt`,
    :meth:`_continual_save_state` and :meth:`_load_continual_save_state`.
    """

    # -- Host contract -------------------------------------------------
    # The host is an AgentModelFreeApproach (or a descendant); these
    # declare what the loop reads and calls on it, so a typo fails
    # type-checking instead of surfacing at run time.
    if TYPE_CHECKING:
        # pylint: disable=unused-argument,missing-function-docstring
        _tool_context: ToolContext
        _agent_session: Optional[SessionManagerProtocol]
        _offline_dataset: Dataset
        _online_trajectories: List[LowLevelTrajectory]
        _train_tasks: List[Any]
        _initial_predicates: Set[Predicate]

        def _get_all_predicates(self) -> Set[Predicate]:
            raise NotImplementedError

        def _sync_tool_context(self) -> None:
            raise NotImplementedError

        def _ensure_agent_session(self) -> None:
            raise NotImplementedError

        def _close_agent_session(self) -> None:
            raise NotImplementedError

        def _query_agent_sync(self, message: str,
                              **query_kwargs: Any) -> List[Dict[str, Any]]:
            raise NotImplementedError

        def save(self, online_learning_cycle: Optional[int] = None) -> None:
            raise NotImplementedError

        # pylint: enable=unused-argument,missing-function-docstring

    # -- Loop state, checkpointed by _continual_save_state -------------
    _play_session: Optional[ProtocolSession] = None
    _continual_level: Optional[int] = None
    _sessions_played: int = 0
    _learn_runs: int = 0
    _session_in_flight: bool = False
    _last_handoff: str = ""
    _episodes_at_last_learn: int = 0
    _last_fit_status: str = ""

    # -- What the arm declares -------------------------------------------

    def _continual_tool_names(self) -> List[str]:
        """The MCP tools of a play session, in prompt order."""
        raise NotImplementedError

    def _learn_callable(  # pylint: disable=useless-return
            self, session: ProtocolSession) -> Optional[Callable[[str], str]]:
        """What ``learn_run`` calls, or ``None`` for an arm without a learning
        session (the tool then reports so)."""
        del session
        return None

    def _learning_status(self, session: ProtocolSession) -> str:
        """The learning-status block of the query; an arm with a belief model
        overrides it."""
        n_eps, n_steps = self._episode_counts(session)
        return render_data_status(n_episodes=n_eps, n_steps=n_steps)

    # -- Hooks the session machinery reads ------------------------------

    def _get_log_dir(self) -> str:
        """A stable directory per run, not per launch, so the sandbox and the
        CLI transcripts survive a requeue (section 6.6)."""
        return os.path.abspath(
            os.path.join(CFG.continual_recordings_dir,
                         utils.get_config_path_str(), "agent"))

    def _get_solve_tool_names(self) -> Optional[List[str]]:
        return list(self._continual_tool_names())

    def _play_system_prompt(self) -> str:
        """The system prompt of the arm's play sessions."""
        return build_play_system_prompt(self._continual_tool_names())

    def prepare_for_continual(self, dataset: Dataset) -> None:
        """Take the offline data without a learning session: when to learn is
        the agent's decision."""
        self._offline_dataset = dataset
        self._sync_tool_context()

    # -- The controller contract ----------------------------------------

    def play_level(self, session: ProtocolSession) -> None:
        """Play sessions until the level is won or lost, or the run ends."""
        self._play_session = session
        self._begin_level(session)
        idle = 0
        while True:
            obs = session.observe()
            if obs.state is EpisodeState.WIN:
                logging.info(
                    "[Continual agent] level %d won; ending its "
                    "sessions", session.level_index + 1)
                self._close_agent_session()
                return
            if session.level_card().lost:
                logging.info(
                    "[Continual agent] level %d lost (GAME_OVER with no "
                    "reset available); ending its sessions",
                    session.level_index + 1)
                self._close_agent_session()
                return
            steps_before = obs.ledger.run_steps
            state = self._play_one_session(session)
            self._sync_level_trajectories(session)
            if state.run_ended is not None:
                reason, note = state.run_ended
                raise _run_ended(reason, note)
            if state.pending_end_run is not None:
                self.save(session.level_index)
                session.end_run(state.pending_end_run)
            steps_after = session.observe().ledger.run_steps
            productive = steps_after > steps_before or state.learn_runs > 0
            idle = 0 if productive else idle + 1
            self.save(session.level_index)
            if idle >= CFG.continual_max_idle_sessions:
                raise _run_ended(
                    "agent_ended", f"stalled: {idle} consecutive sessions "
                    "without an environment step or a learning session")

    # -- One session --------------------------------------------------------

    def _play_one_session(self, session: ProtocolSession) -> PlayState:
        ctx = self._tool_context
        state = PlayState()
        ctx.extra_mcp_tools = build_continual_tools(
            ctx,
            session,
            state,
            save_render=self._save_render,
            tool_names=[
                n for n in self._continual_tool_names()
                if n in CONTINUAL_TOOL_NAMES
            ],
            learn=self._learn_callable(session))
        resume_id = self._resume_session_id()
        if resume_id is None:
            # Every session is a fresh context over the journal.
            self._close_agent_session()
        self._ensure_agent_session()
        assert self._agent_session is not None
        if resume_id is not None:
            self._agent_session.resume_session_id = resume_id
            logging.info("[Continual agent] resuming session %s", resume_id)
        # The query reads the journal and the tools save renders into
        # the sandbox, so it must exist before the first query.
        ensure_sandbox = getattr(self._agent_session, "_ensure_sandbox_dir",
                                 None)
        if ensure_sandbox is not None:
            ensure_sandbox()
        query = self._build_query(session, resumed=resume_id is not None)
        session_number = self._sessions_played + 1
        ctx.begin_attempt(session_number, CFG.continual_session_wall_clock)
        self._session_in_flight = True
        self.save(session.level_index)
        entries_before = len(session.index_entries())
        started = time.time()
        try:
            responses = self._query_agent_sync(query, kind=SESSION_KIND)
        finally:
            ctx.attempt_start = None
            ctx.attempt_deadline = None
            self._session_in_flight = False
            self._agent_session.resume_session_id = None
        dead = query_fatal_error(responses)
        if dead is not None:
            raise AgentSessionFatalError(
                f"play session died without doing work ({dead})")
        self._sessions_played += 1
        self._last_handoff = state.handoff
        self._account_session(session, responses, ctx.attempt_rollout_count)
        self._record_session(session, session_number, entries_before, state,
                             time.time() - started, responses)
        return state

    def _resume_session_id(self) -> Optional[str]:
        """The CLI session to continue after a requeue, if any."""
        if not self._session_in_flight or not CFG.agent_sdk_resume_session:
            return None
        path = os.path.join(self._get_log_dir(), "session_info.json")
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                info = json.load(f)
        except (OSError, ValueError):
            return None
        sid = info.get("session_id")
        return str(sid) if sid else None

    def _build_query(self, session: ProtocolSession, resumed: bool) -> str:
        ctx = self._tool_context
        obs = session.observe()
        ctx.current_observation = obs.frame
        sandbox = ctx.sandbox_dir
        render = self._save_render(f"session_{self._sessions_played + 1:03d}")
        observation = format_observation(
            obs,
            ctx,
            with_state=True,
            render_path=render,
            env_names=env_predicate_names(session))
        # The ledger is already the observation's last line; the query
        # shows it once more on its own so it cannot be missed.
        return build_play_query(
            session_number=self._sessions_played + 1,
            resumed=resumed,
            level_number=obs.level.index + 1,
            levels_total=obs.ledger.levels_total,
            goal_nl=obs.level.task.goal_nl or "",
            goal_atoms=visible_goal(ctx, obs.level.task),
            ledger=obs.ledger.footer(),
            observation=observation,
            skills=render_options_digest(
                session.list_skills(),
                gt_options_ref_path=ctx.gt_options_ref_path),
            predicates=self._render_predicates(),
            types=render_types_digest(ctx.types),
            learning=self._learning_status(session),
            journal=journal_mod.read_journal(sandbox),
            attempts=journal_mod.read_journal(
                sandbox, filename=journal_mod.ATTEMPTS_FILENAME),
            handoff=self._last_handoff,
        )

    def _render_predicates(self) -> str:
        env_names = {p.name for p in self._initial_predicates}
        lines = []
        for pred in sorted(self._get_all_predicates(), key=lambda p: p.name):
            sig = ", ".join(t.name for t in pred.types)
            origin = "environment" if pred.name in env_names else "yours"
            lines.append(f"- {pred.name}({sig}) [{origin}]")
        return "\n".join(lines) or "(none)"

    def _save_render(self, tag: str) -> Optional[str]:
        """Save a render into the sandbox's image dir; returns its sandbox-
        relative path."""
        session = self._play_session
        ctx = self._tool_context
        if session is None or not ctx.image_save_dir:
            return None
        try:
            src = session.render(tag)
        except Exception as e:  # pylint: disable=broad-except
            logging.debug("[Continual agent] render failed: %s", e)
            return None
        if not src:
            return None
        os.makedirs(ctx.image_save_dir, exist_ok=True)
        name = os.path.basename(src)
        dst = os.path.join(ctx.image_save_dir, name)
        try:
            shutil.copyfile(src, dst)
        except OSError as e:
            logging.debug("[Continual agent] render copy failed: %s", e)
            return None
        return f"./{os.path.basename(ctx.image_save_dir)}/{name}"

    # -- Data ---------------------------------------------------------------

    def _begin_level(self, session: ProtocolSession) -> None:
        k = session.level_index
        if self._continual_level != k:
            self._continual_level = k
            self._close_agent_session()
        # Only the levels reached so far are visible to the learner.
        self._train_tasks = [spec.task for spec in session.levels[:k + 1]]
        self._tool_context.train_tasks = list(self._train_tasks)
        self._tool_context.current_task = session.levels[k].task
        self._tool_context.test_task_idx = None
        self._sync_level_trajectories(session)
        session.abstract_predicates = set(self._get_all_predicates())

    def _sync_level_trajectories(self, session: ProtocolSession) -> None:
        """Rebuild the online trajectories from the recorded episodes of every
        level up to the current one."""
        k = session.level_index
        by_level: Dict[int, List[LowLevelTrajectory]] = {}
        for traj in self._online_trajectories:
            idx = traj.train_task_idx
            if idx is not None and idx < k:
                by_level.setdefault(int(idx), []).append(traj)
        for j in range(k):
            if j not in by_level:
                by_level[j] = self._episodes_to_trajectories(
                    session.previous_level_episodes(j), j)
        by_level[k] = self._episodes_to_trajectories(session.level_episodes(),
                                                     k)
        self._online_trajectories = [
            t for j in sorted(by_level) for t in by_level[j]
        ]
        self._sync_tool_context()

    def _episodes_to_trajectories(self, episodes: Sequence[Dict[str, Any]],
                                  level: int) -> List[LowLevelTrajectory]:
        out = []
        for ep in episodes:
            if not ep["actions"]:
                continue
            states = list(ep["states"])
            actions = list(ep["actions"])
            if len(states) != len(actions) + 1:
                continue
            out.append(
                LowLevelTrajectory(
                    states,
                    actions,
                    _train_task_idx=level,
                    _source_simulator_version=getattr(
                        self, "_current_simulator_version", None),
                    _source_predicates_version=getattr(
                        self, "_current_predicates_version", None),
                    _source_samplers_version=getattr(
                        self, "_current_samplers_version", None),
                    _env_reward=ep.get("reward"),
                    _env_terminated=ep.get("terminated"),
                ))
        return out

    def _episode_counts(self, session: ProtocolSession) -> Tuple[int, int]:
        """(recorded episodes, their steps) across the levels so far."""
        del session  # the online trajectories mirror the recording
        n_eps = len(self._online_trajectories)
        n_steps = sum(len(t.actions) for t in self._online_trajectories)
        return n_eps, n_steps

    # -- Records ------------------------------------------------------------

    def _account_session(self, session: ProtocolSession,
                         responses: List[Dict[str,
                                              Any]], rollouts: int) -> None:
        cost = 0.0
        turns = 0
        for entry in responses:
            if entry.get("type") == "result":
                if entry.get("total_cost_usd") is not None:
                    cost = float(entry["total_cost_usd"])
                if entry.get("num_turns") is not None:
                    turns = int(entry["num_turns"])
        session.record_sandbox("sessions", 1)
        session.record_sandbox("turns", turns)
        session.record_sandbox("llm_cost_usd", cost)
        session.record_sandbox("sim_rollouts", rollouts)

    def _record_session(self, session: ProtocolSession, number: int,
                        entries_before: int, state: PlayState, seconds: float,
                        responses: List[Dict[str, Any]]) -> None:
        """Append the harness's account of the session to attempts.md."""
        entries = session.index_entries()[entries_before:]
        lines = []
        for e in entries:
            event = e.get("event")
            if event == "invoke":
                params = ", ".join(f"{float(p):.3g}"
                                   for p in e.get("params", []))
                line = (f"- {e.get('skill')}[{params}]: {e.get('status')} "
                        f"in {e.get('steps')} steps; episode "
                        f"{e.get('state')}")
                if e.get("missing") or e.get("present"):
                    line += (" DIVERGED (missing " +
                             ", ".join(e.get("missing", [])) + "; present " +
                             ", ".join(e.get("present", [])) + ")")
                lines.append(line)
            elif event in ("reset", "win", "game_over"):
                lines.append(f"- {event} {e.get('reason', '')}".rstrip())
        subtype = next((e.get("subtype")
                        for e in responses if e.get("type") == "result"), None)
        how = "ended by session_end" if state.session_ended else (
            "hit the turn cap"
            if subtype == "error_max_turns" else "ended by the harness")
        card = session.level_card()
        body = (f"Level {card.index + 1}; {how}; {seconds:.0f} s; level "
                f"steps now {card.steps}, resets {card.resets}, invocations "
                f"{card.skill_invocations}.\n" +
                ("\n".join(lines) if lines else "- no environment action") +
                (f"\nLearning sessions run inside this session: "
                 f"{state.learn_runs}" if state.learn_runs else "") +
                (f"\nHandoff: {state.handoff}" if state.handoff else ""))
        journal_mod.append_entry(self._tool_context.sandbox_dir
                                 or self._get_log_dir(),
                                 f"Session {number}",
                                 body,
                                 filename=journal_mod.ATTEMPTS_FILENAME)

    # -- Checkpoint -----------------------------------------------------------

    def _continual_save_state(self) -> Dict[str, Any]:
        """The loop's state for the arm's checkpoint (under ``continual``)."""
        return {
            "continual": {
                "level": self._continual_level,
                "sessions_played": self._sessions_played,
                "learn_runs": self._learn_runs,
                "session_in_flight": self._session_in_flight,
                "last_handoff": self._last_handoff,
                "episodes_at_last_learn": self._episodes_at_last_learn,
                "last_fit_status": self._last_fit_status,
            }
        }

    def _load_continual_save_state(self, save_dict: Dict[str, Any]) -> None:
        """Inverse of :meth:`_continual_save_state`."""
        cont = save_dict.get("continual") or {}
        self._continual_level = cont.get("level")
        self._sessions_played = int(cont.get("sessions_played", 0))
        self._learn_runs = int(cont.get("learn_runs", 0))
        self._session_in_flight = bool(cont.get("session_in_flight", False))
        self._last_handoff = str(cont.get("last_handoff", ""))
        self._episodes_at_last_learn = int(
            cont.get("episodes_at_last_learn", 0))
        self._last_fit_status = str(cont.get("last_fit_status", ""))
