"""The LLM agent arm of the continual protocol (docs/continual-protocol.md).

``AgentContinualApproach`` plays one level at a time through play
sessions: sandbox sessions of the existing SDK machinery whose tool
surface is the protocol's env and skill tools (``agent_sdk.tools.
continual_tools``) plus the belief probe. Between sessions the arm
runs the learning the agent asked for (the inherited simulator
synthesis, parameter fit and predicate invention), records the session
in ``attempts.md``, refreshes the learning data from the recorded
episodes, and checkpoints. A requeue resumes the interrupted session's
transcript (section 6.6).

The harness never chooses for the agent: whether to act, reset, learn
or end is decided inside the session; the loop here only services what
the session asked for and enforces two operational guards, the
per-session wall clock and the idle-session limit.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from predicators import utils
from predicators.agent_sdk import journal as journal_mod
from predicators.agent_sdk.play_prompts import build_play_query, \
    build_play_system_prompt, render_learning_status
from predicators.agent_sdk.session_base import AgentSessionFatalError, \
    query_fatal_error
from predicators.agent_sdk.tools.continual_tools import CONTINUAL_TOOL_NAMES, \
    PlayState, build_continual_tools, format_observation, visible_goal
from predicators.agent_sdk.tools.digests import render_options_digest, \
    render_types_digest
from predicators.approaches.agent_model_free_approach import \
    AgentModelFreeApproach
from predicators.approaches.agent_sim_predicate_invention_approach import \
    AgentSimPredicateInventionApproach
from predicators.run.episode import EpisodeState
from predicators.settings import CFG
from predicators.structs import Dataset, LowLevelTrajectory

if TYPE_CHECKING:  # pragma: no cover - the run package imports approaches
    from predicators.run.continual import ProtocolSession

SESSION_KIND = "play"


def _run_ended(reason: str, note: str = "") -> Exception:
    """``RunEnded`` without a module-level import of the run package."""
    # pylint: disable-next=import-outside-toplevel
    from predicators.run.continual import RunEnded
    return RunEnded(reason, note)


class AgentContinualApproach(AgentSimPredicateInventionApproach):
    """C1's learner (hybrid simulator, parameter fit, predicate invention)
    playing under the continual protocol."""

    _save_suffix = "AgentContinual"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._play_session: Optional[ProtocolSession] = None
        self._continual_level: Optional[int] = None
        self._sessions_played = 0
        self._learn_runs = 0
        self._session_in_flight = False
        self._last_handoff = ""
        self._episodes_at_last_learn = 0
        self._last_fit_status = ""

    @classmethod
    def get_name(cls) -> str:
        return "agent_continual"

    # -- Hooks the session machinery reads ------------------------------

    def _get_log_dir(self) -> str:
        """A stable directory per run, not per launch, so the sandbox and the
        CLI transcripts survive a requeue (section 6.6)."""
        if CFG.experiment_protocol == "continual":
            return os.path.abspath(
                os.path.join(CFG.continual_recordings_dir,
                             utils.get_config_path_str(), "agent"))
        return super()._get_log_dir()

    def prepare_for_continual(self, dataset: Dataset) -> None:
        """Take the offline data without a learning session: when to learn is
        the agent's decision."""
        AgentModelFreeApproach.learn_from_offline_dataset(self, dataset)
        self._sync_tool_context()

    def _get_agent_system_prompt(self) -> str:
        if self._learning_mode:
            return super()._get_agent_system_prompt()
        return build_play_system_prompt(self._get_solve_tool_names() or [],
                                        reset_cost=CFG.continual_reset_cost)

    def _get_solve_tool_names(self) -> Optional[List[str]]:
        return ["run_python"] + list(CONTINUAL_TOOL_NAMES)

    # -- The controller contract ----------------------------------------

    def play_level(self, session: ProtocolSession) -> None:
        """Play sessions until the level is won or the run ends."""
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
            learned = False
            if state.pending_learn is not None:
                self._run_learn(session, state.pending_learn)
                learned = True
            if state.pending_end_run is not None:
                self.save(session.level_index)
                session.end_run(state.pending_end_run)
            steps_after = session.observe().ledger.run_steps
            idle = 0 if (steps_after > steps_before or learned) else idle + 1
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
            ctx, session, state, save_render=self._save_render)
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
        sandbox = ctx.sandbox_dir
        render = self._save_render(f"session_{self._sessions_played + 1:03d}")
        observation = format_observation(obs,
                                         ctx,
                                         with_state=True,
                                         render_path=render)
        # The ledger is already the observation's last line; the query
        # shows it once more on its own so it cannot be missed.
        n_eps, n_steps = self._episode_counts(session)
        learning = render_learning_status(
            n_learn=self._learn_runs,
            sim_version=getattr(self, "_current_simulator_version", None),
            pred_version=getattr(self, "_current_predicates_version", None),
            fit_status=self._last_fit_status,
            n_episodes=n_eps,
            n_steps=n_steps,
            n_new_episodes=max(0, n_eps - self._episodes_at_last_learn))
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
            learning=learning,
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

    # -- Learning -----------------------------------------------------------

    def _run_learn(self, session: ProtocolSession, note: str) -> None:
        """Run the inherited learning over every recorded episode."""
        self._close_agent_session()
        trajectories = self._get_all_trajectories()
        logging.info(
            "[Continual agent] learning session %d requested (%s) over %d "
            "trajectories", self._learn_runs + 1, note, len(trajectories))
        t0 = time.time()
        try:
            self._learn_simulator(trajectories)
            self._last_fit_status = self._fit_status_text()
            outcome = "completed"
        except AgentSessionFatalError:
            raise
        except Exception as e:  # pylint: disable=broad-except
            logging.exception("[Continual agent] learning failed")
            self._last_fit_status = f"last learning failed: {e}"
            outcome = f"failed: {e}"
        self._learn_runs += 1
        n_eps, _ = self._episode_counts(session)
        self._episodes_at_last_learn = n_eps
        session.record_sandbox("learn_sessions", 1)
        session.record_sandbox("fits", 1)
        # The learner may have installed predicates: the runner's
        # abstraction (Wait targets, divergence checks) follows.
        self._sync_tool_context()
        session.abstract_predicates = set(self._get_all_predicates())
        journal_mod.append_entry(
            self._tool_context.sandbox_dir or self._get_log_dir(),
            f"Learning session {self._learn_runs} ({outcome}, "
            f"{time.time() - t0:.0f} s)",
            f"Requested with note: {note or '(none)'}. Trajectories: "
            f"{len(trajectories)}. Simulator version: "
            f"{getattr(self, '_current_simulator_version', None)}; "
            f"predicates version: "
            f"{getattr(self, '_current_predicates_version', None)}.",
            filename=journal_mod.ATTEMPTS_FILENAME)
        self.save(session.level_index)

    def _fit_status_text(self) -> str:
        """The last fit as one line for the prompt: the point estimate per
        parameter and the posterior sample count, never the raw result (its
        Jacobian dump is noise to the agent)."""
        result = getattr(self, "_last_fit_result", None)
        if result is None:
            return "no fit result"
        try:
            estimate = dict(result.point_estimate)
            n_samples = int(result.samples.shape[0])
        except (AttributeError, TypeError, ValueError, IndexError):
            return str(result)[:300]
        params = ", ".join(f"{k}={v:.4g}" for k, v in estimate.items())
        return (f"fitted {len(estimate)} parameter(s) from {n_samples} "
                f"posterior sample(s): {params}")[:600]

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
        n_eps = sum(1 for t in self._online_trajectories)
        n_steps = sum(len(t.actions) for t in self._online_trajectories)
        del session
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
                (f"\nLearning requested: {state.pending_learn}"
                 if state.pending_learn else "") +
                (f"\nHandoff: {state.handoff}" if state.handoff else ""))
        journal_mod.append_entry(self._tool_context.sandbox_dir
                                 or self._get_log_dir(),
                                 f"Session {number}",
                                 body,
                                 filename=journal_mod.ATTEMPTS_FILENAME)

    # -- Checkpoint -----------------------------------------------------------

    def _extra_save_state(self) -> Dict[str, Any]:
        state = super()._extra_save_state()
        state["continual"] = {
            "level": self._continual_level,
            "sessions_played": self._sessions_played,
            "learn_runs": self._learn_runs,
            "session_in_flight": self._session_in_flight,
            "last_handoff": self._last_handoff,
            "episodes_at_last_learn": self._episodes_at_last_learn,
            "last_fit_status": self._last_fit_status,
        }
        return state

    def _load_extra_save_state(self, save_dict: Dict[str, Any]) -> None:
        super()._load_extra_save_state(save_dict)
        cont = save_dict.get("continual") or {}
        self._continual_level = cont.get("level")
        self._sessions_played = int(cont.get("sessions_played", 0))
        self._learn_runs = int(cont.get("learn_runs", 0))
        self._session_in_flight = bool(cont.get("session_in_flight", False))
        self._last_handoff = str(cont.get("last_handoff", ""))
        self._episodes_at_last_learn = int(
            cont.get("episodes_at_last_learn", 0))
        self._last_fit_status = str(cont.get("last_fit_status", ""))
