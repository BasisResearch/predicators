"""Agent bilevel explorer: sketch → refine against mental model → execute real.

Produces a plan *sketch* via a Claude agent, runs backtracking refinement
against the approach's currently-learned option model (read from
``tool_context.option_model``), then rolls the refined plan out in the
real environment. When the mental model disagrees with reality (e.g. a
subgoal atom the mental model expected after a Wait doesn't actually
hold), the resulting trajectory provides a targeted learning signal for
online simulator synthesis.

Parallels ``AgentPlanExplorer`` for session plumbing and
``AgentBilevelApproach`` for the sketch/refine workflow.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.agent_sdk import bilevel_sketch
from predicators.agent_sdk.session_manager import AgentSessionManager, \
    run_query_sync
from predicators.agent_sdk.tools import ToolContext
from predicators.explorers.base_explorer import BaseExplorer
from predicators.settings import CFG
from predicators.structs import Action, ExplorationStrategy, \
    ParameterizedOption, Predicate, State, Task, Type


class AgentBilevelExplorer(BaseExplorer):
    """Queries a Claude agent for a plan sketch, refines it, and executes."""

    def __init__(self, predicates: Set[Predicate],
                 options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task],
                 max_steps_before_termination: int, tool_context: ToolContext,
                 agent_session: AgentSessionManager) -> None:
        super().__init__(predicates, options, types, action_space, train_tasks,
                         max_steps_before_termination)
        self._tool_context = tool_context
        self._agent_session = agent_session

    @classmethod
    def get_name(cls) -> str:
        return "agent_bilevel"

    # ------------------------------------------------------------------ #
    # Exploration strategy
    # ------------------------------------------------------------------ #

    def _get_exploration_strategy(self, train_task_idx: int,
                                  timeout: int) -> ExplorationStrategy:
        task = self._train_tasks[train_task_idx]
        # The approach syncs tool_context.option_model right before
        # constructing this explorer, so reading here picks up the most
        # recently learned model.
        option_model = self._tool_context.option_model
        assert option_model is not None, \
            "agent_bilevel explorer needs a synced option_model"

        try:
            prompt = bilevel_sketch.build_solve_prompt(
                task,
                all_predicates=self._predicates,
                all_options=self._options,
                trajectory_summary=self._build_trajectory_summary(),
                tool_names=self._agent_tool_names(),
            )
            responses = run_query_sync(self._agent_session, prompt)
            plan_text = self._extract_option_plan_text(responses)
            if not plan_text:
                raise ValueError("agent returned empty plan text")

            sketch = bilevel_sketch.parse_sketch_from_text(
                plan_text,
                task,
                predicates=self._predicates,
                options=self._options,
                types=self._types,
            )
            if not sketch:
                raise ValueError("parsed empty plan sketch")

            self._tool_context.last_sketch_subgoals = [
                (s.subgoal_atoms, s.subgoal_neg_atoms) for s in sketch
            ]
            self._tool_context.last_sketch_options = [
                (s.option.name, [o.name for o in s.objects]) for s in sketch
            ]

            # Explorer mode: keep subgoal validation ON so the mental
            # model can tell us which step it can't predict, but when
            # that happens, truncate the plan at that step (inclusive)
            # instead of backtracking. Steps beyond the first
            # disagreement are built on a false mental-model state, so
            # executing them in the real env adds noise rather than
            # signal. The truncated plan — Pick → ... → first failing
            # step — is the experiment we want to run. Final-goal check
            # is also off: the explorer isn't trying to solve the task
            # in the mental model.
            plan, success, _ = bilevel_sketch.refine_sketch(
                task,
                sketch,
                option_model,
                predicates=self._predicates,
                timeout=float(timeout),
                rng=np.random.default_rng(CFG.seed),
                max_samples_per_step=CFG.
                agent_bilevel_explorer_max_samples_per_step,
                check_subgoals=True,
                check_final_goal=False,
                truncate_on_subgoal_fail=True,
                log_state=CFG.agent_bilevel_log_state,
                run_id="agent_bilevel_explorer",
            )
            logging.info(
                f"agent_bilevel explorer: sketch has {len(sketch)} steps, "
                f"refined {len(plan)} "
                f"({'success' if success else 'partial'}).")
            if plan:
                plan_strs = []
                for i, opt in enumerate(plan):
                    obj_s = ", ".join(o.name for o in opt.objects)
                    par_s = ", ".join(f"{p:.4f}" for p in opt.params)
                    plan_strs.append(f"  {i}: {opt.name}({obj_s})[{par_s}]")
                logging.info("agent_bilevel explorer: experiment plan:\n%s",
                             "\n".join(plan_strs))

            if plan:
                policy = utils.option_plan_to_policy(
                    plan,
                    abstract_function=lambda s: utils.abstract(
                        s, self._predicates))
                return self._wrap_policy(policy), lambda _: False

            logging.info("agent_bilevel explorer: refinement produced zero "
                         "steps, falling back to random.")
        except Exception as e:  # pylint: disable=broad-except
            logging.warning(f"agent_bilevel explorer failed: {e}. "
                            "Falling back to random options.")

        if not CFG.agent_explorer_fallback_to_random:
            raise utils.RequestActPolicyFailure(
                "agent_bilevel explorer failed and fallback disabled.")
        return self._random_options_fallback()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _wrap_policy(
            self, policy: Callable[[State],
                                   Action]) -> Callable[[State], Action]:
        """Convert OptionExecutionFailure into RequestActPolicyFailure.

        This lets the main loop cleanly terminate the episode when the
        refined plan finishes or fails mid-execution (which is exactly
        the disagreement signal we want to collect).
        """

        def _wrapped(state: State) -> Action:
            try:
                return policy(state)
            except utils.OptionExecutionFailure as e:
                raise utils.RequestActPolicyFailure(e.args[0], e.info) from e

        return _wrapped

    def _random_options_fallback(self) -> ExplorationStrategy:
        """Fall back to random option sampling."""

        def fallback_policy(state: State) -> Action:
            del state
            raise utils.RequestActPolicyFailure(
                "Random option sampling failed!")

        policy = utils.create_random_option_policy(self._options, self._rng,
                                                   fallback_policy)
        return policy, lambda _: False

    def _agent_tool_names(self) -> Optional[List[str]]:
        """Return tool names exposed by the current session, if any."""
        return getattr(self._agent_session, "tool_names", None)

    def _build_trajectory_summary(self) -> str:
        """Summarize trajectory data for the agent."""
        all_trajs = (self._tool_context.offline_trajectories +
                     self._tool_context.online_trajectories)
        if not all_trajs:
            return ""

        max_trajs = CFG.agent_sdk_max_trajectories_in_context
        recent = all_trajs[-max_trajs:]
        lines = [
            f"\n## Trajectory Summary ({len(all_trajs)} total, "
            f"showing last {len(recent)})"
        ]

        for i, traj in enumerate(recent):
            n_steps = len(traj.actions)
            init_atoms = utils.abstract(traj.states[0], self._predicates)
            final_atoms = utils.abstract(traj.states[-1], self._predicates)
            new_atoms = final_atoms - init_atoms
            lost_atoms = init_atoms - final_atoms
            lines.append(f"\nTrajectory {i}: {n_steps} steps")
            if new_atoms:
                lines.append(
                    "  Gained: " +
                    f"{', '.join(str(a) for a in sorted(new_atoms, key=str))}")
            if lost_atoms:
                lines.append(
                    "  Lost: " +
                    f"{', '.join(str(a) for a in sorted(lost_atoms, key=str))}"
                )

        return "\n".join(lines)

    def _extract_option_plan_text(self, responses: List[Dict[str,
                                                             Any]]) -> str:
        """Extract plan text from the last assistant text response."""
        last_text_parts: List[str] = []
        for resp in responses:
            if resp.get("type") == "assistant":
                parts = [
                    block.get("text", "") for block in resp.get("content", [])
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                if parts:
                    last_text_parts = parts
        return "\n".join(last_text_parts)
