"""Agent planner approach: fixed-vocabulary open-loop planning.

Combines online trajectory collection (via AgentPlanExplorer) with open-loop
option plan generation (via Claude Agent SDK). No predicate/process/type
invention — just stores trajectories and generates plans.

Example command:
    python predicators/main.py --env pybullet_domino \
        --approach agent_planner --seed 0 \
        --num_train_tasks 1 --num_test_tasks 1 \
        --num_online_learning_cycles 1 --explorer agent_plan
"""
import datetime
import inspect as _inspect
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, \
    cast

import dill as pkl
import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.agent_sdk.agent_session_mixin import AgentSessionMixin
from predicators.approaches import ApproachFailure
from predicators.approaches.base_approach import BaseApproach
from predicators.explorers import create_explorer
from predicators.explorers.base_explorer import BaseExplorer
from predicators.option_model import _OptionModelBase, create_option_model
from predicators.settings import CFG
from predicators.structs import Action, Dataset, GroundAtom, \
    InteractionRequest, InteractionResult, LowLevelTrajectory, Object, \
    ParameterizedOption, Predicate, State, Task, Type


class AgentPlannerApproach(AgentSessionMixin, BaseApproach):
    """Fixed-vocabulary open-loop planning via Claude Agent SDK.

    - Collects trajectories online using AgentPlanExplorer
    - At solve time, queries the agent for an option plan
    - No predicate/process/type invention
    """

    def __init__(self,
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 types: Set[Type],
                 action_space: Box,
                 train_tasks: List[Task],
                 *args: Any,
                 option_model: Optional[_OptionModelBase] = None,
                 **kwargs: Any) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks, *args, **kwargs)
        self._offline_dataset = Dataset([])
        self._online_trajectories: List[LowLevelTrajectory] = []
        if option_model is not None:
            self._option_model = option_model
        else:
            self._option_model = create_option_model(CFG.option_model_name)
        # Let the option model terminate Wait on atom change using the
        # approach's predicates (which may include invented ones). Looked
        # up lazily so the lambda picks up predicates invented after
        # __init__.
        if CFG.wait_option_terminate_on_atom_change:
            cast(  # pylint: disable=protected-access
                Any, self._option_model)._abstract_function = (
                    lambda s: utils.abstract(s, self._get_all_predicates()))
        self._online_learning_cycle = 0
        self._requests_train_task_idxs: Optional[List[int]] = None
        self._run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._pre_test_conversation_log: Optional[List[Dict[str, Any]]] = None
        self._agent_session_id: Optional[str] = None

        self._init_agent_session_state(types, initial_predicates,
                                       initial_options, train_tasks)

    @classmethod
    def get_name(cls) -> str:
        return "agent_planner"

    @property
    def is_learning_based(self) -> bool:
        return True

    def _get_log_dir(self) -> str:
        """Return per-run log directory (created by configure_logging)."""
        log_dir = super()._get_log_dir()
        os.makedirs(log_dir, exist_ok=True)
        logging.info(f"Logging agent queries/responses to: {log_dir}")
        return log_dir

    # ------------------------------------------------------------------ #
    # Overridable helpers (for subclass customisation)
    # ------------------------------------------------------------------ #

    def _get_all_options(self) -> Set[ParameterizedOption]:
        """Return the full set of options available for planning."""
        return self._initial_options

    def _get_all_predicates(self) -> Set[Predicate]:
        """Return the full set of predicates for abstraction."""
        return self._initial_predicates

    def _get_all_trajectories(self) -> List[LowLevelTrajectory]:
        """Return all trajectories (offline + online)."""
        return self._offline_dataset.trajectories + self._online_trajectories

    # ------------------------------------------------------------------ #
    # AgentSessionMixin hooks
    # ------------------------------------------------------------------ #

    # -- Prompt building blocks ----------------------------------------- #

    _SYSTEM_PROMPT_BASE = (
        "You are a planning agent. You observe task environments through "
        "inspection tools and generate option plans to achieve goals. "
        "You have access to read-only tools to inspect predicates, "
        "options, trajectories, and training tasks. Use these to "
        "understand the environment and generate effective plans.\n\n"
        "Some effects may not be immediate — if an action triggers a "
        "delayed process (e.g. water filling, dominoes cascading, "
        "heating), insert a Wait after it so the effect has time "
        "to occur before the next action. The Wait action holds the "
        "robot's current pose. You can annotate Wait with target atoms "
        "using `-> {atoms}` to specify exactly when it should terminate "
        "(e.g. `Wait(robot:Robot)[] -> {Boiled(water:water_type)}`). "
        "Use `NOT Pred(...)` for atoms that should become false. "
        "If no annotation is provided, the Wait terminates on any atom "
        "change. Without a Wait, the robot will proceed to the next "
        "action before the delayed effect has occurred, which might "
        "cause the plan to fail.")

    _SCRATCHPAD_SECTION = """
## Scratchpad — CRITICAL
You MUST maintain `./notes.md` as your working memory. \
**Read it at the very start of the session** and **read it \
again before every test_option_plan call** to remind yourself \
what you already tried. **Update it immediately after every \
test_option_plan call** — no exceptions.

Use this exact format for each option you are tuning:

```
## <OptionName> — Parameter Search
| # | params | outcome | notes |
|---|--------|---------|-------|
| 1 | [x, y, ...] | IK fail | ... |
| 2 | [x, y, ...] | success, JugNotAt... | ... |
```

After every test, append a row and update these summary fields:
- **Confirmed working params**: (list any that achieve the desired atoms)
- **Explored ranges**: e.g. "x: 0.9–1.05, y: 1.4–1.55" — look for GAPS
- **Unreachable region**: e.g. "y > 1.47 always IK-fails"
- **Next hypothesis**: what to try and why

The cycle is: Read notes → plan next experiment → run test → \
update notes → repeat. Without this loop you WILL forget what \
you tried and repeat the same failed parameters. Treat notes.md \
as your lab notebook — write after every single experiment.

**If you notice you have NOT updated notes after a test, STOP \
and update before doing anything else.**"""

    _VISUALIZE_STATE_SECTION = """
**visualize_state** modifies any object features (x, y, z, \
rotation, water_volume, is_on, etc.) and renders the scene \
WITHOUT running the full simulation. It is FREE (no physics, \
no failure modes) — use it liberally to build spatial \
understanding before spending expensive test_option_plan calls.

**When to use visualize_state:**
- **At the start**: visualize key objects to understand the \
layout and geometry (e.g. which direction does a part extend? \
where exactly would spatial relations like "under" or "on" \
be satisfied?)
- **Before testing params**: visualize the object at your \
candidate position to check if it looks right. Try multiple \
positions AND orientations — orientation changes how the \
object sits relative to nearby objects.
- **After a failed action**: visualize the object at BOTH \
where it actually ended up AND where you wanted it. Compare \
visually to understand the offset.
- **When stuck (3+ failures on the same step)**: STOP testing \
and switch to visualize_state. Move the object to 4-5 spread \
out positions to visually locate the right region. Also try \
different orientations — they change the offset between the \
action's target coordinates and the object's final position.
- **To understand reference geometry**: Visualize nearby \
objects and look at their shapes. The functional point of an \
object is often offset from its reported (x,y) position."""

    _ANNOTATE_SCENE_SECTION = """
**annotate_scene** draws markers, lines, and rectangles on \
the scene to mark reference points (object origins, target \
positions, reachable boundaries)."""

    _COMPOSE_SECTION = """
The two compose: visualize_state sets up the hypothetical \
scene, then annotate_scene overlays markers on it."""

    # -- System prompt --------------------------------------------------- #

    def _get_agent_system_prompt(self) -> str:
        use_scratchpad = CFG.agent_planner_use_scratchpad
        use_visualize = CFG.agent_planner_use_visualize_state
        use_annotate = CFG.agent_planner_use_annotate_scene

        sections = [self._SYSTEM_PROMPT_BASE]

        # Scratchpad
        if use_scratchpad:
            sections.append(self._SCRATCHPAD_SECTION)

        # Scene visualization
        if use_visualize or use_annotate:
            tools_str = " and ".join(t for flag, t in [
                (use_visualize, "visualize_state"),
                (use_annotate, "annotate_scene"),
            ] if flag)
            sections.append(
                f"\n## Scene Visualization — CRITICAL\n"
                f"You MUST use {tools_str} throughout debugging. "
                f"Without them you are guessing blindly at spatial parameters."
            )
            if use_visualize:
                sections.append(self._VISUALIZE_STATE_SECTION)
            if use_annotate:
                sections.append(self._ANNOTATE_SCENE_SECTION)
            if use_visualize and use_annotate:
                sections.append(self._COMPOSE_SECTION)

        # Tuning workflow (numbered steps, dynamic)
        steps = []
        if use_visualize or use_annotate:
            viz_tool = "visualize_state" if use_visualize else "annotate_scene"
            steps.append(
                f"**Use {viz_tool} first** to understand the spatial "
                "layout and narrow candidate positions before testing.")
        if use_scratchpad:
            steps.append(
                "**Read `./notes.md` before every test**, then **update it "
                "immediately after every test_option_plan call**. Record "
                "what you tried, what happened, and what you learned. "
                "This is your memory — without it you will repeat failures.")
        steps += [
            "**Review past session logs** in `./session_logs/` if available. "
            "Previous queries and tool results from earlier sessions are "
            "saved there. Read them to build on prior knowledge.",
            "**Inspect rendered images** from `./test_images/` when "
            "something goes wrong to understand the actual outcome. "
            "For finer-grained debugging, pass `save_low_level_action_images: "
            "true` to test_option_plan — this saves per-simulator-step images "
            "to `./test_images_low_level/`.",
            "**Expect geometric offsets.** The target position for "
            "options is often offset from the reference object's reported "
            "position due to object geometry. Explore a wide range around "
            "the object's coordinates, not just values close to the "
            "reported position.",
            "**Search coarse-to-fine.** For each continuous parameter, "
            "start with a WIDE grid spanning most of the valid range "
            "(e.g. test 4–5 spread-out values across [low, high]). "
            "Identify which coarse region works, THEN refine within it. "
            "Never spend more than 3 attempts tweaking values in a small "
            "neighborhood — if none work, jump to a different region. "
            "Check your notes for gaps in the explored range.",
            "**Vary ALL params, not just position.** Orientation and "
            "other parameters change offsets and feasibility. If an "
            "action fails at a position, try different values for the "
            "other parameters before giving up on that region. Test at "
            "least 2-3 values for each non-position parameter.",
        ]
        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
        sections.append(
            f"\n## Continuous Parameter Tuning\nFollow this workflow:\n"
            f"{numbered}")

        return "\n".join(sections)

    def _get_sandbox_reference_files(self) -> Dict[str, str]:
        files: Dict[str, str] = {
            "skill_factories/base.py":
            "predicators/ground_truth_models/skill_factories/base.py",
            "skill_factories/__init__.py":
            "predicators/ground_truth_models/skill_factories/__init__.py",
            "skill_factories/pick.py":
            "predicators/ground_truth_models/skill_factories/pick.py",
            "skill_factories/move_to.py":
            "predicators/ground_truth_models/skill_factories/move_to.py",
            "skill_factories/place.py":
            "predicators/ground_truth_models/skill_factories/place.py",
            "skill_factories/push.py":
            "predicators/ground_truth_models/skill_factories/push.py",
            "skill_factories/pour.py":
            "predicators/ground_truth_models/skill_factories/pour.py",
            "skill_factories/wait.py":
            "predicators/ground_truth_models/skill_factories/wait.py",
        }
        options_path = _get_gt_options_module_path(CFG.env)
        if options_path:
            files["options.py"] = options_path
        return files

    def _get_agent_tool_names(self) -> Optional[List[str]]:
        tools = [
            "inspect_options", "inspect_trajectories", "inspect_train_tasks",
            "test_option_plan"
        ]
        if CFG.agent_planner_use_annotate_scene:
            tools.append("annotate_scene")
        if CFG.agent_planner_use_visualize_state:
            tools.append("visualize_state")
        return tools

    # ------------------------------------------------------------------ #
    # Learning
    # ------------------------------------------------------------------ #

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        self._offline_dataset = dataset
        self._tool_context.offline_trajectories = dataset.trajectories
        if dataset.trajectories:
            self._tool_context.example_state = \
                dataset.trajectories[0].states[0]

    def get_interaction_requests(self) -> List[InteractionRequest]:
        explorer = self._create_explorer()
        requests: List[InteractionRequest] = []
        self._requests_train_task_idxs = []
        for _ in range(CFG.online_nsrt_learning_requests_per_cycle):
            task_idx = self._rng.choice(len(self._train_tasks))
            policy, termination_function = explorer.get_exploration_strategy(
                task_idx, CFG.timeout)
            req = InteractionRequest(train_task_idx=task_idx,
                                     act_policy=policy,
                                     query_policy=lambda s: None,
                                     termination_function=termination_function)
            requests.append(req)
            self._requests_train_task_idxs.append(task_idx)
        return requests

    def learn_from_interaction_results(
            self, results: Sequence[InteractionResult]) -> None:
        assert self._requests_train_task_idxs is not None
        for i, result in enumerate(results):
            task_idx = self._requests_train_task_idxs[i]
            traj = LowLevelTrajectory(result.states,
                                      result.actions,
                                      _train_task_idx=task_idx)
            self._online_trajectories.append(traj)

        # Update tool context
        self._sync_tool_context()

        logging.info(
            f"[Run {self._run_id}] Cycle {self._online_learning_cycle}: "
            f"collected {len(results)} trajectories, "
            f"{len(self._online_trajectories)} total online.")

        self.save(self._online_learning_cycle)
        self._online_learning_cycle += 1

    # ------------------------------------------------------------------ #
    # Solving
    # ------------------------------------------------------------------ #

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        self._sync_tool_context()
        self._tool_context.current_task = task
        try:
            option_plan = self._query_agent_for_option_plan(task)
        except Exception as e:
            raise ApproachFailure(f"Agent failed to produce option plan: {e}")

        preds = self._get_all_predicates()
        policy = utils.option_plan_to_policy(
            option_plan, abstract_function=lambda s: utils.abstract(s, preds))

        def _policy(s: State) -> Action:
            try:
                return policy(s)
            except utils.OptionExecutionFailure as e:
                raise ApproachFailure(e.args[0], e.info)

        return _policy

    # ------------------------------------------------------------------ #
    # Test phase lifecycle
    # ------------------------------------------------------------------ #

    def begin_test_phase(self) -> None:
        """Snapshot the learning conversation log before testing."""
        if self._agent_session is not None:
            import copy  # pylint: disable=import-outside-toplevel
            self._pre_test_conversation_log = copy.deepcopy(
                self._agent_session.conversation_log)
        else:
            self._pre_test_conversation_log = None

    def end_test_phase(self) -> None:
        """Restore the conversation log to its pre-test state."""
        if self._agent_session is not None \
                and self._pre_test_conversation_log is not None:
            self._agent_session._conversation_log = \
                self._pre_test_conversation_log  # pylint: disable=protected-access
        self._pre_test_conversation_log = None

    def _query_agent_for_option_plan(self, task: Task) -> list:
        """Query the agent for an option plan and parse it."""
        prompt = self._build_solve_prompt(task)
        responses = self._query_agent_sync(prompt)
        plan_text = self._extract_option_plan_text(responses)

        if not plan_text:
            # Log the raw responses for debugging
            n_responses = len(responses)
            types = [r.get("type") for r in responses]
            raise ApproachFailure(
                f"Agent returned empty plan text. "
                f"Got {n_responses} responses with types: {types}")

        return self._parse_and_ground_plan(plan_text, task)

    def _solve_prompt_scratchpad_line(self) -> str:
        """Return the notes.md bullet for the solve prompt, or empty."""
        if CFG.agent_planner_use_scratchpad:
            return (
                "- **Read `./notes.md` before every test_option_plan call** "
                "and **update it immediately after each call** — append a "
                "row to the parameter table and update the explored-ranges "
                "summary. If you realize you forgot to update, STOP and "
                "update before doing anything else.\n")
        return ""

    def _build_solve_prompt(self, task: Task) -> str:
        """Build the prompt for generating an option plan."""
        init_state = task.init
        objects = list(init_state)

        # Objects
        obj_strs = []
        for obj in sorted(objects, key=lambda o: o.name):
            obj_strs.append(f"  {obj.name}: {obj.type.name}")

        # Goal
        goal_strs = [str(a) for a in sorted(task.goal, key=str)]

        # Options
        option_strs = []
        for opt in sorted(self._get_all_options(), key=lambda o: o.name):
            type_sig = ", ".join(t.name for t in opt.types)
            params_dim = opt.params_space.shape[0]
            if params_dim > 0:
                low = opt.params_space.low.tolist()
                high = opt.params_space.high.tolist()
                if opt.params_description:
                    desc = ", ".join(opt.params_description)
                    param_info = (f", params=[{desc}], "
                                  f"low={low}, high={high}")
                else:
                    param_info = (f", params_dim={params_dim}, "
                                  f"low={low}, high={high}")
            else:
                param_info = ""
            option_strs.append(f"  {opt.name}({type_sig}{param_info})")

        # Current atoms
        atoms = utils.abstract(init_state, self._get_all_predicates())
        atom_strs = [str(a) for a in sorted(atoms, key=str)]

        # Trajectory summary
        traj_summary = self._build_trajectory_summary()

        # State features (compact)
        state_str = init_state.dict_str(indent=2)

        # Available tools
        tool_names = self._get_agent_tool_names()
        tools_str = ""
        if tool_names:
            tool_list = "\n".join(f"  - {t}" for t in tool_names)
            tools_str = f"\n## Available Tools\n{tool_list}\n"

        # Natural language goal description (if available)
        goal_nl_section = ""
        if task.goal_nl:
            goal_nl_section = f"""
## Goal Description
{task.goal_nl}
"""

        prompt = f"""You are solving a task. \
Generate an option plan to achieve the goal.
{goal_nl_section}
## Goal Atoms
{chr(10).join(goal_strs)}

## Initial State Atoms
{chr(10).join(atom_strs)}

## Initial State Features
{state_str}

## Objects
{chr(10).join(obj_strs)}

## Available Options
{chr(10).join(option_strs)}
{traj_summary}{tools_str}
## Instructions
Use your available tools to inspect the environment and test your plan before committing to it.

Based on the task information and any past trajectory data, output an option plan to achieve the goal.

After any action whose desired subgoal depends on a delayed process (e.g. water \
filling, dominoes cascading, heating), insert a Wait action to let the process \
complete before proceeding. You can annotate Wait with target atoms using \
`-> {{atoms}}` to specify exactly when it should terminate. Use `NOT Pred(...)` for \
atoms that should become false. If no annotation is provided, the Wait terminates on \
any atom change. Only use Wait when there is a genuine delayed effect; do not insert \
it between actions with immediate effects (e.g. Pick, Place).

For Wait with target atoms: `Wait(robot:Robot)[] -> {{Boiled(water:water_type)}}`
For negated targets: `Wait(robot:Robot)[] -> {{NOT Touching(a:block, b:block)}}`

**Important — parameter tuning workflow:**
- When a step fails or produces unexpected results, inspect the rendered images \
in `./test_images/` to see what actually happened in the scene. For step-by-step \
low-level frames, pass `save_low_level_action_images: true` — images are saved to \
`./test_images_low_level/`.
{self._solve_prompt_scratchpad_line()}\
- Review past session logs in `./session_logs/` if available — they contain prior queries and results.
- When a step fails (e.g. IK error), use the image + object poses to reason about \
WHY and adjust params directionally. Don't just try random nearby values.
- **Use visualize_state when stuck** — after 3+ failures on the same step, STOP \
testing and use visualize_state to move the object to several candidate positions \
and orientations. It's free (no physics). Find the right region visually, then test.
- **Vary all parameters, not just position** — orientation and other params affect \
both the outcome and whether the action succeeds. Try 2-3 values for each \
non-position parameter per target region.
- **Search coarse-to-fine**: spread initial attempts across the full parameter range. \
If 3 nearby values all fail the same way, jump to a very different region instead of \
continuing to tweak. Check your notes for gaps in explored ranges.

Output the plan with one option per line in this exact format:
 OptionName(obj1:type1, obj2:type2)[param1, param2]

If an option has no continuous parameters, use empty brackets: OptionName(obj1:type1)[]

Output ONLY the option plan lines at the end, after any analysis."""

        return prompt

    def _build_trajectory_summary(self) -> str:
        """Summarize trajectory data for context."""
        all_trajs = self._get_all_trajectories()
        if not all_trajs:
            return ""

        max_trajs = CFG.agent_sdk_max_trajectories_in_context
        recent = all_trajs[-max_trajs:]
        all_preds = self._get_all_predicates()
        lines = [
            f"\n## Trajectory Summary ({len(all_trajs)} total, "
            f"showing last {len(recent)})"
        ]

        for i, traj in enumerate(recent):
            n_steps = len(traj.actions)
            init_atoms = utils.abstract(traj.states[0], all_preds)
            final_atoms = utils.abstract(traj.states[-1], all_preds)
            new_atoms = final_atoms - init_atoms
            lost_atoms = init_atoms - final_atoms
            lines.append(f"\nTrajectory {i}: {n_steps} steps")
            if new_atoms:
                lines.append(
                    f"  Gained: "
                    f"{', '.join(str(a) for a in sorted(new_atoms, key=str))}")
            if lost_atoms:
                lines.append(
                    f"  Lost: "
                    f"{', '.join(str(a) for a in sorted(lost_atoms, key=str))}"
                )

        return "\n".join(lines)

    def _extract_option_plan_text(self, responses: List[Dict[str,
                                                             Any]]) -> str:
        """Extract plan text from the last assistant text response.

        Only uses the final assistant message to avoid including
        intermediate reasoning/tool-call text that precedes the actual
        option plan.
        """
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

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Strip markdown code fences wrapping the plan text."""
        lines = text.split('\n')
        # Remove leading/trailing ``` lines (with optional language tag)
        while lines and lines[0].strip().startswith('```'):
            lines.pop(0)
        while lines and lines[-1].strip().startswith('```'):
            lines.pop()
        return '\n'.join(lines)

    def _parse_wait_annotations(
        self,
        text: str,
        predicates: Set[Predicate],
        objects: Sequence[Object],
    ) -> List[Tuple[Set[GroundAtom], Set[GroundAtom]]]:
        """Parse ``-> {atoms}`` annotations from plan lines.

        Returns a list parallel to the option lines in the text. Each
        entry is ``(positive_atoms, negative_atoms)`` for Wait lines
        with annotations, or ``(set(), set())`` otherwise.
        """
        results: List[Tuple[Set[GroundAtom], Set[GroundAtom]]] = []
        option_names = {o.name for o in self._get_all_options()}
        for line in text.split('\n'):
            stripped = line.strip()
            if not stripped:
                continue
            first_token = stripped.split('(')[0]
            if first_token not in option_names:
                if results:
                    break
                continue
            if first_token == "Wait" and '->' in stripped:
                pos, neg = utils.parse_wait_target_annotations(
                    stripped, predicates, objects)
                results.append((pos, neg))
            else:
                results.append((set(), set()))
        return results

    def _parse_and_ground_plan(self, plan_text: str, task: Task) -> list:
        """Parse option plan text and ground into executable options."""
        objects = list(task.init)
        all_options = self._get_all_options()
        option_names = sorted(o.name for o in all_options)

        # Strip markdown code fences that agents often wrap plans in.
        cleaned_text = self._strip_code_fences(plan_text)

        # Extract Wait target annotations before stripping them
        wait_annotations = self._parse_wait_annotations(
            cleaned_text, self._get_all_predicates(), objects)

        # Strip annotations so the option plan parser doesn't choke
        parseable_text = utils.strip_wait_annotations(cleaned_text)

        parsed = utils.parse_model_output_into_option_plan(
            parseable_text,
            objects,
            self._types,
            all_options,
            parse_continuous_params=True)
        if not parsed:
            raise ApproachFailure(f"Parsed empty option plan from agent.\n"
                                  f"  Plan text:\n{plan_text}\n"
                                  f"  Available option names: {option_names}")

        grounded = []
        for i, (option, objs, params) in enumerate(parsed):
            try:
                params_arr = np.array(params, dtype=np.float32)
                ground_opt = option.ground(objs, params_arr)
                # Inject Wait target atoms from annotations
                if (ground_opt.name == "Wait" and i < len(wait_annotations)):
                    pos, neg = wait_annotations[i]
                    if pos:
                        ground_opt.memory["wait_target_atoms"] = pos
                    if neg:
                        ground_opt.memory["wait_target_neg_atoms"] = neg
                grounded.append(ground_opt)
            except Exception as e:  # pylint: disable=broad-except
                logging.warning(
                    f"[Run {self._run_id}] Failed to ground option "
                    f"{option.name}: {e}")
                break

        if not grounded:
            raise ApproachFailure("No options successfully grounded.")
        logging.info(f"[Run {self._run_id}] Agent produced plan with "
                     f"{len(grounded)} options.")
        return grounded

    # ------------------------------------------------------------------ #
    # Explorer
    # ------------------------------------------------------------------ #

    def _create_explorer(self) -> BaseExplorer:
        """Create explorer for interaction requests."""
        if CFG.explorer in ("agent_plan", "agent_bilevel"):
            self._sync_tool_context()
            return self._create_agent_explorer(
                self._get_all_predicates(),
                self._get_all_options(),
                name=CFG.explorer,
            )
        return create_explorer(
            CFG.explorer,
            self._get_all_predicates(),
            self._get_all_options(),
            self._types,
            self._action_space,
            self._train_tasks,
        )

    def _sync_tool_context(self) -> None:
        """Push current approach state into the shared ToolContext.

        The MCP tools (inspect_options, test_option_plan, etc.) read
        from the ToolContext dataclass, not from the approach directly.
        This method keeps them in sync after mutations (e.g. new
        trajectories collected, options added).  Called before each
        solve and learning interaction.  Subclasses should call super()
        and then set any additional fields (e.g. skill_factory_context).
        """
        self._tool_context.types = self._types
        self._tool_context.predicates = self._initial_predicates
        self._tool_context.options = self._initial_options
        self._tool_context.show_option_source = True
        ref_root = "/sandbox" if CFG.agent_sdk_use_docker_sandbox else "."
        self._tool_context.gt_options_ref_path = \
            f"{ref_root}/reference/options.py"
        self._tool_context.train_tasks = self._train_tasks
        self._tool_context.offline_trajectories = \
            self._offline_dataset.trajectories
        self._tool_context.online_trajectories = self._online_trajectories

        self._tool_context.log_dir = self._get_log_dir()
        self._tool_context.option_model = self._option_model
        all_trajs = (self._offline_dataset.trajectories +
                     self._online_trajectories)
        if all_trajs:
            self._tool_context.example_state = all_trajs[0].states[0]

        # Extract env from option model for scene rendering
        if self._option_model is not None and \
                hasattr(self._option_model, '_simulator'):
            self._tool_context.env = getattr(
                self._option_model._simulator,  # pylint: disable=protected-access
                '__self__',
                None)

    # ------------------------------------------------------------------ #
    # Save / Load
    # ------------------------------------------------------------------ #

    def save(self, online_learning_cycle: Optional[int] = None) -> None:
        """Save approach state to disk."""
        save_path = utils.get_approach_save_path_str()
        with open(f"{save_path}_{online_learning_cycle}.AgentPlanner",
                  "wb") as f:
            save_dict = {
                "offline_dataset":
                self._offline_dataset,
                "online_trajectories":
                self._online_trajectories,
                "online_learning_cycle":
                self._online_learning_cycle,
                "run_id":
                self._run_id,
                "agent_session_id": (self._agent_session.session_id
                                     if self._agent_session else None),
            }
            pkl.dump(save_dict, f)
            logging.info(f"[Run {self._run_id}] Saved approach to {save_path}_"
                         f"{online_learning_cycle}.AgentPlanner")

    def load(self, online_learning_cycle: Optional[int] = None) -> None:
        save_path = utils.get_approach_load_path_str()
        with open(f"{save_path}_{online_learning_cycle}.AgentPlanner",
                  "rb") as f:
            save_dict = pkl.load(f)

        self._offline_dataset = save_dict["offline_dataset"]
        self._online_trajectories = save_dict["online_trajectories"]
        self._online_learning_cycle = \
            save_dict["online_learning_cycle"] + 1
        self._agent_session_id = save_dict.get("agent_session_id")

        # Create new run_id for continued execution (each run gets own dir)
        # but log the original run_id for reference
        original_run_id = save_dict.get("run_id", "unknown")
        self._run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Re-sync tool context
        self._sync_tool_context()

        logging.info(
            f"[Run {self._run_id}] Loaded from previous run {original_run_id}: "
            f"{len(self._offline_dataset.trajectories)} offline, "
            f"{len(self._online_trajectories)} online trajectories")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _get_gt_options_module_path(env_name: str) -> Optional[str]:
    """Return repo-relative path to the options.py for the given env.

    Looks up the GroundTruthOptionFactory subclass that handles
    *env_name* and returns the path to its module file, relative to
    the repository root (e.g.
    ``predicators/ground_truth_models/boil/options.py``).
    """
    # Importing ground_truth_models triggers import_submodules, which
    # ensures all factory subclasses are registered.
    from predicators.ground_truth_models import \
        GroundTruthOptionFactory  # pylint: disable=import-outside-toplevel
    for cls in utils.get_all_subclasses(GroundTruthOptionFactory):
        if not cls.__abstractmethods__ and env_name in cls.get_env_names():
            module = _inspect.getmodule(cls)
            if module and module.__name__:
                return module.__name__.replace(".", os.sep) + ".py"
    return None
