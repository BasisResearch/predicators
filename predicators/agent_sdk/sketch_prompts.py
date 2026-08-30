"""Prompt construction for the solve and explore phases.

Two builders, both rendered from the Markdown templates in
``predicators/agent_sdk/prompts`` (see :mod:`prompt_templates`):

- :func:`build_solve_system_prompt` composes ``solve_system.md``: the
  agent's identity, its deliverable contract, the plan grammar, the
  tool semantics, the working principles, the run-record protocol,
  and (explore phase) the exploration setting. Everything that holds
  for every episode of a phase lives here.
- :func:`build_solve_prompt` composes ``solve_query.md``: the task
  (goal, scene, vocabulary), the run state (records, scheduled plans,
  open questions), and this episode's instructions.

The query never restates a rule from the system prompt; the split is
what keeps each rule stated exactly once.
"""
import re
from typing import Optional, Sequence, Set

from predicators import utils
from predicators.agent_sdk.prompt_templates import render
from predicators.settings import CFG
from predicators.structs import ParameterizedOption, Predicate, Task

_BLANK_RUN_RE = re.compile(r"\n{3,}")


def _join(parts: Sequence[str]) -> str:
    """Join non-empty prompt parts with blank lines, squeezing runs of blank
    lines that empty optional sections leave behind."""
    text = "\n\n".join(p for p in parts if p)
    return _BLANK_RUN_RE.sub("\n\n", text).strip("\n") + "\n"


def build_early_stop_note() -> str:
    """The exploration setting's early-stop sentence, from ``CFG``.

    Empty when early stopping is off. Describes the rule the run uses
    (train-driven: every attempt must be certified and solve for real;
    test-driven: consecutive perfect test phases).
    """
    if not CFG.online_learning_early_stopping:
        return ""
    if CFG.online_learning_early_stopping_by_test_solve_rate:
        n_perfect = CFG.online_learning_early_stopping_consecutive_perfect_tests
        phases = ("the next test phase solves" if n_perfect <= 1 else
                  f"{n_perfect} consecutive test phases each solve")
        return render("solve_system", "early_stop_test", phases=phases)
    attempts_clause = ("every episode of a cycle" if
                       CFG.online_learning_early_stopping_require_all_attempts
                       else "each train task's first episode of a cycle")
    return render("solve_system",
                  "early_stop_train",
                  attempts_clause=attempts_clause)


def build_solve_system_prompt(
    *,
    explore: bool,
    policy_mode: bool = False,
    propose_params: bool = True,
    ground_samplers: bool = False,
    physics_margin: bool = False,
    rule_param_margin: bool = False,
    use_journal: bool = True,
    execute_certified_plan: bool = True,
    early_stop_note: str = "",
    policy_max_options: int = 0,
    policy_max_repeated_failures: int = 0,
    policy_max_repeated_noops: int = 0,
) -> str:
    """Compose the solve-phase or explore-phase system prompt.

    ``explore`` selects the exploration identity, deliverable, and
    setting (``early_stop_note`` and ``execute_certified_plan`` only
    matter there); otherwise ``policy_mode`` selects the closed-loop
    policy deliverable over the captured plan. ``propose_params`` and
    ``ground_samplers`` shape the grammar; ``physics_margin`` and
    ``rule_param_margin`` describe the capture gate's extra checks;
    ``use_journal`` includes the run-record protocol.
    """
    assert not (explore and policy_mode), (
        "exploration delivers a plan sketch even in policy-mode runs")
    identity = render("solve_system",
                      "identity_explore" if explore else "identity_solve")
    if explore:
        certified_note = ""
        if execute_certified_plan:
            certified_note = " " + render("solve_system", "certified_note")
        deliverable = render("solve_system",
                             "deliverable_explore",
                             certified_note=certified_note)
    elif policy_mode:
        deliverable = render(
            "solve_system",
            "deliverable_policy",
            max_options=str(policy_max_options),
            max_repeated_failures=str(policy_max_repeated_failures),
            max_repeated_noops=str(policy_max_repeated_noops))
    else:
        deliverable = render("solve_system", "deliverable_plan")

    params_rule = render(
        "solve_system",
        "params_rule_propose" if propose_params else "params_rule_search")
    ground_sampler_rule = ""
    if propose_params and ground_samplers:
        ground_sampler_rule = "\n" + render("solve_system",
                                            "ground_sampler_rule")
    grammar = render("solve_system",
                     "grammar",
                     param_slot="[p1, p2]" if propose_params else "",
                     wait_slot="[]" if propose_params else "",
                     params_rule=params_rule,
                     ground_sampler_rule=ground_sampler_rule)

    validation_gate = ""
    if physics_margin:
        validation_gate += " " + render("solve_system",
                                        "validation_gate_physics")
    if rule_param_margin:
        validation_gate += " " + render("solve_system",
                                        "validation_gate_rule_params")
    tools = render("solve_system", "tools", validation_gate=validation_gate)

    principles = render("solve_system", "principles")
    if not explore:
        principles += "\n" + render("solve_system", "banking")

    run_records = ""
    if use_journal:
        journal_protocol = ""
        if not explore:
            journal_protocol = "\n\n" + render("solve_system",
                                               "journal_protocol_solve")
        run_records = render("solve_system",
                             "run_records",
                             journal_protocol=journal_protocol)

    setting = ""
    if explore:
        note = (" " + early_stop_note.strip()) if early_stop_note else ""
        setting = render("solve_system",
                         "exploration_setting",
                         early_stop_note=note)

    return _join([
        identity, deliverable, grammar, tools, principles, run_records, setting
    ])


def build_solve_prompt(
    task: Task,
    *,
    all_predicates: Set[Predicate],
    all_options: Set[ParameterizedOption],
    trajectory_summary: str = "",
    tool_names: Optional[Sequence[str]] = None,
    experiment_guidance: str = "",
    scheduled_plans: Optional[Sequence[str]] = None,
    initial_image_section: str = "",
    propose_params: bool = False,
    require_tool_validation: bool = False,
    explore_mode: bool = False,
    journal: str = "",
    strategy: str = "",
    attempts: str = "",
) -> str:
    """Compose the solve or explore query for ``task``.

    ``propose_params`` labels each option's parameter box as proposed by
    the agent (otherwise as found by the search) and picks the stuck-
    step advice. ``require_tool_validation`` selects the capture-gate
    instructions; ``explore_mode`` the experiment instructions (the two
    are exclusive). ``scheduled_plans`` lists the plans this cycle
    already queued, so the agent is asked for a complementary one.
    ``journal``, ``attempts``, and ``strategy`` are the run records'
    contents (``journal.py``); the protocol for using them is in the
    system prompt.
    """
    assert not (explore_mode and require_tool_validation), (
        "explore_mode accepts an uncaptured experiment sketch, which "
        "contradicts the hard capture gate of require_tool_validation")

    init_state = task.init
    objects = sorted(init_state, key=lambda o: o.name)
    obj_lines = [f"  {obj.name}: {obj.type.name}" for obj in objects]

    # Only expose goal atoms whose predicate is in the agent's current
    # predicate set: approaches that strip env predicates rely on
    # goal_nl and must not leak atoms of predicates the agent invents.
    goal_atoms = [
        str(a) for a in sorted(task.goal, key=str)
        if a.predicate in all_predicates
    ]

    option_lines = []
    for opt in sorted(all_options, key=lambda o: o.name):
        type_sig = ", ".join(t.name for t in opt.types)
        params_dim = opt.params_space.shape[0]
        param_info = ""
        if params_dim > 0:
            low = opt.params_space.low.tolist()
            high = opt.params_space.high.tolist()
            label = "params" if propose_params else "auto-searched params"
            desc = (", ".join(opt.params_description)
                    if opt.params_description else f"{params_dim}d")
            param_info = f"  [{label}: {desc}, range {low} to {high}]"
        option_lines.append(f"  {opt.name}({type_sig}){param_info}")

    atoms = utils.abstract(init_state, all_predicates)
    atom_lines = [str(a) for a in sorted(atoms, key=str)]

    pred_lines = []
    for pred in sorted(all_predicates, key=lambda p: p.name):
        type_sig = ", ".join(t.name for t in pred.types)
        line = f"  {pred.name}({type_sig})"
        if pred.natural_language_assertion is not None:
            names = [t.name for t in pred.types]
            line += f": {pred.natural_language_assertion(names)}"
        pred_lines.append(line)

    goal_nl_section = ""
    if task.goal_nl:
        goal_nl_section = render("solve_query",
                                 "goal_nl",
                                 goal_nl=task.goal_nl)

    # The env's public reward form (success condition plus costs) when
    # the task ships an evaluator that states one; never oracle values.
    scoring_section = ""
    evaluator = getattr(task, "evaluator", None)
    if evaluator is not None and evaluator.objective_description():
        scoring_section = render("solve_query",
                                 "scoring",
                                 objective=evaluator.objective_description())

    goal_atoms_section = ""
    if goal_atoms:
        goal_atoms_section = render("solve_query",
                                    "goal_atoms",
                                    goal_atoms="\n".join(goal_atoms))

    experiment_section = ""
    if experiment_guidance:
        experiment_section = render("solve_query",
                                    "experiment_guidance",
                                    guidance=experiment_guidance)

    tools_section = ""
    if tool_names:
        tools_section = render("solve_query",
                               "tools",
                               tool_list="\n".join(f"  - {t}"
                                                   for t in tool_names))

    strategy_section = ""
    if strategy:
        strategy_section = render("solve_query", "strategy", strategy=strategy)

    attempts_section = ""
    if attempts:
        attempts_section = render("solve_query", "attempts", attempts=attempts)

    journal_section = ""
    if journal or attempts:
        journal_section = render("solve_query",
                                 "journal",
                                 journal=journal
                                 or render("solve_query", "no_journal"))

    scheduled_section = ""
    if scheduled_plans:
        plans = "\n".join(f"Plan {i + 1}:\n{p}"
                          for i, p in enumerate(scheduled_plans))
        certified_rule = ""
        if any("belief-certified" in p for p in scheduled_plans):
            certified_rule = "\n\n" + render("solve_query", "certified_rule")
        scheduled_section = render("solve_query",
                                   "scheduled_plans",
                                   plans=plans,
                                   certified_rule=certified_rule)

    if explore_mode:
        instructions = render("solve_query", "instructions_explore")
    elif require_tool_validation:
        stuck = render("solve_query",
                       "stuck_tune" if propose_params else "stuck_revise")
        instructions = render("solve_query",
                              "instructions_capture",
                              stuck_advice=stuck)
    else:
        instructions = render("solve_query",
                              "instructions_capture",
                              stuck_advice=render("solve_query",
                                                  "stuck_revise"))

    body = render(
        "solve_query",
        "skeleton",
        opening=render("solve_query",
                       "opening_explore" if explore_mode else "opening_solve"),
        goal_nl_section=goal_nl_section,
        scoring_section=scoring_section,
        goal_atoms_section=goal_atoms_section,
        experiment_section=experiment_section,
        atoms="\n".join(atom_lines) if atom_lines else render(
            "solve_query", "no_atoms"),
        state=init_state.dict_str(indent=2),
        image_section=initial_image_section.strip("\n"),
        objects="\n".join(obj_lines),
        options="\n".join(option_lines),
        predicates="\n".join(pred_lines),
        trajectory_summary=trajectory_summary.strip("\n"),
        tools_section=tools_section,
        strategy_section=strategy_section,
        attempts_section=attempts_section,
        journal_section=journal_section,
        scheduled_plans_section=scheduled_section,
        instructions=instructions,
    )
    return _join([body])
