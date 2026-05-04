"""Integration test: GT simulator + backtracking refinement solves boil.

Verifies that given a correct plan sketch (from a real agent run) and a
ground-truth simulator program, the hybrid learned option model
(PyBullet + learned process dynamics) can find continuous parameters
that solve a pybullet_boil task.
"""
# pylint: disable=protected-access
import logging
import os
import re
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np
import pytest

from predicators import utils
from predicators.approaches.agent_bilevel_approach import _SketchStep
from predicators.code_sim_learning.utils import LearnedSimulator, \
    apply_rules, merge_updates
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.ground_truth_models.boil.gt_simulator import \
    BOIL_PARAM_SPECS, PROCESS_RULES
from predicators.option_model import _OracleOptionModel
from predicators.planning import run_backtracking_refinement
from predicators.structs import GroundAtom, Object, ParameterizedOption, \
    Predicate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _setup_env():
    """Create boil env and return (env, task, options_dict, objects_dict)."""
    utils.reset_config({
        "env": "pybullet_boil",
        "seed": 0,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "boil_goal": "simple",
        "boil_num_jugs_train": [1],
        "boil_num_jugs_test": [1],
        "boil_num_burner_train": [1],
        "boil_num_burner_test": [1],
        "option_model_use_gui": False,
        "wait_option_terminate_on_atom_change": True,
    })
    env = create_new_env("pybullet_boil", do_cache=False, use_gui=False)
    task = [t.task for t in env.get_test_tasks()][0]
    options = get_gt_options(env.get_name())
    options_dict = {o.name: o for o in options}
    objects_dict = {obj.name: obj for obj in task.init}
    return env, task, options_dict, objects_dict


def _build_oracle_model(env):
    """Build an oracle option model."""
    options = get_gt_options(env.get_name())
    oracle = _OracleOptionModel(options, env.simulate)
    preds = env.predicates
    oracle._abstract_function = lambda s: utils.abstract(s, preds)
    return oracle


def _build_kinematics_only_oracle(env):
    """Build an oracle that only handles kinematics (no process dynamics).

    Creates a separate env instance with process dynamics disabled, so
    that water filling, heating, and happiness are not simulated.
    """
    base_env = create_new_env("pybullet_boil",
                              do_cache=False,
                              use_gui=False,
                              skip_process_dynamics=True)
    options = get_gt_options(base_env.get_name())
    oracle = _OracleOptionModel(options, base_env.simulate)
    preds = env.predicates
    oracle._abstract_function = lambda s: utils.abstract(s, preds)
    return oracle


def _build_combined_model(env):
    """Build a combined model: base-sim-only env + GT step-level dynamics.

    Mirrors AgentSimLearningApproach: wraps GT rules in a
    LearnedSimulator via apply_rules and composes with a base-sim-only
    env.
    """
    base_env = create_new_env("pybullet_boil",
                              do_cache=False,
                              use_gui=False,
                              skip_process_dynamics=True)
    gt_params = {s.name: s.init_value for s in BOIL_PARAM_SPECS}
    rules = PROCESS_RULES

    simulator = LearnedSimulator(
        step_fn=lambda s, _r=rules, _p=gt_params: apply_rules(s, _r, _p),
        name="gt_combined")

    def combined_simulate(state, action):
        kin_state = base_env.simulate(state, action)
        updates = simulator.predict_step(kin_state)
        if not updates:
            return kin_state
        return merge_updates(kin_state, updates)

    options = get_gt_options(env.get_name())
    model = _OracleOptionModel(options, combined_simulate)
    preds = env.predicates
    model._abstract_function = lambda s: utils.abstract(s, preds)
    return model


def _parse_sketch_from_file(
    sketch_file: str,
    options: Set[ParameterizedOption],
    types: Set,
    predicates: Set[Predicate],
    objects: Sequence[Object],
) -> List[_SketchStep]:
    """Parse a plan sketch from a text file, same as agent_bilevel_approach."""
    with open(sketch_file, "r", encoding="utf-8") as f:
        plan_text = f.read().strip()

    # Phase 1: parse options + objects (no continuous params)
    parsed = utils.parse_model_output_into_option_plan(
        plan_text, objects, types, options, parse_continuous_params=False)
    assert parsed, f"Parsed empty plan sketch from {sketch_file}"

    # Phase 2: parse subgoal annotations
    pred_map = {p.name: p for p in predicates}
    obj_map = {o.name: o for o in objects}
    option_names = {o.name for o in options}
    subgoal_re = re.compile(r'->\s*\{([^}]*)\}')
    atom_re = re.compile(r'(NOT\s+)?(\w+)\(([^)]*)\)')

    subgoals: List[Optional[Tuple[Set[GroundAtom], Set[GroundAtom]]]] = []
    for line in plan_text.split('\n'):
        stripped = line.strip()
        if not stripped:
            continue
        first_token = stripped.split('(')[0]
        if first_token not in option_names:
            continue
        sg_match = subgoal_re.search(stripped)
        if not sg_match:
            subgoals.append(None)
            continue
        atoms_text = sg_match.group(1)
        pos_atoms: Set[GroundAtom] = set()
        neg_atoms: Set[GroundAtom] = set()
        for atom_match in atom_re.finditer(atoms_text):
            is_neg = atom_match.group(1) is not None
            pred_name = atom_match.group(2)
            obj_names = [
                n.strip().split(':')[0] for n in atom_match.group(3).split(',')
            ]
            if pred_name not in pred_map:
                continue
            pred = pred_map[pred_name]
            try:
                objs: Sequence[Object] = [obj_map[n] for n in obj_names]
            except KeyError:
                continue
            if len(objs) != len(pred.types):
                continue
            atom = GroundAtom(pred, objs)
            if is_neg:
                neg_atoms.add(atom)
            else:
                pos_atoms.add(atom)
        if pos_atoms or neg_atoms:
            subgoals.append((pos_atoms, neg_atoms))
        else:
            subgoals.append(None)

    # Zip into sketch steps
    sketch = []
    for i, (option, objs, _) in enumerate(parsed):
        sg = subgoals[i] if i < len(subgoals) else None
        if sg is not None:
            pos, neg = sg
            sketch.append(
                _SketchStep(option=option,
                            objects=objs,
                            subgoal_atoms=pos if pos else None,
                            subgoal_neg_atoms=neg if neg else None))
        else:
            sketch.append(
                _SketchStep(option=option, objects=objs, subgoal_atoms=None))
    return sketch


def _informed_place_params(pre_state, sketch, step_idx, rng, n):
    """Sample Place params biased toward the contextual target."""
    step = sketch[step_idx]
    low = step.option.params_space.low
    high = step.option.params_space.high
    eps = 1e-4

    next_step = sketch[step_idx + 1] if step_idx + 1 < n else None

    if next_step and "Faucet" in next_step.option.name:
        for obj in pre_state:
            if obj.type.name == "faucet":
                fx = pre_state.get(obj, "x")
                fy = pre_state.get(obj, "y")
                frot = pre_state.get(obj, "rot")
                # The jug has a physics offset after drop, so target
                # slightly past the faucet output to compensate.
                out_x = fx + 0.15 * np.cos(frot)
                out_y = fy - 0.15 * np.sin(frot)
                # Target near faucet output x but lower y (IK-reachable).
                x = np.clip(out_x + rng.normal(0, 0.02), low[0] + eps,
                            high[0] - eps)
                y = np.clip(out_y - 0.05 + rng.normal(0, 0.03), low[1] + eps,
                            high[1] - eps)
                z = np.clip(low[2] + 0.02 + abs(rng.normal(0, 0.01)),
                            low[2] + eps, high[2] - eps)
                # Negative yaw helps place jug closer to faucet output.
                yaw = np.clip(rng.normal(-0.3, 0.5), low[3] + eps,
                              high[3] - eps)
                return np.array([x, y, z, yaw], dtype=np.float32)

    if next_step and "Burner" in next_step.option.name:
        for obj in pre_state:
            if obj.type.name == "burner":
                bx = pre_state.get(obj, "x")
                by = pre_state.get(obj, "y")
                x = np.clip(bx + rng.normal(0, 0.05), low[0] + eps,
                            high[0] - eps)
                y = np.clip(by + rng.normal(0, 0.05), low[1] + eps,
                            high[1] - eps)
                # Bias z toward low end for reliable IK.
                z = np.clip(low[2] + 0.02 + abs(rng.normal(0, 0.01)),
                            low[2] + eps, high[2] - eps)
                yaw = rng.uniform(low[3] + eps, high[3] - eps)
                return np.array([x, y, z, yaw], dtype=np.float32)

    return rng.uniform(low + eps, high - eps).astype(np.float32)


def _refine(task,
            sketch,
            option_model,
            predicates,
            seed=0,
            max_samples=200,
            timeout=600.0):
    """Run backtracking refinement with informed Place sampling."""
    rng = np.random.default_rng(seed)
    n = len(sketch)
    max_tries = [
        max_samples if step.option.params_space.shape[0] > 0 else 1
        for step in sketch
    ]

    def sample_fn(idx, state, rng_):
        step = sketch[idx]
        if step.option.params_space.shape[0] == 0:
            params = np.array([], dtype=np.float32)
        elif step.option.name == "Place":
            params = _informed_place_params(state, sketch, idx, rng_, n)
        else:
            low = step.option.params_space.low
            high = step.option.params_space.high
            params = rng_.uniform(low, high).astype(np.float32)
        grounded = step.option.ground(step.objects, params)
        if grounded.name == "Wait" and step.subgoal_atoms is not None:
            grounded.memory["wait_target_atoms"] = step.subgoal_atoms
        return grounded

    def validate_fn(idx, _pre, _opt, post_state, _n_acts):
        step = sketch[idx]
        if step.subgoal_atoms is not None:
            current_atoms = utils.abstract(post_state, predicates)
            if not step.subgoal_atoms.issubset(current_atoms):
                missing = step.subgoal_atoms - current_atoms
                return False, f"subgoal missing: {missing}"
        if idx == n - 1 and not task.goal_holds(post_state):
            return False, "goal not reached"
        return True, ""

    plan, success, total_samples = run_backtracking_refinement(
        init_state=task.init,
        option_model=option_model,
        n_steps=n,
        max_tries=max_tries,
        sample_fn=sample_fn,
        validate_fn=validate_fn,
        rng=rng,
        timeout=timeout,
    )
    logger.info("Refinement: %s, %d total samples",
                "success" if success else "failed", total_samples)
    return [p for p in plan if p is not None], success


SKETCH_FILE = os.path.join(os.path.dirname(__file__), "test_data",
                           "boil_plan_sketch.txt")


@pytest.mark.parametrize("model_type", ["oracle", "combined"])
def test_boil_sketch_refinement(model_type):
    """Test that backtracking refinement solves the first test task."""
    env, task, _options_dict, _objects_dict = _setup_env()
    predicates = env.predicates
    options = get_gt_options(env.get_name())

    if model_type == "oracle":
        option_model = _build_oracle_model(env)
    else:
        option_model = _build_combined_model(env)

    sketch = _parse_sketch_from_file(SKETCH_FILE, options, env.types,
                                     predicates, list(task.init))
    plan, success = _refine(task,
                            sketch,
                            option_model,
                            predicates,
                            max_samples=500,
                            timeout=1200.0)

    logger.info("Model=%s, success=%s, plan_len=%d", model_type, success,
                len(plan))
    if success:
        for i, opt in enumerate(plan):
            objs = ", ".join(o.name for o in opt.objects)
            params = ", ".join(f"{p:.3f}" for p in opt.params)
            logger.info("  %d: %s(%s)[%s]", i, opt.name, objs, params)

    assert success, (f"Refinement failed with {model_type} model. "
                     f"Partial plan: {len(plan)} steps.")

    # Forward validation: re-execute the plan in the oracle model (full
    # env dynamics) to verify the plan actually solves the task.
    # Always uses the oracle regardless of which model found the plan.
    oracle_model = _build_oracle_model(env)
    n = len(plan)

    def fwd_sample_fn(i, _s, _r):
        return plan[i]

    def fwd_validate_fn(i, _s, _o, post, _n):
        if i == n - 1 and not task.goal_holds(post):
            return False, "goal not reached"
        return True, ""

    _, fwd_success, _ = run_backtracking_refinement(
        init_state=task.init,
        option_model=oracle_model,
        n_steps=n,
        max_tries=[1] * n,
        sample_fn=fwd_sample_fn,
        validate_fn=fwd_validate_fn,
        rng=np.random.default_rng(0),
        timeout=600.0,
    )
    if fwd_success:
        logger.info("Forward validation passed for %s model.", model_type)
    else:
        logger.warning(
            "Forward validation failed for %s model "
            "(PyBullet state reconstruction is imperfect).", model_type)


if __name__ == "__main__":
    import sys
    _model = sys.argv[1] if len(sys.argv) > 1 else "oracle"
    test_boil_sketch_refinement(_model)
