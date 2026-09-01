"""Construction of the environment, its tasks, the approach and the offline
dataset from the config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set

from predicators import utils
from predicators.approaches import create_approach
from predicators.approaches.base_approach import BaseApproach
from predicators.datasets import create_dataset
from predicators.envs import BaseEnv, create_new_env
from predicators.ground_truth_models import get_gt_options, \
    parse_config_included_options
from predicators.perception import create_perceiver
from predicators.pybullet_helpers.real_robot_executor import attach_real_robot
from predicators.settings import CFG
from predicators.structs import Dataset, Predicate, Task


@dataclass(frozen=True)
class EnvironmentSetup:
    """The environment and the task views built from it."""
    env: BaseEnv
    # The predicates the approach sees (excluded ones stripped, goals
    # replaced by agent-specific goals unless the approach is the oracle).
    preds: Set[Predicate]
    # Train tasks as the approach sees them (stripped, alt goals).
    approach_train_tasks: List[Task]
    # Train tasks as perceived, unstripped.
    train_tasks: List[Task]


def setup_environment() -> EnvironmentSetup:
    """Create the environment, select the predicates, and build the train task
    views."""
    # Create environment. Under real_robot_execute an executor is attached so
    # its rollouts drive the arm; deliberately HERE and not inside
    # create_new_env, because the planner builds its own envs through that
    # factory (the option model's private simulator, the shared skill
    # simulator) and those must stay pure simulation.
    env = create_new_env(CFG.env, do_cache=True, use_gui=CFG.use_gui)
    attach_real_robot(env)
    env.action_space.seed(CFG.seed)
    assert env.goal_predicates.issubset(env.predicates)

    preds = select_predicates(env)

    # Create train tasks
    env_train_tasks = env.get_train_tasks()
    perceiver = create_perceiver(CFG.perceiver)
    train_tasks = [perceiver.reset(t) for t in env_train_tasks]

    # Strip excluded predicates and prepare approach tasks
    stripped_train_tasks = [
        utils.strip_task(task, preds) for task in train_tasks
    ]
    approach_train_tasks = [
        task.replace_goal_with_alt_goal() for task in stripped_train_tasks
    ]
    return EnvironmentSetup(env, preds, approach_train_tasks, train_tasks)


def select_predicates(env: BaseEnv) -> Set[Predicate]:
    """The predicates the approach is given for ``env`` under the config."""
    included_preds, excluded_preds = utils.parse_config_excluded_predicates(
        env)
    if CFG.approach == "oracle":
        return included_preds
    return utils.replace_goals_with_agent_specific_goals(
        included_preds, excluded_preds, env)


def _options(env: BaseEnv) -> Set:
    if CFG.option_learner == "no_learning":
        return get_gt_options(env.get_name())
    return parse_config_included_options(env)


def setup_approach(env: BaseEnv, preds: Set[Predicate],
                   approach_train_tasks: List[Task]) -> BaseApproach:
    """Create the approach/agent named by the config."""
    approach_name = CFG.approach
    if CFG.approach_wrapper:
        approach_name = f"{CFG.approach_wrapper}[{approach_name}]"
    return create_approach(approach_name, preds, _options(env), env.types,
                           env.action_space, approach_train_tasks)


def create_offline_dataset(env: BaseEnv, train_tasks: List[Task],
                           preds: Set[Predicate],
                           approach: BaseApproach) -> Optional[Dataset]:
    """The offline dataset, if the approach or the demo renders need one."""
    if approach.is_learning_based or CFG.make_demo_videos or \
        CFG.make_demo_images:
        return create_dataset(env, train_tasks, _options(env), preds)
    return None
