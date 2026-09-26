"""Utility functions for interacting with clusters."""

import copy
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from typing import Any, Collection, Dict, Iterator, List, Optional, Tuple

import yaml

SAVE_DIRS = [
    "results", "logs", "saved_datasets", "saved_approaches",
    "eval_trajectories"
]
SUPERCLOUD_IP = "txe1-login.mit.edu"
DEFAULT_BRANCH = "master"


@dataclass(frozen=True)
class RunConfig:
    """Config for a single run."""
    experiment_id: str
    approach: str
    env: str
    args: List[str]  # e.g. --make_test_videos
    flags: Dict[str, Any]  # e.g. --num_train_tasks 1
    use_gpu: bool  # e.g. --use_gpu True
    use_mujoco: bool  # needed for supercloud only
    train_refinement_estimator: bool  # e.g. --train_refinement_estimator True

    def __post_init__(self) -> None:
        # For simplicity, disallow overrides of the SAVE_DIRS.
        assert "results_dir" not in self.flags
        assert "log_dir" not in self.flags
        assert "approach_dir" not in self.flags
        assert "data_dir" not in self.flags


@dataclass(frozen=True)
class SingleSeedRunConfig(RunConfig):
    """Config for a single run with a single seed."""
    seed: int


@dataclass(frozen=True)
class BatchSeedRunConfig(RunConfig):
    """Config for a run where seeds are batched together."""
    start_seed: int
    num_seeds: int


def config_to_logfile(cfg: RunConfig, suffix: str = ".log") -> str:
    """Create a log file name from a run config."""
    if isinstance(cfg, SingleSeedRunConfig):
        seed = cfg.seed
    else:
        assert isinstance(cfg, BatchSeedRunConfig)
        seed = None
    name = "train_" if cfg.train_refinement_estimator else ""
    name += f"{cfg.env}__{cfg.approach}__{cfg.experiment_id}__{seed}" + suffix
    return name


def _cmd_flag_value(value: Any) -> str:
    """Render one flag value as a single shell token.

    List/tuple values (e.g. agent_sim_learn_kept_predicates_names) use
    the space-free bracketed form that utils.string_to_python_object
    parses back; str(list) would repr into multiple argv tokens and
    break parse_args' flag/value pairing. shlex.quote keeps the brackets
    one token under zsh/bash glob expansion.
    """
    if isinstance(value, (list, tuple)):
        inner = ",".join(str(x) for x in value)
        rendered = f"[{inner}]" if isinstance(value, list) else f"({inner})"
        return shlex.quote(rendered)
    return shlex.quote(str(value))


def config_to_cmd_flags(cfg: RunConfig) -> str:
    """Create a string of command flags from a run config."""
    arg_str = " ".join(f"--{a}" for a in cfg.args)
    flag_str = " ".join(f"--{f} {_cmd_flag_value(v)}"
                        for f, v in cfg.flags.items())
    args_and_flags_str = (f"--env {cfg.env} "
                          f"--approach {cfg.approach} "
                          f"--experiment_id {cfg.experiment_id} "
                          f"{arg_str} "
                          f"{flag_str}")
    if isinstance(cfg, SingleSeedRunConfig):
        args_and_flags_str += f" --seed {cfg.seed}"
    return args_and_flags_str


def _deep_merge(base: Dict[str, Any], override: Dict[str,
                                                     Any]) -> Dict[str, Any]:
    """Recursively merge override into base.

    Lists are concatenated, dicts are merged, scalars are overwritten by
    override.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result:
            if isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = _deep_merge(result[key], value)
            elif isinstance(result[key], list) and isinstance(value, list):
                result[key] = result[key] + value
            else:
                result[key] = value
        else:
            result[key] = copy.deepcopy(value)
    return result


def _resolve_config(config_filepath: str) -> Dict[str, Any]:
    """Load a single YAML file and recursively resolve its 'includes'."""
    with open(config_filepath, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    includes = config.pop("includes", [])
    # Resolve includes relative to the directory of the including file.
    base_dir = os.path.dirname(config_filepath)
    merged: Dict[str, Any] = {}
    for inc_path in includes:
        inc_filepath = os.path.join(base_dir, inc_path)
        inc_config = _resolve_config(inc_filepath)
        merged = _deep_merge(merged, inc_config)
    # The including file's own keys override the included ones.
    merged = _deep_merge(merged, config)
    return merged


def parse_seed_range(text: str) -> Tuple[int, int]:
    """Parse a seed range as (start, count): "3" is seed 3 alone, "2-4" is
    seeds 2, 3 and 4."""
    match = re.fullmatch(r"(\d+)(?:-(\d+))?", text.strip())
    if match is None:
        raise ValueError(f"seed range {text!r} is not N or N-M")
    start = int(match.group(1))
    stop = int(match.group(2)) if match.group(2) is not None else start
    if stop < start:
        raise ValueError(f"seed range {text!r} ends before it starts")
    return start, stop - start + 1


def _select(section: Dict[str, Any], keys: Optional[Collection[str]],
            kind: str) -> Dict[str, Any]:
    """The entries of an ENVS or APPROACHES section that a launch runs:

    the ``keys`` named on the command line, whether or not the config
    parks them, else every entry the config does not SKIP.
    """
    if keys is None:
        return {
            key: entry
            for key, entry in section.items() if not entry.get("SKIP", False)
        }
    unknown = sorted(set(keys) - set(section))
    if unknown:
        raise ValueError(f"unknown {kind} {unknown}; the config defines "
                         f"{sorted(section)}")
    return {key: entry for key, entry in section.items() if key in keys}


def parse_configs(config_filename: str) -> Iterator[Dict[str, Any]]:
    """Parse the YAML config file, resolving any 'includes' directives."""
    scripts_dir = os.path.dirname(os.path.realpath(__file__))
    configs_dir = os.path.join(scripts_dir, "configs")
    config_filepath = os.path.join(configs_dir, config_filename)
    # If the file uses includes, it must be a single document (no multi-doc).
    # Try resolving includes first; fall back to multi-doc for legacy files.
    with open(config_filepath, "r", encoding="utf-8") as f:
        raw_docs = list(yaml.safe_load_all(f))
    for raw_config in raw_docs:
        if raw_config and "includes" in raw_config:
            yield _resolve_config(config_filepath)
            return  # includes-based files are single-document
    # Legacy path: no includes, yield each document as-is.
    for config in raw_docs:
        if config is not None:
            yield config


def generate_run_configs(config_filename: str,
                         batch_seeds: bool = False,
                         round_name: Optional[str] = None,
                         envs: Optional[Collection[str]] = None,
                         approaches: Optional[Collection[str]] = None,
                         seeds: Optional[Tuple[int, int]] = None,
                         require_round: bool = False) -> Iterator[RunConfig]:
    """Generate run configs from a (local path) config file.

    The experiment id is ``<env key>-<approach key>``, suffixed with
    ``_<round>`` when the launch names a round: ``round_name``, else the
    config's ROUND key. ``envs`` and ``approaches`` pick entries by key
    and ``seeds`` is a (start, count) pair that overrides START_SEED and
    NUM_SEEDS. With ``require_round``, a continual-protocol run without
    a round is an error: its runs auto-resume from their run folders, so
    an unnamed relaunch would resume the previous launch.
    """
    for config in parse_configs(config_filename):
        start_seed, num_seeds = seeds or (config["START_SEED"],
                                          config["NUM_SEEDS"])
        args = config["ARGS"]
        flags = config["FLAGS"]
        if "USE_GPU" in config.keys():
            use_gpu = config["USE_GPU"]
        else:
            use_gpu = False
        if "USE_MUJOCO" in config.keys():
            use_mujoco = config["USE_MUJOCO"]
        else:
            use_mujoco = False
        if "TRAIN_REFINEMENT_ESTIMATOR" in config.keys():
            train_refinement_estimator = config["TRAIN_REFINEMENT_ESTIMATOR"]
        else:
            train_refinement_estimator = False
        launch_round = round_name or config.get("ROUND")
        if launch_round is not None and not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_]*", str(launch_round)):
            raise ValueError(f"round {launch_round!r} must be letters, "
                             "digits and underscores")
        suffix = f"_{launch_round}" if launch_round is not None else ""
        selected_approaches = _select(config["APPROACHES"], approaches,
                                      "approaches")
        selected_envs = _select(config["ENVS"], envs, "envs")
        # Loop over approaches.
        for approach_exp_id, approach_config in selected_approaches.items():
            approach = approach_config["NAME"]
            # Loop over envs.
            for env_exp_id, env_config in selected_envs.items():
                env = env_config["NAME"]
                # Create the experiment ID, args, and flags.
                experiment_id = f"{env_exp_id}-{approach_exp_id}{suffix}"
                run_args = list(args)
                if "ARGS" in approach_config:
                    run_args.extend(approach_config["ARGS"])
                if "ARGS" in env_config:
                    run_args.extend(env_config["ARGS"])
                run_flags = flags.copy()
                if "FLAGS" in approach_config:
                    run_flags.update(approach_config["FLAGS"])
                if "FLAGS" in env_config:
                    run_flags.update(env_config["FLAGS"])
                if (require_round and not suffix and
                        run_flags.get("experiment_protocol") == "continual"):
                    raise ValueError(
                        f"{experiment_id} runs the continual protocol, "
                        "whose runs auto-resume from their run folders: "
                        "name a round (--round or the config's ROUND) so "
                        "this launch cannot resume an earlier one")
                # Loop or batch over seeds.
                if batch_seeds:
                    yield BatchSeedRunConfig(experiment_id, approach, env,
                                             run_args, run_flags, use_gpu,
                                             use_mujoco,
                                             train_refinement_estimator,
                                             start_seed, num_seeds)
                else:
                    for seed in range(start_seed, start_seed + num_seeds):
                        yield SingleSeedRunConfig(experiment_id, approach, env,
                                                  run_args, run_flags, use_gpu,
                                                  use_mujoco,
                                                  train_refinement_estimator,
                                                  seed)


def get_cmds_to_prep_repo(branch: str) -> List[str]:
    """Get the commands that should be run while already in the repository but
    before launching the experiments."""
    old_dir_pattern = " ".join(f"{d}/" for d in SAVE_DIRS)
    return [
        "git stash",
        "git fetch --all",
        f"git checkout {branch}",
        "git pull",
        # Remove old results.
        f"rm -rf {old_dir_pattern}",
        "mkdir -p logs",
    ]


def run_cmds_on_machine(
    cmds: List[str],
    user: str,
    machine: str,
    ssh_key: Optional[str] = None,
    allowed_return_codes: Tuple[int, ...] = (0, )
) -> None:
    """SSH into the machine, run the commands, then exit."""
    host = f"{user}@{machine}"
    ssh_cmd = f"ssh -tt -o StrictHostKeyChecking=no {host}"
    if ssh_key is not None:
        ssh_cmd += f" -i {ssh_key}"
    server_cmd_str = "\n".join(cmds + ["exit"])
    final_cmd = f"{ssh_cmd} << EOF\n{server_cmd_str}\nEOF"
    response = subprocess.run(final_cmd,
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.STDOUT,
                              shell=True,
                              check=False)
    if response.returncode not in allowed_return_codes:
        raise RuntimeError(f"Command failed: {final_cmd}")
