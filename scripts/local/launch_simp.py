"""Run the code by taking in a YAML config file, in an interactive mode, as
opposed to submitting a slurm job."""
import argparse
import subprocess

from scripts.cluster_utils import config_to_cmd_flags, generate_run_configs


def _main() -> None:
    # Set up argparse.
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, type=str)
    args = parser.parse_args()

    # # generate configs--will only take the first one
    # cfg = next(generate_run_configs(args.config))
    # cmd_str = config_to_cmd_flags(cfg)

    cmds = []
    # Loop through all experiments
    for cfg in generate_run_configs(args.config):
        cmd_str = config_to_cmd_flags(cfg)
        if "use_classification_problem_setting" in cfg.flags:
            use_classification_problem_setting = cfg.flags[
                'use_classification_problem_setting']
        else:
            use_classification_problem_setting = False

        if use_classification_problem_setting:
            entry_point = "main_classification.py"
        else:
            entry_point = "main.py"
        cmd = f"python predicators/{entry_point} {cmd_str}"
        cmds.append(cmd)

    # run the command
    num_cmds = len(cmds)
    for i, cmd in enumerate(cmds):
        print(f"********* RUNNING COMMAND {i+1} of {num_cmds} *********")
        subprocess.run(cmd, shell=True, check=False)


if __name__ == "__main__":
    _main()
