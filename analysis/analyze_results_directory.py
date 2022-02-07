"""Script to analyze experiments resulting from running the script
analysis/run_supercloud_experiments.sh."""

from typing import Tuple, Sequence
import glob
import dill as pkl
import numpy as np
import pandas as pd
from predicators.src.settings import CFG

GROUPS = [
    # "ENV",
    # "APPROACH",
    # "EXCLUDED_PREDICATES",
    "EXPERIMENT_ID",
    # "NUM_TRAIN_TASKS",
    # "CYCLE"
]

COLUMN_NAMES_AND_KEYS = [
    # ("ENV", "env"),
    # ("APPROACH", "approach"),
    # ("EXCLUDED_PREDICATES", "excluded_predicates"),
    ("EXPERIMENT_ID", "experiment_id"),
    ("SEED", "seed"),
    # ("NUM_TRAIN_TASKS", "num_train_tasks"),
    # ("CYCLE", "cycle"),
    ("NUM_SOLVED", "num_solved"),
    ("AVG_NUM_PREDS", "avg_num_preds"),
    ("AVG_TEST_TIME", "avg_suc_time"),
    ("AVG_SKELETONS", "avg_num_skeletons_optimized"),
    ("MIN_SKELETONS", "min_skeletons_optimized"),
    ("MAX_SKELETONS", "max_skeletons_optimized"),
    ("AVG_NODES_EXPANDED", "avg_num_nodes_expanded"),
    ("AVG_NODES_CREATED", "avg_num_nodes_created"),
    ("AVG_NUM_NSRTS", "avg_num_nsrts"),
    ("AVG_DISCOVERED_FAILURES", "avg_num_failures_discovered"),
    ("AVG_PLAN_LEN", "avg_plan_length"),
    ("AVG_EXECUTION_FAILURES", "avg_execution_failures"),
    ("NUM_TRANSITIONS", "num_transitions"),
    ("LEARNING_TIME", "learning_time"),
]


def create_dataframes(
        column_names_and_keys: Sequence[Tuple[str, str]], groups: Sequence[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns means, standard deviations, and sizes.

    When included, config_keys_to_include is a dict from column display
    name to config name (in CFG).
    """
    all_data = []
    column_names = [c for (c, _) in column_names_and_keys]
    for group in groups:
        assert group in column_names, f"Missing column {group}"
    for filepath in sorted(glob.glob(f"{CFG.results_dir}/*")):
        with open(filepath, "rb") as f:
            outdata = pkl.load(f)
        if "config" in outdata:
            config = outdata["config"].__dict__.copy()
            run_data_defaultdict = outdata["results"]
            assert not set(config.keys()) & set(run_data_defaultdict.keys())
            run_data_defaultdict.update(config)
        else:
            run_data_defaultdict = outdata
        (env, approach, seed, excluded_predicates, experiment_id,
         online_learning_cycle) = filepath[8:-4].split("__")
        if not excluded_predicates:
            excluded_predicates = "none"
        run_data = dict(
            run_data_defaultdict)  # want to crash if key not found!
        run_data.update({
            "env": env,
            "approach": approach,
            "seed": seed,
            "excluded_predicates": excluded_predicates,
            "experiment_id": experiment_id,
            "cycle": online_learning_cycle,
        })
        data = [run_data.get(k, np.nan) for (_, k) in column_names_and_keys]
        all_data.append(data)
    if not all_data:
        raise ValueError(f"No data found in {CFG.results_dir}/")
    # Group & aggregate data.
    pd.set_option("display.max_rows", 999999)
    df = pd.DataFrame(all_data)
    df.columns = column_names
    print("RAW DATA:")
    print(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    grouped = df.groupby(groups)
    means = grouped.mean()
    stds = grouped.std(ddof=0)
    sizes = grouped.size()
    return means, stds, sizes


def _main() -> None:
    means, stds, sizes = create_dataframes(COLUMN_NAMES_AND_KEYS, GROUPS)
    # Add standard deviations to the printout.
    for col in means:
        for row in means[col].keys():
            mean = means.loc[row, col]
            std = stds.loc[row, col]
            means.loc[row, col] = f"{mean:.2f} ({std:.2f})"
    means["NUM_SEEDS"] = sizes
    print("\n\nAGGREGATED DATA OVER SEEDS:")
    print(means)
    means.to_csv("supercloud_analysis.csv")
    print("\n\nWrote out table to supercloud_analysis.csv")


if __name__ == "__main__":
    _main()
