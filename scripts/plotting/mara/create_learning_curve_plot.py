"""Create learning curves showing percentage solved over online learning
iterations.

Shows how different approaches improve over online learning cycles, with
each line representing a different approach and x-axis showing
iterations.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.analyze_results_directory import create_raw_dataframe, \
    pd_create_equal_selector

plt.style.use('ggplot')
pd.set_option('chained_assignment', None)

############################ Change below here ################################

# Details about the plt figure.
DPI = 500
FONT_SIZE = 18
Y_LIM = (-5, 105)

# Color palette for different approaches
COLORS = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # olive
    '#17becf',  # cyan
]

# All column names and keys to load into the pandas tables.
COLUMN_NAMES_AND_KEYS = [
    ("ENV", "env"),
    ("APPROACH", "approach"),
    ("EXCLUDED_PREDICATES", "excluded_predicates"),
    ("EXPERIMENT_ID", "experiment_id"),
    ("SEED", "seed"),
    ("AVG_TEST_TIME", "avg_suc_time"),
    ("AVG_NODES_CREATED", "avg_num_nodes_created"),
    ("LEARNING_TIME", "learning_time"),
    ("PERC_SOLVED", "perc_solved"),
    ("ONLINE_LEARNING_CYCLE", "cycle"),
]

DERIVED_KEYS = [("perc_solved",
                 lambda r: 100 * r["num_solved"] / r["num_test_tasks"])]

# The keys of the dict are labels for the legend, and the dict values are
# selectors to filter the dataframe for each approach that do online learning.
APPROACH_GROUPS = [
    # ("Ours",
    #  lambda df: df["EXPERIMENT_ID"].apply(lambda v: "predicate_invention" in v)),
    # ("Online NSRT",
    #  lambda df: df["EXPERIMENT_ID"].apply(lambda v: "online_nsrt_learning" in v)),
    ("MAPLE", lambda df: df["EXPERIMENT_ID"].apply(lambda v: "maple_q" in v)),
    ("Ours",
     lambda df: df["EXPERIMENT_ID"].apply(lambda v: "predicate_invention" in v)
     ),
    # ("Ours",
    #  lambda df: df["EXPERIMENT_ID"].apply(lambda v: "ours_1_request" in v)),
    # ("No param learn",
    #  lambda df: df["EXPERIMENT_ID"].apply(lambda v: "no_invent" in v)),
]

# Approaches that don't do online learning - show as horizontal lines at final performance
HORIZONTAL_LINE_GROUPS = [
    ("Oracle",
     lambda df: df["EXPERIMENT_ID"].apply(lambda v: "oracle" in v)),
    # ("Ours",
    #  lambda df: df["EXPERIMENT_ID"].apply(lambda v: "predicate_invention" in v)),
    # ("ViLa (zs)",
    #  lambda df: df["EXPERIMENT_ID"].apply(lambda v: "vlm_plan_zero_shot" in v)),
    # ("ViLa (fs)",
    #  lambda df: df["EXPERIMENT_ID"].apply(lambda v: "vlm_plan_few_shot" in v)),
]

# Which environments to create plots for
PLOT_ENVS = [
    ("Boil", pd_create_equal_selector("ENV", "pybullet_boil")),
    ("Coffee", pd_create_equal_selector("ENV", "pybullet_coffee")),
    ("Grow", pd_create_equal_selector("ENV", "pybullet_grow")),
    ("Fan", pd_create_equal_selector("ENV", "pybullet_fan")),
    ("Domino", pd_create_equal_selector("ENV", "pybullet_domino")),
]

#################### Should not need to change below here #####################


def _convert_cycle_to_numeric(cycle_str):
    """Convert cycle string to numeric value, with None -> -1 for sorting."""
    if cycle_str == "None" or cycle_str is None:
        return -1
    try:
        return int(cycle_str)
    except (ValueError, TypeError):
        return -1


def _get_learning_curves_for_approach(df, approach_selector, env_selector):
    """Get learning curves for a specific approach and environment.

    Returns:
        x_values: List of cycle numbers (starting from 0 for None)
        y_means: List of mean percentages solved
        y_stds: List of standard deviations
    """
    # Filter data for this approach and environment
    filtered_df = df[approach_selector(df) & env_selector(df)].copy()

    if filtered_df.empty:
        return [], [], []

    # Convert cycles to numeric and add offset so None (-1) becomes 0
    filtered_df['CYCLE_NUMERIC'] = filtered_df['ONLINE_LEARNING_CYCLE'].apply(
        _convert_cycle_to_numeric)
    filtered_df['X_VALUE'] = filtered_df[
        'CYCLE_NUMERIC'] + 1  # None (-1) -> 0, 0 -> 1, 1 -> 2, etc.

    # Group by cycle and compute mean/std across seeds
    grouped = filtered_df.groupby('X_VALUE')['PERC_SOLVED'].agg(
        ['mean', 'std']).reset_index()

    x_values = grouped['X_VALUE'].tolist()
    y_means = grouped['mean'].tolist()
    y_stds = grouped['std'].fillna(
        0).tolist()  # Fill NaN std with 0 for single data points

    return x_values, y_means, y_stds


def _get_final_performance_for_approach(df, approach_selector, env_selector):
    """Get final performance (mean and std) for approaches that don't do online
    learning.

    Returns:
        mean: Mean percentage solved across seeds
        std: Standard deviation across seeds
    """
    # Filter data for this approach and environment
    filtered_df = df[approach_selector(df) & env_selector(df)].copy()

    if filtered_df.empty:
        return None, None

    # For non-online learning approaches, we just take the mean/std across all seeds
    mean = filtered_df['PERC_SOLVED'].mean()
    std = filtered_df['PERC_SOLVED'].std()
    if pd.isna(std):  # Single data point case
        std = 0

    return mean, std


def _main() -> None:
    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "results", "learning_curves")
    os.makedirs(outdir, exist_ok=True)
    matplotlib.rcParams.update({'font.size': FONT_SIZE})

    # Load all raw data (don't group/aggregate yet)
    df = create_raw_dataframe(COLUMN_NAMES_AND_KEYS, DERIVED_KEYS)

    # Create learning curve plots for each environment
    for env_name, env_selector in PLOT_ENVS:
        fig, ax = plt.subplots(figsize=(10, 8))

        # Determine x-axis range by finding max iteration across all online learning approaches
        max_x = 0
        for approach_label, approach_selector in APPROACH_GROUPS:
            x_vals, _, _ = _get_learning_curves_for_approach(
                df, approach_selector, env_selector)
            if x_vals:
                max_x = max(max_x, max(x_vals))

        # Use at least 5 as max for reasonable plot range
        max_x = max(max_x, 5)

        # Plot each online learning approach as a separate line
        color_idx = 0
        for approach_label, approach_selector in APPROACH_GROUPS:
            x_vals, y_means, y_stds = _get_learning_curves_for_approach(
                df, approach_selector, env_selector)

            if not x_vals:  # Skip if no data for this approach
                continue

            color = COLORS[color_idx % len(COLORS)]
            color_idx += 1

            # Plot the line with error bars
            ax.errorbar(x_vals,
                        y_means,
                        yerr=y_stds,
                        label=approach_label,
                        marker='o',
                        capsize=5,
                        color=color)

        # Plot horizontal lines for non-online learning approaches
        for approach_label, approach_selector in HORIZONTAL_LINE_GROUPS:
            mean, std = _get_final_performance_for_approach(
                df, approach_selector, env_selector)

            if mean is None:  # Skip if no data for this approach
                continue

            color = COLORS[color_idx % len(COLORS)]
            color_idx += 1

            # Plot horizontal line spanning the full x range
            ax.axhline(y=mean,
                       label=approach_label,
                       linestyle='--',
                       alpha=0.8,
                       color=color)

            # Add error band around the horizontal line
            if std > 0:
                ax.fill_between([0, max_x],
                                mean - std,
                                mean + std,
                                alpha=0.2,
                                color=color)

        # Customize the plot
        ax.set_xlabel('Online Learning Iteration', color='black')
        ax.set_ylabel('Percentage Solved (%)', color='black')
        ax.set_title(f'{env_name}', color='black')
        ax.set_xlim(-0.5, max_x + 0.5)
        ax.set_ylim(Y_LIM)
        ax.legend(bbox_to_anchor=(1, 1.02), loc='upper left')
        # ax.grid(True, alpha=0.3)

        # Save the plot
        plt.tight_layout()
        filename = f"{env_name.lower()}_learning_curves.png"
        outfile = os.path.join(outdir, filename)
        plt.savefig(outfile, dpi=DPI, bbox_inches='tight')
        print(f"Wrote out to {outfile}")
        plt.close()


if __name__ == "__main__":
    _main()
