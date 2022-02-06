"""Create plots with online learning episodes on the x axis."""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
from predicators.analysis.analyze_results_directory import create_dataframes

pd.options.mode.chained_assignment = None  # default='warn'

############################ Change below here ################################

# PLOT_GROUPS is a nested dict where each outer dict corresponds to one plot,
# and each inner entry corresponds to one line on the plot.
# The keys of the outer dict are are (outfile name, plot title).
# The keys of the inner dict are experiment IDs, and the values are labels
# for the legend.
PLOT_GROUPS = {
    ("cover_regrasp", "Cover (Regrasp)"): {
        "cover_regrasp_naive_allexclude": "Naive (All Excluded)",
        "cover_regrasp_targeted_allexclude": "Targeted (All Excluded)",
    },
    ("blocks", "Blocks"): {
        "blocks_naive_allexclude": "Naive (All Excluded)",
        "blocks_targeted_allexclude": "Targeted (All Excluded)",
    },
}

# Y_KEY is the name of the metric that will be plotted on the y axes. See
# analyze_results_directory.py for all available metrics.
# Y_LABEL is used to label the y axis.
Y_KEY = "NUM_SOLVED"
Y_LABEL = "Test tasks solved"

#################### Should not need to change below here #####################


def _get_df_for_experiment_id(df: pd.DataFrame,
                              experiment_id: str) -> pd.DataFrame:
    df = df[df["EXPERIMENT_ID"] == experiment_id]
    df["CYCLE"].replace("None", "-1", inplace=True)
    df["CYCLE"] = df["CYCLE"].map(pd.to_numeric) + 1
    df = df.sort_values("CYCLE")
    return df


def _main() -> None:
    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "results")
    os.makedirs(outdir, exist_ok=True)
    matplotlib.rcParams.update({'font.size': 16})
    grouped_means, grouped_stds, _ = create_dataframes()
    means = grouped_means.reset_index()
    stds = grouped_stds.reset_index()
    for (group_name, plot_title), experiment_id_dict in PLOT_GROUPS.items():
        _, ax = plt.subplots()
        for experiment_id, label in experiment_id_dict.items():
            exp_means = _get_df_for_experiment_id(means, experiment_id)
            exp_stds = _get_df_for_experiment_id(stds, experiment_id)
            ax.errorbar(exp_means["CYCLE"],
                        exp_means[Y_KEY],
                        yerr=exp_stds[Y_KEY],
                        label=label)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(plot_title)
        ax.set_xlabel("Num online learning episodes")
        ax.set_ylabel(Y_LABEL)
        plt.legend()
        plt.tight_layout()
        outfile = os.path.join(outdir, f"{group_name}.png")
        plt.savefig(outfile)
        print(f"Wrote out to {outfile}.")


if __name__ == "__main__":
    _main()
