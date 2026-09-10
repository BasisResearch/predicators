"""Plot a completed run's replay and a separately labeled counterfactual."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "logs/balloons_hatch_seed5_diagnostic_20260910"
OUTPUT = Path(__file__).resolve().parent / "figures"


def main():
    """Use true replay states to show both parts of the goal condition."""
    report = json.loads((DATA / "report.json").read_text())
    recorded = json.loads((DATA / "recorded_trace.json").read_text())
    alternative = json.loads(
        (DATA /
         "after_recorded_green_wait_release_blue_trace.json").read_text())
    lo = report["initial"]["band"]["lo"]
    hi = report["initial"]["band"]["hi"]
    plt.rcParams.update({
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False
    })
    fig, (height, speed) = plt.subplots(2,
                                        1,
                                        figsize=(10, 7),
                                        sharex=True,
                                        gridspec_kw={"height_ratios": [2, 1]},
                                        constrained_layout=True)
    fig.suptitle("Balloons hatch, MB seed 5: passing through is insufficient",
                 fontsize=16,
                 fontweight="bold")
    for rows, color, label in ((recorded, "#6d4bb3",
                                "Recorded: green, red, blue"),
                               (alternative, "#117d67",
                                "Replay alternative: green, blue")):
        steps = np.array([row["step"] for row in rows])
        height.plot(steps, [row["z"] for row in rows],
                    color=color,
                    linewidth=2,
                    label=label)
        speed.semilogy(steps,
                       np.maximum([row["speed"] for row in rows], 1e-5),
                       color=color,
                       linewidth=2)
    height.axhspan(lo,
                   hi,
                   color="#8fd1a7",
                   alpha=.3,
                   label="Target centre height: 0.8044 to 0.8544 m")
    height.axhline(hi, color="#398658", linewidth=.7)
    height.axhline(lo, color="#398658", linewidth=.7)
    height.set(ylabel="Payload centre height (m)", ylim=(.54, 1.025))
    height.legend(loc="upper right", frameon=False, fontsize=10)
    success = alternative[-1]
    height.scatter(success["step"],
                   success["z"],
                   color="#117d67",
                   marker="*",
                   s=150,
                   zorder=5)
    height.annotate("Alternative wins\n37 actions after the branch",
                    xy=(success["step"], success["z"]),
                    xytext=(1120, .95),
                    fontsize=10,
                    color="#117d67",
                    arrowprops={
                        "arrowstyle": "->",
                        "color": "#117d67"
                    })
    final = recorded[-1]
    height.scatter(final["step"], final["z"], color="#6d4bb3", s=28)
    height.annotate("Recorded final centre: 0.8645 m\n10.1 mm above the band",
                    xy=(final["step"], final["z"]),
                    xytext=(1530, .67),
                    fontsize=10,
                    color="#6d4bb3",
                    arrowprops={
                        "arrowstyle": "->",
                        "color": "#6d4bb3"
                    })
    speed.axhspan(1e-5, .01, color="#8fd1a7", alpha=.3)
    speed.axhline(.01, color="#398658", linestyle="--", linewidth=1)
    speed.text(1800,
               .014,
               "Win also requires speed < 0.01 m/s",
               fontsize=10,
               ha="right",
               color="#245c3b")
    speed.scatter(success["step"],
                  success["speed"],
                  color="#117d67",
                  marker="*",
                  s=100,
                  zorder=5)
    speed.set(ylabel="Payload speed (m/s)",
              xlabel="Primitive actions on the test level",
              ylim=(1e-5, 1),
              xlim=(1000, 1860))
    for axis in (height, speed):
        axis.grid(axis="y", alpha=.15)
        axis.axvline(1041, color="#777777", linestyle=":", linewidth=1)
        axis.axvline(1438, color="#777777", linestyle=":", linewidth=1)
    height.text(1048, .555, "Branch: red or blue", fontsize=9)
    height.text(1445, .555, "Recorded blue release", fontsize=9)
    fig.supxlabel(
        "Diagnostic on the same saved state and physics; the alternative is not a new agent result.",
        fontsize=10)
    OUTPUT.mkdir(exist_ok=True)
    for suffix in ("svg", "png"):
        path = OUTPUT / f"balloons-hatch-seed5-diagnostic.{suffix}"
        fig.savefig(path, dpi=160)
        if suffix == "svg":
            path.write_text("\n".join(
                line.rstrip()
                for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
