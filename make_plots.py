"""
Figures from the results written by run_experiment.py.

  results/regret_curves.png          mean cumulative regret over time, scenarios A, B, C
  results/final_regret_vs_paper.png  final regret of ETC, UCB and UCB-Tuned: paper vs this run

Usage:  python3 run_experiment.py && python3 make_plots.py
"""
import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "results"

PALETTE = {
    "background": "#F6F4EF",
    "ink": "#1D2B2F",
    "muted": "#6B7780",
    "grid": "#DDD8CC",
    "ETC": "#1F5C63",
    "ETC-strict": "#0F2F34",
    "UCB": "#8A949B",
    "UCB-Tuned": "#C0613A",
    "paper": "#C9BFA9",
    "ours": "#1F5C63",
}
# ETC exploration length shown per scenario (the paper's best-performing setting)
ETC_M = {"A": 100, "B": 10000, "C": 10000}
TITLES = {"A": "Scenario A  (0.80 vs 0.90)",
          "B": "Scenario B  (0.895 vs 0.90)",
          "C": "Scenario C  (0.89 vs 0.895)"}


def style(ax):
    ax.set_facecolor(PALETTE["background"])
    ax.grid(True, color=PALETTE["grid"], linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PALETTE["grid"])
    ax.tick_params(colors=PALETTE["muted"], labelsize=9)


def plot_curves(raw):
    marks = raw["checkpoints"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor=PALETTE["background"])
    for ax, scen in zip(axes, "ABC"):
        style(ax)
        curves = raw["mean_regret_curves"][scen]
        m = ETC_M[scen]
        lines = [
            (f"ETC-greedy(m={m})", f"ETC-greedy m={m}", PALETTE["ETC"], "-"),
            (f"ETC(m={m})", f"ETC strict m={m}", PALETTE["ETC-strict"], "--"),
            ("UCB", "UCB", PALETTE["UCB"], "-"),
            ("UCB-Tuned", "UCB-Tuned", PALETTE["UCB-Tuned"], "-"),
        ]
        for key, label, color, ls in lines:
            ax.plot(marks, curves[key], color=color, linewidth=2.0, linestyle=ls,
                    label=f"{label}  ({curves[key][-1]:,.2f})")
        ax.set_xscale("log")
        ax.set_title(TITLES[scen], fontsize=11, color=PALETTE["ink"], loc="left")
        ax.set_xlabel("time step t (log scale)", fontsize=9, color=PALETTE["muted"])
        if scen == "A":
            ax.set_ylabel("mean cumulative regret", fontsize=9, color=PALETTE["muted"])
        ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=PALETTE["ink"])
    fig.text(0.5, 0.025,
             f"T = {raw['horizon']:,}; {raw['trials']:,} trials per algorithm. "
             "Legend values are final mean regrets, rounded to two decimals.",
             ha="center", fontsize=8.5, color=PALETTE["muted"])
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(os.path.join(RESULTS, "regret_curves.png"), dpi=170, facecolor=PALETTE["background"])
    plt.close(fig)


def plot_vs_paper():
    rows = list(csv.DictReader(open(os.path.join(RESULTS, "comparison.csv"))))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), facecolor=PALETTE["background"])
    for ax, scen in zip(axes, "ABC"):
        style(ax)
        m = ETC_M[scen]
        wanted = [(f"ETC(m={m})", f"ETC-greedy(m={m})", "ETC-greedy"),
                  (f"ETC(m={m})", f"ETC(m={m})", "ETC strict"),
                  ("UCB", "UCB", "UCB"),
                  ("UCB-Tuned", "UCB-Tuned", "UCB-Tuned")]
        for i, (paper_name, our_name, label) in enumerate(wanted):
            r = next(r for r in rows if r["scenario"] == scen and r["paper_algorithm"] == paper_name
                     and r["our_algorithm"] == our_name)
            paper, ours, sem = (float(r[key]) for key in ("paper_regret", "our_regret", "our_sem"))
            ax.bar(i - 0.2, paper, 0.38, color=PALETTE["paper"], label="paper mean" if i == 0 else None)
            ax.bar(i + 0.2, ours, 0.38, color=PALETTE["ours"], yerr=sem, capsize=3,
                   error_kw={"ecolor": PALETTE["ink"], "linewidth": 1},
                   label="ours, mean +/- 1 SE" if i == 0 else None)
            ax.annotate(f"{paper:,.2f}", (i - 0.2, paper), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=7.5, color=PALETTE["ink"])
            # Stagger adjacent labels, keeping our mean above its error-bar cap
            # and above the paper label even when the two means are very close.
            ax.annotate(f"{ours:,.2f}", (i + 0.2, max(ours + sem, paper)), xytext=(0, 15),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=7.5, color=PALETTE["ink"])
        ax.set_xticks(range(len(wanted)))
        ax.set_xticklabels([w[2] for w in wanted], fontsize=8.5, color=PALETTE["ink"])
        ax.set_title(TITLES[scen] + f"   ETC m={m}", fontsize=10.5, color=PALETTE["ink"], loc="left")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
        if scen == "A":
            ax.set_ylabel("final cumulative regret", fontsize=9, color=PALETTE["muted"])
            ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=PALETTE["ink"])
    horizon_trials = {(int(r["horizon"]), int(r["trials"])) for r in rows}
    if len(horizon_trials) != 1:
        raise ValueError("Paper-comparison rows must use one common horizon and trial count.")
    horizon, trials = next(iter(horizon_trials))
    fig.text(0.5, 0.035,
             f"T = {horizon:,}; {trials:,} trials. The same paper ETC mean is shown beside both local ETC variants.\n"
             "Local error bars show one standard error, not confidence intervals. Mean labels are rounded to two decimals.",
             ha="center", fontsize=8, color=PALETTE["muted"])
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.savefig(os.path.join(RESULTS, "final_regret_vs_paper.png"), dpi=170,
                facecolor=PALETTE["background"])
    plt.close(fig)


if __name__ == "__main__":
    with open(os.path.join(RESULTS, "raw_results.json")) as f:
        raw = json.load(f)
    plot_curves(raw)
    plot_vs_paper()
    print("saved regret_curves.png and final_regret_vs_paper.png in", RESULTS)
