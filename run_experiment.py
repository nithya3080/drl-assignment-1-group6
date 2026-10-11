"""
Regret of ETC, ETC-greedy, UCB and UCB-Tuned on the three scenarios of
Wolf (2025), "A Framework for Fair Evaluation of Variance-Aware Bandit
Algorithms", arXiv:2510.27001.

  Scenario A  p = (0.80, 0.90)    large gap, Delta = 0.1
  Scenario B  p = (0.895, 0.90)   micro-gap, Delta = 0.005, lower variance
  Scenario C  p = (0.89, 0.895)   micro-gap, Delta = 0.005, higher variance

Protocol (paper, Section 3.2): horizon T = 1,000,000 steps, 100 independent
trials, ETC exploration lengths m in {10, 100, 1000, 10^4, 10^5}.

Usage
  python3 run_experiment.py              # full protocol, about 10 minutes on 2 cores
  python3 run_experiment.py --quick      # T = 100,000, 20 trials, about 1 minute
  python3 run_experiment.py --workers 4  # number of processes (default: all cores)

Output (folder results/)
  summary.csv       mean / std / median regret per scenario and algorithm
  etc_expected.csv  expected ETC regret from many simulated trials
  comparison.csv    our final regret next to the values printed in the paper
  raw_results.json  mean regret curves for plotting (see make_plots.py)
"""
import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import bandit_sim as sim

SCENARIOS = {"A": (0.80, 0.90), "B": (0.895, 0.90), "C": (0.89, 0.895)}
ETC_M = (10, 100, 1_000, 10_000, 100_000)
SEED = 42
# Exploration lengths for which the expected ETC-greedy regret is estimated from more trials
GREEDY_EXPECTATION = {("A", 100), ("B", 10_000), ("C", 10_000)}

# Values printed in the paper (Tables 3-5, T = 10^6, 100 runs): (average regret, suboptimal ratio)
PAPER = {
    ("A", "ETC(m=100)"): (10.10, 0.00010),
    ("A", "ETC(m=1000)"): (100.00, 0.00100),
    ("A", "ETC(m=10000)"): (1000.00, 0.01000),
    ("A", "UCB"): (238.17, 0.00238),
    ("A", "UCB-Tuned"): (30.67, 0.00031),
    ("B", "ETC(m=10000)"): (150.66, 0.03013),
    ("B", "UCB"): (1127.17, 0.22543),
    ("B", "UCB-Tuned"): (212.21, 0.04242),
    ("C", "ETC(m=10)"): (2150.00, 0.43000),
    ("C", "ETC(m=10000)"): (170.53, 0.03411),
    ("C", "ETC(m=100000)"): (500.00, 0.10000),
    ("C", "UCB"): (1172.57, 0.23451),
    ("C", "UCB-Tuned"): (226.09, 0.04522),
}


def describe(final_regret, subopt_ratio):
    n = len(final_regret)
    return {
        "mean": float(final_regret.mean()),
        "std": float(final_regret.std(ddof=1)),
        "sem": float(final_regret.std(ddof=1) / np.sqrt(n)),
        "median": float(np.median(final_regret)),
        "subopt_ratio": float(np.mean(subopt_ratio)),
    }


def run_task(task):
    """One (scenario, algorithm) simulation. Seeds depend only on the task, not on execution order."""
    kind, scen, m, horizon, trials, expectation_trials = task
    p = SCENARIOS[scen]
    s_idx = list(SCENARIOS).index(scen)
    code = {"etc": 0, "etc_exp": 1, "greedy": 2, "greedy_exp": 3, "ucb": 4, "ucb_tuned": 5}[kind]
    rng = np.random.default_rng([SEED, s_idx, code, m])
    marks = sim.checkpoints(horizon)
    result = {"kind": kind, "scenario": scen, "m": m}

    if kind == "etc":
        means = sim.trial_arm_means(p, trials, rng)
        regret, subopt, flag = sim.simulate_etc(means, m, horizon, marks, rng)
    elif kind == "greedy":
        means = sim.trial_arm_means(p, trials, rng)
        regret, subopt, flag = sim.simulate_etc_greedy(means, m, horizon, marks, rng)
    elif kind == "etc_exp":
        means = sim.trial_arm_means(p, expectation_trials, rng)
        regret, subopt, flag = sim.simulate_etc(means, m, horizon, marks[-1:], rng)
        result.update(trials=expectation_trials, expected_regret=float(regret[-1].mean()),
                      sem=float(regret[-1].std(ddof=1) / np.sqrt(expectation_trials)),
                      flagged_rate=float(flag.mean()))
        return result
    elif kind == "greedy_exp":
        means = sim.trial_arm_means(p, expectation_trials, rng)
        regret, subopt, flag = sim.simulate_etc_greedy(means, m, horizon, marks[-1:], rng)
        result.update(trials=expectation_trials, expected_regret=float(regret[-1].mean()),
                      sem=float(regret[-1].std(ddof=1) / np.sqrt(expectation_trials)),
                      flagged_rate=float(flag.mean()))
        return result
    else:
        means = sim.trial_arm_means(p, trials, rng)
        regret, subopt = sim.simulate_ucb(means, horizon, marks, rng, tuned=(kind == "ucb_tuned"))
        flag = None

    result.update(curve=regret.mean(axis=1).tolist(), flagged=None if flag is None else int(flag.sum()),
                  **describe(regret[-1], subopt))
    return result


def algorithm_name(kind, m):
    return {"etc": f"ETC(m={m})", "greedy": f"ETC-greedy(m={m})", "ucb": "UCB", "ucb_tuned": "UCB-Tuned"}[kind]


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([round(v, 5) if isinstance(v, float) else v for v in r])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="T = 100,000 and 20 trials")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    horizon, trials = (100_000, 20) if args.quick else (1_000_000, 100)
    strict_exp, greedy_exp = (5_000, 100) if args.quick else (100_000, 1_000)
    os.makedirs(args.out, exist_ok=True)

    tasks = []
    for scen in SCENARIOS:
        for m in ETC_M:
            tasks.append(("greedy", scen, m, horizon, trials, 0))
            tasks.append(("etc", scen, m, horizon, trials, 0))
            tasks.append(("etc_exp", scen, m, horizon, 0, strict_exp))
            if (scen, m) in GREEDY_EXPECTATION:
                tasks.append(("greedy_exp", scen, m, horizon, 0, greedy_exp))
        tasks.append(("ucb", scen, 0, horizon, trials, 0))
        tasks.append(("ucb_tuned", scen, 0, horizon, trials, 0))
    # long tasks first, so the worker pool stays busy to the end
    tasks.sort(key=lambda t: {"greedy_exp": 0, "greedy": 1, "ucb_tuned": 2, "ucb": 3}.get(t[0], 4))

    started = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for res in pool.map(run_task, tasks):
            results.append(res)
            label = algorithm_name(res["kind"], res["m"]) if "curve" in res else f"{res['kind']} m={res['m']}"
            print(f"[{time.time() - started:5.0f}s] scenario {res['scenario']}  {label}", flush=True)

    # ---- tables
    summary, curves, expected = [], {s: {} for s in SCENARIOS}, []
    for res in results:
        if "curve" in res:
            name = algorithm_name(res["kind"], res["m"])
            curves[res["scenario"]][name] = res["curve"]
            summary.append((res["scenario"], name, horizon, trials, res["mean"], res["std"], res["sem"],
                            res["median"], res["subopt_ratio"], "" if res["flagged"] is None else res["flagged"]))
        else:
            expected.append((res["scenario"], "ETC" if res["kind"] == "etc_exp" else "ETC-greedy", res["m"],
                             horizon, res["trials"], res["expected_regret"], res["sem"], res["flagged_rate"]))
    order = {"A": 0, "B": 1, "C": 2}
    algo_order = lambda n: (0 if n.startswith("ETC(") else 1 if n.startswith("ETC-g") else 2 if n == "UCB" else 3)
    summary.sort(key=lambda r: (order[r[0]], algo_order(r[1]), int(r[1].split("=")[1].rstrip(")")) if "=" in r[1] else 0))
    expected.sort(key=lambda r: (order[r[0]], r[1], r[2]))

    write_csv(os.path.join(args.out, "summary.csv"),
              ["scenario", "algorithm", "horizon", "trials", "mean", "std", "sem", "median", "subopt_ratio",
               "trials_on_worse_arm"], summary)
    write_csv(os.path.join(args.out, "etc_expected.csv"),
              ["scenario", "algorithm", "m", "horizon", "trials", "expected_regret", "sem", "worse_arm_rate"],
              expected)

    by_key = {(r[0], r[1]): r for r in summary}
    rows = []
    for (scen, name), (reg, ratio) in PAPER.items():
        variants = [name] if not name.startswith("ETC") else [name, name.replace("ETC(", "ETC-greedy(")]
        for ours in variants:
            r = by_key[(scen, ours)]
            rows.append((scen, name, ours, reg, r[4], r[6], r[4] / reg, ratio, r[8], horizon, trials))
    write_csv(os.path.join(args.out, "comparison.csv"),
              ["scenario", "paper_algorithm", "our_algorithm", "paper_regret", "our_regret", "our_sem",
               "our_over_paper", "paper_subopt_ratio", "our_subopt_ratio", "horizon", "trials"], rows)

    with open(os.path.join(args.out, "raw_results.json"), "w") as f:
        json.dump({"horizon": horizon, "trials": trials, "scenarios": SCENARIOS,
                   "checkpoints": sim.checkpoints(horizon).tolist(), "mean_regret_curves": curves}, f)

    print(f"\nSaved summary.csv, etc_expected.csv, comparison.csv, raw_results.json to {args.out}/"
          f"  ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
