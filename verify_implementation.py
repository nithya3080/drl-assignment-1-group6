"""
Checks that bandit_sim.py behaves as the algorithm definitions say.

1. ETC: the share of trials that commit to the worse arm matches the normal
   approximation  P(wrong) = Phi(-Delta / sqrt((s0^2 + s1^2) / m)),
   where s_a^2 = p_a (1 - p_a), plus a half-weight for exact ties.
2. UCB, UCB-Tuned and ETC-greedy: mean regret from bandit_sim.py agrees with a
   separate, plain-Python single-trial implementation written directly from the
   definitions.
3. Regret bookkeeping: regret is non-decreasing in t and bounded by gap * t.
4. ETC exploration ends at the horizon, including odd horizons, and final regret
   equals gap * horizon * suboptimal-pull ratio.

Run:  python3 verify_implementation.py
"""
import math
import random
import numpy as np
import bandit_sim as sim

SCENARIOS = {"A": (0.80, 0.90), "B": (0.895, 0.90), "C": (0.89, 0.895)}
results = []


def report(name, ok, detail):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


# ---------------------------------------------------------------- 1. ETC lock-in
def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def etc_wrong_probability(p, m):
    gap = abs(p[1] - p[0])
    sd = math.sqrt((p[0] * (1 - p[0]) + p[1] * (1 - p[1])) / m)
    return norm_cdf(-gap / sd)


rng = np.random.default_rng(11)
for scen, m in (("A", 100), ("B", 10_000), ("C", 10_000), ("C", 100_000)):
    p = SCENARIOS[scen]
    n = 200_000
    means = sim.trial_arm_means(p, n, rng)
    _, _, wrong = sim.simulate_etc(means, m, 4 * m, np.array([4 * m]), rng)
    observed = wrong.mean()
    expected = etc_wrong_probability(p, m)
    tol = 4 * math.sqrt(expected * (1 - expected) / n) + 0.01  # sampling error + normal-approx error
    report(f"ETC wrong-commit rate, scenario {scen}, m={m}",
           abs(observed - expected) <= tol, f"simulated {observed:.4f}, normal approximation {expected:.4f}")


# ----------------------------------------------- 2. UCB / UCB-Tuned vs plain loops
def reference_run(p, horizon, tuned, seed):
    """One trial of UCB / UCB-Tuned written with plain Python lists, no vectorisation."""
    r = random.Random(seed)
    best = max(p)
    n_pulls = [0, 0]
    total = [0.0, 0.0]
    total_sq = [0.0, 0.0]
    regret = 0.0
    for t in range(1, horizon + 1):
        if t <= 2:
            arm = t - 1
        else:
            index = []
            for a in (0, 1):
                mean = total[a] / n_pulls[a]
                if tuned:
                    v = total_sq[a] / n_pulls[a] - mean ** 2 + math.sqrt(2 * math.log(t) / n_pulls[a])
                    bonus = math.sqrt(math.log(t) / n_pulls[a] * min(0.25, v))
                else:
                    bonus = math.sqrt(2 * math.log(t) / n_pulls[a])
                index.append(mean + bonus)
            arm = 1 if index[1] > index[0] else 0
        x = 1.0 if r.random() < p[arm] else 0.0
        n_pulls[arm] += 1
        total[arm] += x
        total_sq[arm] += x * x
        regret += best - p[arm]
    return regret


horizon, trials = 5_000, 300
for scen in ("A", "C"):
    p = SCENARIOS[scen]
    for tuned in (False, True):
        ref = np.array([reference_run(p, horizon, tuned, s) for s in range(trials)])
        means = sim.trial_arm_means(p, trials, np.random.default_rng(5))
        out, _ = sim.simulate_ucb(means, horizon, np.array([horizon]), np.random.default_rng(6), tuned=tuned)
        vec = out[0]
        se = math.sqrt(ref.var(ddof=1) / trials + vec.var(ddof=1) / trials)
        diff = abs(ref.mean() - vec.mean())
        report(f"{'UCB-Tuned' if tuned else 'UCB'} vs plain-loop reference, scenario {scen}, T={horizon}",
               diff <= 3.5 * se,
               f"reference {ref.mean():.2f}, simulation {vec.mean():.2f}, difference {diff:.2f} (3.5 s.e. = {3.5 * se:.2f})")

def reference_etc_greedy(p, m, horizon, seed):
    """One trial of ETC-greedy with plain Python lists: m pulls per arm, then greedy on running means."""
    r = random.Random(seed)
    best = max(p)
    n_pulls = [0, 0]
    total = [0.0, 0.0]
    regret = 0.0
    for t in range(1, horizon + 1):
        if t <= 2 * m:
            arm = (t - 1) % 2
        else:
            means = [total[a] / n_pulls[a] for a in (0, 1)]
            arm = 1 if means[1] > means[0] else 0
        x = 1.0 if r.random() < p[arm] else 0.0
        n_pulls[arm] += 1
        total[arm] += x
        regret += best - p[arm]
    return regret


horizon, trials, m = 4_000, 400, 100
for scen in ("A", "C"):
    p = SCENARIOS[scen]
    ref = np.array([reference_etc_greedy(p, m, horizon, s) for s in range(trials)])
    means = sim.trial_arm_means(p, trials, np.random.default_rng(8))
    out, _, _ = sim.simulate_etc_greedy(means, m, horizon, np.array([horizon]), np.random.default_rng(9))
    vec = out[0]
    se = math.sqrt(ref.var(ddof=1) / trials + vec.var(ddof=1) / trials)
    diff = abs(ref.mean() - vec.mean())
    report(f"ETC-greedy vs plain-loop reference, scenario {scen}, m={m}, T={horizon}",
           diff <= 3.5 * se,
           f"reference {ref.mean():.2f}, simulation {vec.mean():.2f}, difference {diff:.2f} (3.5 s.e. = {3.5 * se:.2f})")

# ------------------------------------------------------ 3. regret bookkeeping
marks = sim.checkpoints(20_000)
means = sim.trial_arm_means(SCENARIOS["A"], 50, np.random.default_rng(3))
for name, (out, *_rest) in {
    "UCB": sim.simulate_ucb(means, 20_000, marks, np.random.default_rng(4), tuned=False),
    "UCB-Tuned": sim.simulate_ucb(means, 20_000, marks, np.random.default_rng(4), tuned=True),
    "ETC m=100": sim.simulate_etc(means, 100, 20_000, marks, np.random.default_rng(4)),
    "ETC-greedy m=100": sim.simulate_etc_greedy(means, 100, 20_000, marks, np.random.default_rng(4)),
}.items():
    monotone = bool(np.all(np.diff(out, axis=0) >= -1e-9))
    bounded = bool(np.all(out <= 0.1 * marks[:, None] + 1e-9))
    report(f"regret bookkeeping, {name}", monotone and bounded, "non-decreasing in t and at most gap * t")

# ----------------------------------------- 4. exploration cutoffs and odd horizons
# Both arm orders make an omitted final arm-0 pull observable. These expectations
# follow directly from one alternating action per round, independent of rewards.
boundary_means = np.array([[0.80, 0.90], [0.90, 0.80]])
boundary_gap = boundary_means.max(axis=1)[:, None] - boundary_means
for name, simulate in (("ETC", sim.simulate_etc), ("ETC-greedy", sim.simulate_etc_greedy)):
    for horizon in (3, 4, 5, 6, 7):
        m = 3
        boundary_marks = np.arange(1, horizon + 1)
        out, ratio, flagged = simulate(boundary_means, m, horizon, boundary_marks, np.random.default_rng(12))
        implied_regret = boundary_gap.max(axis=1) * horizon * ratio
        ratio_matches = bool(np.allclose(out[-1], implied_regret, atol=1e-12, rtol=0))
        exploration_marks = boundary_marks[boundary_marks <= 2 * m]
        expected_exploration = (((exploration_marks + 1) // 2)[:, None] * boundary_gap[:, 0]
                                + (exploration_marks // 2)[:, None] * boundary_gap[:, 1])
        exploration_matches = bool(np.allclose(out[:len(exploration_marks)], expected_exploration,
                                              atol=1e-12, rtol=0))
        no_commit_before_exploitation = horizon > 2 * m or not bool(np.any(flagged))
        report(f"exploration cutoff, {name}, m={m}, T={horizon}",
               ratio_matches and exploration_matches and no_commit_before_exploitation,
               "alternating pulls through cutoff; regret = gap * T * suboptimal ratio")

# The documented quick mode has m=100,000 at T=100,000: exactly half the pulls
# are suboptimal, with no post-exploration action and regret 250 for gap 0.005.
quick_means = np.array([[0.89, 0.895], [0.895, 0.89]])
for name, simulate in (("ETC", sim.simulate_etc), ("ETC-greedy", sim.simulate_etc_greedy)):
    out, ratio, flagged = simulate(quick_means, 100_000, 100_000, np.array([100_000]),
                                  np.random.default_rng(13))
    ok = bool(np.allclose(out[-1], 250.0) and np.allclose(ratio, 0.5) and not np.any(flagged))
    report(f"quick-mode exploration cutoff, {name}", ok, "regret 250; suboptimal ratio 0.5; no commitment")

print()
print("ALL CHECKS PASSED" if all(results) else "SOME CHECKS FAILED")
raise SystemExit(0 if all(results) else 1)
