"""
Simulation core for two-armed Bernoulli bandits.

Algorithms (definitions from Wolf, 2025, Section 2):
  ETC        explore each arm m times, then commit to the arm with the larger
             empirical mean for the remaining steps (decision fixed after step 2m).
  ETC-greedy explore each arm m times, then at every step play the arm with the
             larger empirical mean; the estimates keep updating, so the played
             arm can change after step 2m. This is how the ETC code of Bandit
             Playground (the paper's framework) behaves.
  UCB        play the arm maximising  Q_hat_a + sqrt(2 ln t / T_a).
  UCB-Tuned  play the arm maximising  Q_hat_a + sqrt( (ln t / T_a) * min(1/4, V_a) ),
             with V_a = (mean of squared rewards) - Q_hat_a^2 + sqrt(2 ln t / T_a)
             (Auer, Cesa-Bianchi and Fischer, 2002).

All trials of one algorithm are simulated together as NumPy arrays of shape
(n_trials, 2). Each trial has its own random assignment of the better arm to
index 0 or 1, so tie-breaking by arm index favours neither arm.

The recorded quantity is cumulative pseudo-regret
  R_t = sum_{k<=t} (Q* - Q_{A_k}),
where Q* is the largest arm mean and A_k is the arm played at step k.
"""
import numpy as np

BLOCK = 2048  # number of time steps for which random numbers are drawn at once


def checkpoints(horizon, per_decade=12):
    """Log-spaced time steps 1 ... horizon at which regret is recorded."""
    n_points = int(per_decade * np.log10(horizon)) + 1
    pts = np.unique(np.round(np.logspace(0, np.log10(horizon), n_points)).astype(np.int64))
    return pts[(pts >= 1) & (pts <= horizon)]


def trial_arm_means(p, n_trials, rng):
    """Arm means per trial, shape (n_trials, 2), with the arm order shuffled per trial."""
    means = np.tile(np.asarray(p, dtype=float), (n_trials, 1))
    swap = rng.random(n_trials) < 0.5
    means[swap] = means[swap][:, ::-1]
    return means


def simulate_etc(means, m, horizon, marks, rng):
    """
    Explore-then-commit with m pulls of each arm.

    Exploration alternates between the arms, so the successes of arm a are
    Binomial(m, p_a) and the commit decision depends only on those counts.
    Returns (regret at each mark, shape (len(marks), n_trials); share of
    suboptimal pulls per trial; boolean array of trials that committed to the
    worse arm).
    """
    n = means.shape[0]
    rows = np.arange(n)
    gap = means.max(axis=1)[:, None] - means          # per-pull regret of each arm
    successes = rng.binomial(m, means)                # exploration outcome, (n, 2)
    commit = (successes[:, 1] > successes[:, 0]).astype(np.intp)
    ties = successes[:, 0] == successes[:, 1]
    commit[ties] = rng.integers(0, 2, ties.sum())     # equal means: pick an arm at random
    committed_gap = gap[rows, commit]

    regret = np.empty((len(marks), n))
    for i, t in enumerate(marks):
        if t <= 2 * m:                                # still exploring: arm 0, arm 1, arm 0, ...
            regret[i] = ((t + 1) // 2) * gap[:, 0] + (t // 2) * gap[:, 1]
        else:
            regret[i] = m * (gap[:, 0] + gap[:, 1]) + (t - 2 * m) * committed_gap
    explore_pulls = min(2 * m, horizon)
    subopt_pulls = (explore_pulls // 2) * (gap[:, 0] > 0) + (explore_pulls // 2) * (gap[:, 1] > 0)
    subopt_pulls = subopt_pulls + max(horizon - 2 * m, 0) * (committed_gap > 0)
    return regret, subopt_pulls / horizon, committed_gap > 0


def simulate_ucb(means, horizon, marks, rng, tuned=False):
    """
    UCB (tuned=False) or UCB-Tuned (tuned=True), all trials in parallel.

    Each arm is played once first; from round 3 on the arm with the larger
    index is played (ties go to arm 0). Returns (regret at each mark, share of
    suboptimal pulls per trial).
    """
    n = means.shape[0]
    rows = np.arange(n)
    gap = means.max(axis=1)[:, None] - means
    log_n = np.log(np.arange(1, horizon + 1))         # log_n[t] = ln(round t + 1)

    counts = np.ones((n, 2))
    sums = np.zeros((n, 2))
    sumsq = np.zeros((n, 2))
    for arm in (0, 1):
        reward = (rng.random(n) < means[:, arm]).astype(float)
        sums[:, arm] = reward
        sumsq[:, arm] = reward
    regret = gap.sum(axis=1)
    pulls_suboptimal = (gap > 0).astype(float).sum(axis=1)

    mark_slot = {int(t): i for i, t in enumerate(marks)}
    out = np.empty((len(marks), n))
    for t in (1, 2):
        if t in mark_slot:
            out[mark_slot[t]] = gap[:, :t].sum(axis=1)

    t = 2
    while t < horizon:
        stop = min(t + BLOCK, horizon)
        uniforms = rng.random((stop - t, n))
        for j in range(stop - t):
            ln_t = log_n[t]
            mean = sums / counts
            if tuned:
                v = sumsq / counts - mean * mean + np.sqrt(2.0 * ln_t / counts)
                index = mean + np.sqrt(ln_t / counts * np.minimum(0.25, v))
            else:
                index = mean + np.sqrt(2.0 * ln_t / counts)
            arm = (index[:, 1] > index[:, 0]).astype(np.intp)
            reward = (uniforms[j] < means[rows, arm]).astype(float)
            counts[rows, arm] += 1.0
            sums[rows, arm] += reward
            sumsq[rows, arm] += reward
            step_gap = gap[rows, arm]
            regret = regret + step_gap
            pulls_suboptimal = pulls_suboptimal + (step_gap > 0)
            t += 1
            slot = mark_slot.get(t)
            if slot is not None:
                out[slot] = regret
    return out, pulls_suboptimal / horizon


def simulate_etc_greedy(means, m, horizon, marks, rng):
    """
    Explore m pulls of each arm (alternating), then play the arm with the larger
    running empirical mean at every step; estimates keep updating.

    Returns (regret at each mark, share of suboptimal pulls per trial,
    boolean array of trials whose final regret exceeds the exploration cost
    by more than 100 * gap, i.e. trials that spent a long period on the worse arm).
    """
    n = means.shape[0]
    rows = np.arange(n)
    gap = means.max(axis=1)[:, None] - means
    counts = np.full((n, 2), float(m))
    sums = rng.binomial(m, means).astype(float)       # exploration outcome, (n, 2)
    explore_regret = m * gap.sum(axis=1)
    regret = explore_regret.copy()
    pulls_suboptimal = np.full(n, float(m)) * ((gap[:, 0] > 0).astype(float) + (gap[:, 1] > 0))

    mark_slot = {int(t): i for i, t in enumerate(marks)}
    out = np.empty((len(marks), n))
    for i, t in enumerate(marks):
        if t <= 2 * m:                                # exploration: arm 0, arm 1, arm 0, ...
            out[i] = ((t + 1) // 2) * gap[:, 0] + (t // 2) * gap[:, 1]

    t = 2 * m
    while t < horizon:
        stop = min(t + BLOCK, horizon)
        uniforms = rng.random((stop - t, n))
        for j in range(stop - t):
            mean = sums / counts
            arm = (mean[:, 1] > mean[:, 0]).astype(np.intp)
            reward = (uniforms[j] < means[rows, arm]).astype(float)
            counts[rows, arm] += 1.0
            sums[rows, arm] += reward
            step_gap = gap[rows, arm]
            regret = regret + step_gap
            pulls_suboptimal = pulls_suboptimal + (step_gap > 0)
            t += 1
            slot = mark_slot.get(t)
            if slot is not None:
                out[slot] = regret
    long_on_worse_arm = (regret - explore_regret) > 100 * gap.max(axis=1)
    return out, pulls_suboptimal / horizon, long_on_worse_arm
