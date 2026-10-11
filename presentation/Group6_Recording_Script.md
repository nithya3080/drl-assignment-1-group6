# Group 6 - Recording script (target 6:30, hard limit 7:00)

SAY is the text to speak, word for word. ON SCREEN tells you where to point in the PPT; do not read it aloud. The same SAY text must appear in the final PPT speaker notes.

## Before anyone records

1. Open the final PPT in slideshow mode. Record the PPT and your voice; all source, code and result proof is embedded in the slides.
2. Remain in the PPT throughout. No browser, terminal, HTML page, worksheet or live experiment demonstration is needed.
3. Nithyasree presents slides 1-4; Arjun 5-7; Saibhargav 8-10; Ashok 11-16. Each member explains their completed contribution using the visible evidence.
4. Prepare a quiet recording and readable slideshow. Read only SAY; move the pointer to each relevant item while explaining it.
5. Aim for 6:30 overall and never exceed 7:00. Rehearse and time the joined video. Decimal numbers take longer to say than their whitespace word counts suggest.
6. After recording, upload the video to Drive, verify viewing access, add its actual URL to slide 16 and rebuild the submission ZIP.

## Timing

| Speaker | Slides | Spoken words | Recording budget |
|---|---|---|---|
| Nithyasree | 1-4 | 211 | 1:34 |
| Arjun | 5-7 | 182 | 1:23 |
| Saibhargav | 8-10 | 201 | 1:34 |
| Ashok | 11-16 | 248 | 1:59 |
| **Total** | **1-16** | **842** | **6:30** |

Counts include only SAY text, using whitespace words. At 145 words/minute this is approximately 5.81 minutes of speech; the 6:30 budget includes careful number pronunciation, pointing and short pauses. The per-slide budgets sum to exactly 390 seconds.

## Slide 1 - Nithyasree (target 20 seconds)
**ON SCREEN:** Show slide 1 with Group 6, all four full names and BITS IDs.

**SAY:** Hello, we are Group 6. This presentation analyses "A Framework for Fair Evaluation of Variance-Aware Bandit Algorithms" by Elise Wolf. The paper builds a reproducible framework for comparing eight bandit algorithms. We explain its methods and results, then our own partial reproduction and critical analysis.

## Slide 2 - Nithyasree (target 25 seconds)
**ON SCREEN:** Stay on slide 2. Point at the three gap cards, the Objective box, then RQ1 and RQ2.

**SAY:** A bandit learner must balance exploring uncertain options against exploiting the best one it knows. Earlier comparisons use narrow algorithm sets or fixed scenarios, and results can be distorted by different initial conditions and tuning budgets. The paper's framework, Bandit Playground, asks: how can performance differences be reliably detected, and when do variance-aware algorithms outperform classical ones?

## Slide 3 - Nithyasree (target 25 seconds)
**ON SCREEN:** Stay on slide 3. Point at per-run pseudo-regret, gap times suboptimal pulls and the worked example, then the verified paper-section references.

**SAY:** Performance is measured by pseudo-regret: at every step, the best arm's mean minus the chosen arm's mean, summed up. With two arms, that equals the gap times the number of worse-arm pulls. In Scenario A the gap is 0.1, so a hundred worse-arm pulls cost ten. Lower is better. I verified the cited paper sections.

## Slide 4 - Nithyasree (target 24 seconds)
**ON SCREEN:** Stay on slide 4. Point down the algorithm families, including UCB-Improved; indicate PAC-UCB's code qualification and the three shaded reproduced methods.

**SAY:** The table covers all eight methods. ETC and epsilon-greedy control exploration directly. UCB adds a confidence bonus. UCB-V and UCB-Tuned adapt to variance, and EUCBV also eliminates arms. UCB-Improved uses phased arm elimination. The paper labels PAC-UCB non-variance-aware, although its linked code uses empirical variance. We reproduce the shaded three: ETC, UCB and UCB-Tuned.

## Slide 5 - Arjun (target 36 seconds)
**ON SCREEN:** Stay on slide 5. Point at each RULE box, then the embedded UCB-Tuned code strip: the line beginning v = and np.minimum(0.25, v). Explain the formula-to-code mapping.

**SAY:** Strict ETC samples each arm m times, then fixes its choice. Exploration costs m times the gap, and a wrong commitment lasts to the end, so m must be chosen well. UCB adds a confidence bonus to each arm's average, large for rarely pulled arms. UCB-Tuned uses the arm's estimated variance plus an uncertainty correction, capped at one quarter. The code strip shows exactly that: the variance term, then the capped bonus. I checked these formulas against the code.

## Slide 6 - Arjun (target 22 seconds)
**ON SCREEN:** Stay on slide 6. Point at Environment, Bias controls and Metrics. Indicate the regret VaR bullet and the partial-reproduction footer.

**SAY:** The paper uses Bernoulli rewards, one million steps and a hundred trials. Common initialisation, fixed seeds, all arm permutations and one pull per step reduce procedural bias. Value-at-Risk is an upper regret percentile, not an absolute worst case. Our reproduction randomises arm order and omits the risk and reward-variance analyses.

## Slide 7 - Arjun (target 25 seconds)
**ON SCREEN:** Stay on slide 7. Point at the gaps, then B/C variances and the note explaining their dependence on the Bernoulli means.

**SAY:** Scenario A compares 0.8 and 0.9, a gap of 0.1. B compares 0.895 with 0.9, and C compares 0.89 with 0.895, so both have a gap of 0.005. C has slightly higher variance than B. But Bernoulli variance depends on the mean, so these scenarios do not isolate variance independently of the means.

## Slide 8 - Saibhargav (target 26 seconds)
**ON SCREEN:** Stay on slide 8. Point at the published ETC and UCB-Tuned values, their highlighted exploration settings and the Also footnotes in A/C.

**SAY:** These are the paper's reported mean regrets. The best ETC settings lead: 10.10, 150.66 and 170.53. That lead needs a suitable exploration budget: a poor choice costs 1,000 in A and 2,150 in C. Among default configurations, UCB-Tuned leads: 30.67, 212.21 and 226.09. These are results for the tested settings, not universal superiority.

## Slide 9 - Saibhargav (target 29 seconds)
**ON SCREEN:** Stay on slide 9. Point at UCB versus UCB-Tuned in A/B/C, then the paper-based absolute savings and ratios in the right-hand card.

**SAY:** UCB-Tuned improves on UCB in all three scenarios. It saves about 208 regret in A, and 915 and 946 in B and C. In ratio terms UCB's regret is 7.8 times larger in A and about 5 times in B and C, so the relative gain is greatest in A. The result is conditional; it does not show that variance-awareness only helps with tiny gaps.

## Slide 10 - Saibhargav (target 39 seconds)
**ON SCREEN:** Stay on slide 10. Point to the full experiment command, one-million-step horizon, 100 trials and base seed 42, then the 26-check summary. Point at B/UCB-Tuned in the saved full-horizon table: paper 212.21, ours 201.06 plus/minus 8.47 SE. No browser, terminal, quick-mode output or live execution is needed.

**SAY:** My contribution covers experiment execution and verification. This is our partial reproduction: three methods at the full horizon, a million steps and a hundred trials with base seed forty-two, and ETC in two forms. The code is in bandit_sim.py, run by run_experiment.py. verify_implementation.py runs twenty-six checks, including plain-loop comparisons and an approximate ETC probability check, and all pass. In results/comparison.csv, Scenario B's UCB-Tuned mean is 201.06 against the paper's 212.21. All six UCB and UCB-Tuned means are within six percent of the paper.

## Slide 11 - Ashok (target 24 seconds)
**ON SCREEN:** Stay on slide 11. Point to the decimal mean labels and standard-error bars: ETC-greedy B 189.81 and C 350.78; strict ETC B 638.00 and C 491.00. These chart values come from 100 trials, as do the six UCB/UCB-Tuned comparisons.

**SAY:** This plot compares published means with our hundred-trial results; error bars show one standard error. UCB and UCB-Tuned stay within six percent of the paper. ETC-greedy reaches about 190 in B and 351 in C, with large variability, and a strict commit gives 638 and 491 over these hundred trials.

## Slide 12 - Ashok (target 35 seconds)
**ON SCREEN:** Stay on slide 12. Point at the embedded authors' ETC.py reselection/update excerpt, physical source lines 45-57. For B/C at m = 10000, point to the larger-trial card: strict 654.61 and 676.32 from 100000 trials; greedy 256.14 and 197.53 from 1000 trials, with their SEs. The plotted curves separately use 100 trials; indicate the dashed strict curves above UCB-Tuned in B/C.

**SAY:** The paper describes a fixed commitment, but its linked code recomputes the best empirical mean every round, as this excerpt from ETC.py shows. Our hundred-thousand-trial estimates for strict ETC are 654.61 and 676.32 in B and C, against 201.06 and 227.82 for UCB-Tuned. These are Monte Carlo estimates. At these budgets strict ETC costs more than UCB-Tuned, while greedy ETC is noisy, so no universal winner is established.

## Slide 13 - Ashok (target 16 seconds)
**ON SCREEN:** Stay on slide 13. Point at limitations, our reproduction scope and future research directions.

**SAY:** Limitations: stationary binary rewards, only two arms and unequal tuning budgets; our reproduction covers three methods and selected regret metrics. Future work: more arms, delayed and contextual rewards, non-stationarity, and runtime and sample complexity.

## Slide 14 - Ashok (target 31 seconds)
**ON SCREEN:** Stay on slide 14. Point to calculations explicitly based on the PAPER'S means: savings 207.50, 914.96 and 946.48; ratios 7.77, 5.31 and 5.19. Indicate the notebook source fragments (dof=999999 and .var()), 100 run totals and usual df=99 under normality, then the conclusion.

**SAY:** In the paper's notebook, the variance test uses the variance of a hundred run totals, but uses 999,999 degrees of freedom. Under normality it should be 99, so those p-values cannot validate regret differences. From the paper's means, I checked savings of 207.50, 914.96 and 946.48 regret, with ratios 7.77, 5.31 and 5.19. Overall, UCB-Tuned improves on UCB everywhere tested, while ETC comparisons depend on tuning, definition and sampling uncertainty.

## Slide 15 - Ashok (target 6 seconds)
**ON SCREEN:** Stay on slide 15. Hold the contribution table briefly. It links each member's completed role to evidence already embedded in this PPT.

**SAY:** This table lists each member's contribution and where it appears in this recording.

## Slide 16 - Ashok (target 7 seconds)
**ON SCREEN:** Stay on slide 16. Hold the reference links briefly. After recording, add the real Google Drive video URL to this final slide and verify viewing access before submission.

**SAY:** Thank you. The paper, the authors' code and our repository are listed here.
