# Evaluation notes

The details behind the [README](../../README.md): the protocol, published results and their
protocols, the figures that did not fit there, and the status of every finding from the last
review. Every result quoted in the README is generated from `artifacts/` by
`python -m predmaint.report` and rechecked by `python -m predmaint.audit`.

## Protocol

**Sampling unit.** The engine. Every split is by engine, never by row, because consecutive cycles
of one engine are strongly dependent.

**Splits, per subset.**

1. **Replay holdout.** 10% of the training engines, stratified by life length, are held out for the
   dashboard replay. No network, scaler or calibration step sees them.
2. **Cross-fitting.** The remaining engines are split into five folds stratified by life length.
   For each fold, a model is trained on the other four.
3. **Inner validation.** Inside those four folds, 15% of the engines are held out to choose the
   input window (30, 40, 50 or 60 cycles, lowest RMSE with every engine weighted equally) and to stop
   training early.

**What each engine is used for.** An engine in fold k is never used to train, stop or choose the
window of fold k's model. It is used for training and inner validation of the other folds' models,
and for the out-of-fold prediction and calibration score of its own.

**Calibration.**

- **Intervals.** Each cross-fitted engine is truncated once, at a length drawn from the official
  test trajectories that are shorter than its life. Only sensor data is used, never labels. Its
  conformity score, from conformalized quantile regression, comes from the one fold model that
  never saw it. CV+ (Barber et al., 2021) combines these out-of-fold scores with each fold model's
  quantiles. Its guarantee is coverage of at least 1 − 2α = 80% for exchangeable engines; the
  nominal level is 90%.
- **Alert probability.** It is temperature-scaled, fitted out of fold on ten truncations per engine.
  Those ten points are dependent within an engine; with one scalar per fold, that matters less than
  it would for a richer model, but it is not one point per engine.
- **Alert threshold.** It is 0.5, fixed before any evaluation.

**The cap.** The target is RUL capped at 125 cycles, as usual for C-MAPSS. An interval whose upper
end reaches the cap is reported as open-ended ("at least the lower end"). Under that reading, the
capped target is inside the interval exactly when the actual remaining life is.

**The test sets.** NASA's official test trajectories stop before failure, and NASA gives the true
RUL at the last cycle. Training never reads those labels: `predmaint.evaluate` does, once per
subset, for all repeats and both models, and appends a "test viewed" entry to the
[experiment ledger](experiment-ledger.md). Version 2's development runs, which printed test scores,
are listed there as well.

**Repeats and comparisons.**

- **Repeats.** The whole pipeline ran three times, with different seeds for folds, inner splits,
  initialisation, LightGBM sampling and truncation draws. Tables give mean ± sd over repeats.
- **Model differences.** These are repeat-averaged and come with a 95% paired bootstrap interval
  over test engines: 2,000 draws, resampling engines with both models' predictions together.
- **Coverage intervals.** These are Wilson intervals over test engines at the mean rate.

**Three populations.**

| Name | What is scored | Used for |
|---|---|---|
| Test | One prediction per official test engine, at its last observed cycle, by the served five-model average | Accuracy, coverage, alert average precision |
| Out of fold | Every cycle of every cross-fitted engine, each predicted by the one fold model that never saw it | Whole-life coverage, alert timing, flapping, reliability |
| Replay | Every cycle of the held-out replay engines, by the served average | The dashboard and the hero figure |

## Published results

The RMSEs below are those listed in Table 4 of Chaoub et al. (2021), next to each study's
protocol.

- **Not checked.** Only Chaoub et al.'s own paper could be read in full; the other three are
  paywalled, so their protocol cells say "not checked" rather than guess.
- **Not comparable.** The studies differ in the RUL cap, the engines used for training, the number
  of runs, the input and, above all, where on each test trajectory the error is measured. The
  numbers therefore do not rank these methods against this project, and they are not drawn on the
  same axes as its results.

| Study | RUL cap | Training engines | Runs reported | Input | Test error measured at | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|---|---:|---:|---:|---:|
| This project | 125 | 90%, cross-fitted in 5 folds by engine; 10% held out for the replay | mean of 3 repeats of a 5-model average | last 30–60 cycles, chosen per fold on inner-validation engines | last cycle of each test engine | see the README | see the README | see the README | see the README |
| MLP-LSTM-MLP, Chaoub et al. (2021) | 130 | 75% by engine, 25% for validation; retraining on all engines not stated | mean of 10 runs | whole trajectory, variable length | every cycle of each test trajectory (their Eq. 4) | 13.26 | 12.49 | 13.11 | 13.97 |
| MS-DCNN, Li et al. (2020) | not checked | not checked | not checked | not checked | not checked | 11.44 | 19.35 | 11.67 | 22.22 |
| DA-CNN, Song et al. (2021) | not checked | not checked | not checked | not checked | not checked | 11.78 | 16.95 | 11.56 | 18.23 |
| HDNN, Al-Dulaimi et al. (2019) | not checked | not checked | not checked | not checked | not checked | 13.02 | 15.24 | 12.22 | 18.15 |

Notes on the table:
- Chaoub et al. report standard deviations over their 10 runs of 0.57, 0.28, 1.28 and 0.48
  (Table 3).
- HDNN's FD001 value is 13.017 in their Table 4, rounded here; they cite the ICASSP 2019 version of
  HDNN.
- Chaoub et al. call the Song et al. model DA-CNN; its title calls it a distributed-attention
  temporal convolutional network.

References:
- A. Chaoub, A. Voisin, C. Cerisara and B. Iung, "Learning representations with end-to-end models
  for improved remaining useful life prognostics", arXiv:2104.05049, 2021.
- H. Li, W. Zhao, Y. Zhang and E. Zio, "Remaining useful life prediction using multi-scale deep
  convolutional neural network", *Applied Soft Computing* 89, 2020,
  [doi:10.1016/j.asoc.2020.106113](https://doi.org/10.1016/j.asoc.2020.106113).
- Y. Song, S. Gao, Y. Li, L. Jia, Q. Li and F. Pang, "Distributed attention-based temporal
  convolutional network for remaining useful life prediction", *IEEE Internet of Things Journal*
  8(12), 2021, [doi:10.1109/JIOT.2020.3004452](https://doi.org/10.1109/JIOT.2020.3004452).
- A. Al-Dulaimi, S. Zabihi, A. Asif and A. Mohammadi, "Hybrid deep neural network model for
  remaining useful life estimation", *ICASSP*, 2019,
  [doi:10.1109/ICASSP.2019.8683763](https://doi.org/10.1109/ICASSP.2019.8683763).

## More figures

**Every test engine.** One prediction per engine at its last recorded cycle, with its interval.
Engines with true RUL above 125 sit on the plateau at the right. This is the population that the
test tables score.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/test-sets-dark.png">
  <img alt="Predicted against capped true RUL for every official test engine of the four subsets, with nominal 90% intervals; open-ended intervals run off the top." src="figures/test-sets-light.png">
</picture>

**Whole lives, out of fold.** These show how the prediction, the interval and the alert behave over
an engine's whole life, for engines the predicting network never saw.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/lives-dark.png">
  <img alt="Whole lives of four FD001 training engines, each predicted out of fold, with intervals and first alerts." src="figures/lives-light.png">
</picture>

### Attribution checks

`predmaint.attribution_checks` writes `artifacts/FD001/attribution-checks.json`. It runs these
checks:
- completeness;
- deletion and insertion curves against random order, the raw input deviation and a network with
  randomised weights (Adebayo et al., 2018);
- the effect of changing the baseline to each engine's own first cycles.

The README states the results.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/attribution-checks-dark.png">
  <img alt="Deletion and insertion curves for integrated gradients, input deviation, a randomised network and random order, and rank correlations between attribution maps under each change." src="figures/attribution-checks-light.png">
</picture>

## Status of the last review's findings

The review covered commit `3048740`, version 2. Each finding was rechecked against this branch.

| Finding | Status | Evidence |
|---|---|---|
| The public `master` branch still advertises the legacy project. | **Open.** Fixing it needs a merge and a push, which are outside this work. | `git show master:README.md` |
| Validation engines chose the model and also calibrated it. | **Fixed.** Window choice and early stopping use inner-validation engines inside each fold, and every calibration score comes from the fold model that never saw the engine. | `predmaint.audit` checks both rules on the committed predictions |
| Calibration pooled dependent windows, while testing uses one endpoint per engine. | **Fixed for the intervals**, which use one truncation point per engine shaped like the test points. **Partly open for the alert temperature**, which uses ten truncations per engine. | Protocol above |
| The prose contradicted itself about evaluating the test set once. | **Fixed.** Each test set was scored once by the final pipeline, and the earlier development exposure is stated. This revision also corrects a leftover sentence that called that exposure "once before". | The experiment ledger |
| Holdout prediction files existed locally but were not committed. | **Fixed.** Every prediction file that the figures and the audit read is tracked. The FD002 to FD004 networks and replay rows are not tracked, but their hashes are in each `manifest.json`. | `git ls-files predictive-maintenance/artifacts` |
| The figure command could not rebuild every figure from a clean clone. | **Fixed.** `python -m predmaint.figures` needs only tracked files; CI runs it from a clean checkout. | CI step "Rebuild the figures" |
| Recorded source hashes did not identify a commit. | **Fixed.** All four `metrics.json` files name commit `a1b69ff` with clean sources, and `source_sha256` recomputed from that commit's tree matches. | `provenance` in each `metrics.json` |
| The alerts figure clipped a caption. | **Fixed.** All figures were redrawn at README width, and their subtitles wrap to it. | `docs/figures/` |
| The ensemble was bolded even where LightGBM had the lower RMSE. | **Fixed.** Bold marks the better model in each column of each subset. | The results tables |

## Open issues that need separate work

These go beyond presentation. They need new evaluation design, a retraining, or a decision about
publishing.

- **The calibration population.** The truncation points match the test sets' trajectory lengths
  but not their distribution of remaining life. That produces over-coverage on the test sets and
  under-coverage in mid-life. Calibrating conditionally on predicted RUL is the natural next step,
  and it needs retraining and re-evaluation.
- **Development exposure.** The C-MAPSS test sets were seen during version 2's development. A clean
  final estimate would need a holdout that no development step touched. For C-MAPSS that has to be
  a held-out set of training engines; the official test sets cannot be unseen.
- **The window grid.** 60 cycles, the longest window, won most folds. Longer windows were not tried.
- **The alert threshold.** It is not tuned to any cost of missed or false alerts.
- **Reproducibility across devices.** GPU training is not bit-for-bit reproducible, so a rerun
  reproduces the protocol, not the exact numbers.
- **Publishing.** This needs four steps:
  - merging the branch into `master`;
  - updating the GitHub description;
  - rotating the version 1 ThingSpeak keys;
  - deciding whether to rewrite history. Commit `a1b69ff` also holds a local scratch path in one
    ledger line, which a later commit corrects.
