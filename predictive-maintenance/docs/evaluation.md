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
and for the out-of-fold prediction and calibration score of its own. So no calibration score comes
from a model that used that engine for training, early stopping or window choice. One step is
shared: the feature scaling and the operating-regime clusters are fitted once on the sensor
readings, never the labels, of all cross-fitted engines, so every fold model's preprocessing has
seen the unlabelled readings of the engines it is calibrated on.

**Calibration.**

- **Intervals.** Each cross-fitted engine is truncated once, at a length drawn from the official
  test trajectories that are shorter than its life. Only the trajectory lengths are used, never
  the test labels. Its conformity score, from conformalized quantile regression, comes from the
  one fold model that never saw it. CV+ (Barber et al., 2021) combines each fold model's quantiles
  for a new engine with the scores of the engines that model did not see; `predmaint.audit`
  rebuilds every committed test interval this way from `test-members.csv` and
  `calibration-points.csv`. The nominal level is 90%;
  [what the guarantee covers](#what-the-conformal-guarantee-covers) is below.
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

**Repeats and comparisons.** Each source of uncertainty is measured separately, and none of the
intervals below covers them all.

- **Repeats.** The whole pipeline ran three times, with different seeds for folds, inner splits,
  initialisation, LightGBM sampling and truncation draws. Tables give the mean ± sample sd of the
  three per-repeat statistics. That ± is training-seed spread on fixed test engines, not the
  precision of the estimate: for RMSE it is 0.1 to 0.3 cycles, while the engine bootstrap below is
  several times wider.
- **Model differences.** These are repeat-averaged and come with a 95% paired bootstrap interval
  over test engines: 2,000 draws (seed 7), each resampling engines once and applying the same draw
  to both models and all three repeats. It measures which engines happen to be in the test set,
  for fixed trained models. It does not include training-seed variation or the sampling of training
  engines, and the 16 differences (4 subsets × 4 metrics) are not corrected for multiple
  comparisons.
- **Coverage intervals.** These are 95% Wilson intervals with n the number of test engines,
  computed at the mean rate over repeats. They treat engines as independent and cover engine
  sampling only, not seed variation or the shared calibration set.

**Three populations.**

| Name | What is scored | Used for |
|---|---|---|
| Test | One prediction per official test engine, at its last observed cycle, by each repeat's five-fold ensemble: the mean of the five fold models, with the CV+ interval over them | Accuracy, coverage, width, alert average precision |
| Out of fold | Every cycle of every cross-fitted engine, by the single fold model that never saw it, with that model's own conformal margin, calibrated at truncation points and applied to every cycle | Whole-life coverage, alert timing, switch-offs, reliability |
| Replay | Every cycle of the held-out replay engines, by the served model: the first repeat's LSTM ensemble | The dashboard, the hero figure and the attribution example |

Out-of-fold probabilities come from single fold models, so the out-of-fold alert statistics and
ECE describe single networks, not the averaged probability the API serves. The replay engines,
which the served ensemble never saw, give its whole-life behaviour on 10 to 26 engines per subset.

### What the conformal guarantee covers

<!-- guarantee:start -->
5-fold CV+ (Barber et al., 2021, Theorem 4) guarantees marginal coverage of at least 1 − 2α − min{2(1 − 1/K)/(n/K + 1), (1 − K/n)/(K + 1)} for n calibration engines and K folds. At the nominal 90% (α = 0.1) and K = 5:

| Subset | Calibration engines | Worst-case CV+ coverage | Calibration points with 125 or more cycles left | Test engines with 125 or more cycles left |
|---|---:|---:|---:|---:|
| FD001 | 90 | 71.6% | 27% | 11% |
| FD002 | 234 | 76.7% | 30% | 22% |
| FD003 | 90 | 71.6% | 40% | 15% |
| FD004 | 224 | 76.5% | 42% | 27% |

The 80% often quoted is the jackknife+ bound, K = n. The guarantee needs each new pair of engine and truncation point to be exchangeable with the calibration pairs, one score per engine, folds fixed independently of the data and training that treats every engine alike. The test sets do not meet that: their engines are less often far from failure than the calibration points (table), the folds are stratified by life length, and the feature scaling and operating-regime clusters were fitted on the sensor readings of every calibration engine. The guarantee is also marginal, so it says nothing about coverage at a given remaining life. The evidence for these intervals is their measured coverage, not the bound.
<!-- guarantee:end -->

### Summaries defined after the test labels were read

The metrics in `metrics.json` were defined in the code of commit `a1b69ff` before that code read
the test labels: RMSE, the PHM08 score, RMSE against the uncapped truth, coverage, the open-ended
share, the mean width of each model's closed intervals, alert average precision, ECE and the
out-of-fold alert statistics. The summaries below were defined afterwards, while writing the
documents. They are descriptive, use no new model output, and are labelled where they appear:

- interval width with open-ended intervals counted up to the cap, and its bootstrap intervals
  (commit `0ecb0ce`);
- the failure-case gallery's error slices (`adae7d2`);
- coverage by true remaining life, pooled coverage, and the RMSE of single fold models
  (`958062a` and later);
- width on the engines where both models' intervals are closed, and switch-offs in the last 30
  cycles (this revision).

## Full results

The README's compact table summarises these. Generated by `python -m predmaint.report` and checked
by `python -m predmaint.audit`.

<!-- results:start -->
**Official test sets.** Each repeat of each model is evaluated as its five-fold ensemble: the point prediction is the mean of the five fold models and the interval is the CV+ interval over them. The API and dashboard serve the first repeat's LSTM. Mean ± sd over 3 repeats of the whole pipeline on the same test engines, so the ± is training-seed spread, not the precision of the estimate; bold marks the better model in each subset. The coverage interval is a 95% Wilson interval over test engines at the mean rate, which treats engines as independent and covers engine sampling only. Width counts an open-ended interval up to the 125-cycle cap, so both models are measured on the same engines; that summary was defined after the test labels were read.

| Subset | Model | RMSE | PHM08 score | RMSE vs actual RUL | Inside the nominal 90% interval (95% CI) | Open-ended intervals | Mean interval width | Alert AP | Alert ECE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FD001 (100 engines) | LSTM | 12.94 ± 0.11 | 306 ± 17 | 14.28 ± 0.14 | 95.3 ± 0.6% (89–98%) | 45 ± 2% | 41.5 ± 1.4 cycles | 0.984 ± 0.004 | 0.045 ± 0.007 |
| FD001 (100 engines) | LightGBM | **12.11 ± 0.24** | **244 ± 3** | **13.36 ± 0.21** | 95.0 ± 1.0% (89–98%) | 39 ± 3% | 46.1 ± 2.7 cycles | **0.984 ± 0.002** | 0.044 ± 0.003 |
| FD002 (259 engines) | LSTM | 12.53 ± 0.11 | 643 ± 43 | 25.27 ± 0.37 | 96.4 ± 0.2% (94–98%) | 43 ± 2% | 39.7 ± 0.8 cycles | 0.996 ± 0.002 | 0.029 ± 0.000 |
| FD002 (259 engines) | LightGBM | **11.47 ± 0.10** | **571 ± 13** | **24.04 ± 0.08** | 95.6 ± 1.2% (93–98%) | 40 ± 4% | 41.6 ± 2.5 cycles | **0.999 ± 0.000** | 0.021 ± 0.002 |
| FD003 (100 engines) | LSTM | 13.22 ± 0.26 | 610 ± 55 | 14.90 ± 0.15 | 94.7 ± 3.1% (89–98%) | 42 ± 1% | 37.6 ± 2.9 cycles | **0.987 ± 0.001** | 0.031 ± 0.010 |
| FD003 (100 engines) | LightGBM | **12.03 ± 0.15** | **281 ± 12** | **13.63 ± 0.16** | 95.0 ± 3.0% (89–98%) | 57 ± 15% | 58.0 ± 11.3 cycles | 0.986 ± 0.006 | 0.031 ± 0.007 |
| FD004 (248 engines) | LSTM | **12.82 ± 0.29** | **737 ± 48** | 25.75 ± 0.06 | 95.8 ± 1.3% (93–98%) | 50 ± 0% | 43.1 ± 1.8 cycles | **0.985 ± 0.002** | 0.025 ± 0.004 |
| FD004 (248 engines) | LightGBM | 13.11 ± 0.15 | 905 ± 34 | **25.41 ± 0.11** | 94.2 ± 0.2% (91–97%) | 62 ± 5% | 56.0 ± 3.5 cycles | 0.984 ± 0.002 | 0.012 ± 0.004 |

**LSTM minus LightGBM on the same test engines.** Repeat-averaged difference, with a 95% interval from a paired bootstrap over test engines (2,000 draws, each applied to both models and every repeat). An interval that excludes 0 means the difference is larger than the noise from which engines are in the test set. It does not include training-seed variation, and the 16 intervals are not corrected for multiple comparisons.

| Subset | RMSE, LSTM minus LightGBM | PHM08 score, LSTM minus LightGBM | Coverage, LSTM minus LightGBM (points) | Interval width, LSTM minus LightGBM |
|---|---:|---:|---:|---:|
| FD001 | +0.83 (-0.70 to +2.39) | +62 (-22 to +180) | +0.3 (-2.3 to +3.0) | -4.6 (-7.7 to -1.8) |
| FD002 | +1.06 (+0.26 to +1.93) | +72 (-42 to +190) | +0.8 (-1.4 to +3.1) | -1.9 (-3.4 to -0.5) |
| FD003 | +1.19 (-0.50 to +3.25) | +329 (-53 to +1048) | -0.3 (-4.0 to +3.7) | -20.4 (-26.5 to -14.9) |
| FD004 | -0.30 (-1.04 to +0.46) | -168 (-438 to +14) | +1.6 (-0.9 to +4.2) | -12.9 (-16.0 to -10.1) |

**The alert over whole lives, out of fold.** Every cycle of every cross-fitted training engine, each predicted by the single fold model that never saw it, not by the five-fold ensemble. Engine counts are means over repeats, ± sd; the ECE interval is the repeat-averaged 95% bootstrap interval over engines.

| Subset | Model | Engines | First alert, median cycles before failure | Missed | First alert with more than 30 cycles left | False-alert episodes per engine | Switch-offs before failure per engine | ECE over all cycles (95% CI) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| FD001 | LSTM | 90 | 30 ± 1 | 0.0 ± 0.0 | 42.7 ± 3.2 | 0.48 ± 0.04 | 0.02 ± 0.01 | 0.008 ± 0.002 (0.006–0.013) |
| FD001 | LightGBM | 90 | 30 ± 0 | 0.0 ± 0.0 | 42.7 ± 1.5 | 0.61 ± 0.07 | 0.39 ± 0.06 | 0.005 ± 0.003 (0.004–0.009) |
| FD002 | LSTM | 234 | 30 ± 2 | 0.0 ± 0.0 | 111.3 ± 18.2 | 0.52 ± 0.10 | 0.15 ± 0.03 | 0.009 ± 0.003 (0.006–0.012) |
| FD002 | LightGBM | 234 | 30 ± 0 | 0.0 ± 0.0 | 110.3 ± 2.5 | 0.62 ± 0.04 | 0.36 ± 0.03 | 0.005 ± 0.001 (0.004–0.007) |
| FD003 | LSTM | 90 | 30 ± 1 | 0.0 ± 0.0 | 40.7 ± 6.7 | 0.47 ± 0.08 | 0.11 ± 0.02 | 0.008 ± 0.002 (0.005–0.012) |
| FD003 | LightGBM | 90 | 30 ± 0 | 0.0 ± 0.0 | 43.7 ± 1.5 | 0.61 ± 0.03 | 0.33 ± 0.06 | 0.004 ± 0.001 (0.003–0.007) |
| FD004 | LSTM | 224 | 31 ± 1 | 1.7 ± 1.5 | 110.0 ± 8.2 | 0.53 ± 0.05 | 0.12 ± 0.02 | 0.005 ± 0.001 (0.003–0.010) |
| FD004 | LightGBM | 224 | 30 ± 0 | 0.0 ± 0.0 | 105.7 ± 2.3 | 0.65 ± 0.01 | 0.49 ± 0.01 | 0.004 ± 0.001 (0.003–0.008) |
<!-- results:end -->

### Findings in detail

<!-- summary:start -->
- **Accuracy.** LightGBM's test RMSE is lower on 3 of 4 subsets; the paired difference (LSTM minus LightGBM) is significant only on FD002, +1.06 cycles (95% CI +0.26 to +1.93). Averaging five fold models lowers RMSE by 0.5 to 1.1 cycles against the mean single fold model, for both families.
- **Uncertainty.** Both models cover 94 to 96% of test engines with a nominal 90% interval, so the intervals are conservative on the test sets. Coverage is not uniform: by true remaining life it falls to 91% for the LSTM (76–100 cycles left) and 81% for LightGBM (51–75 cycles left). Out of fold, over whole lives, it is 88 to 93%, below nominal for LightGBM on FD001 and FD002 and the LSTM on FD003. The LSTM's intervals are narrower on 4 of 4 subsets, by 5 to 35%, significantly on 4.
- **Alerting.** Both alerts rank test engines by risk almost perfectly: average precision 0.984 to 0.999, where a random score would get the base rate of 20 to 25%. Ranking says nothing about calibration, which is measured separately. Over whole lives the first alert comes a median of 30 to 31 cycles before failure; the LSTM's alert switches off again 0.02 to 0.15 times per engine, LightGBM's 0.33 to 0.49 times.
<!-- summary:end -->

### Alert ranking and calibration

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/alert-calibration-dark.png">
  <img alt="Average precision of the alert probability on each test set against its base rate, and reliability diagrams over whole lives out of fold, with the number of cycles in each probability bin." src="figures/alert-calibration-light.png">
</picture>

- **Average precision** is computed on the test sets from the probability of failure within 30
  cycles, with an engine positive when it fails within 30 cycles of its last observed cycle. It
  measures how well the probability ranks engines, not whether it is right; a random score would
  get the base rate.
- **Reliability** is out of fold, over every cycle of every cross-fitted engine, for the served
  repeat. Most cycles have a probability below 0.1, and consecutive cycles of one engine are
  dependent, so the ECE intervals in the tables resample whole engines.

## Attribution checks

`predmaint.attribution_checks` writes `artifacts/FD001/attribution-checks.json`. It tests the
integrated-gradients maps of the served model on held-out replay engines:
- completeness;
- deletion and insertion curves against random order, the raw input deviation and a network with
  randomised weights (Adebayo et al., 2018);
- the effect of changing the baseline to each engine's own first cycles.

<!-- attribution:start -->
On 40 windows from 10 FD001 replay engines (at 100, 60, 30, 10 cycles before failure), with the served five-network average:

- **Completeness holds numerically.** Integrated gradients satisfy it by construction, so this checks the numerical integration: the attributions add up to the change in prediction to within 0.04 cycles (median), against a median change of 69 cycles.
- **In the median window they find readings the output depends on.** Removing readings in attribution order moves the prediction to the baseline faster than random order: median deletion area 0.097 against 0.557 (lower is better). It is not better in every window: it beats random order in 32 of 40, and in the worst tenth of windows its deletion area exceeds 1.016.
- **Much of that is input deviation.** Ranking readings simply by how far they are from the baseline does about as well: median deletion area 0.132, and a worst tenth above only 0.267. Attributions beat it in 26 of 40 windows on deletion and in 18 of 40 on insertion.
- **The trained weights matter, moderately.** A randomly re-initialised network gives maps that rank-correlate 0.55 with the trained ones and delete less well (0.190).
- **The baseline matters.** Using each engine's own first cycles as the baseline instead of the regime average changes the maps (rank correlation 0.54; the top three sensors keep 2 of 3 in the median window).
- **Scope.** 40 windows from 10 engines of one subset, for the served network's point prediction, with one seed and no confidence intervals. There is no ground truth for which readings matter.
<!-- attribution:end -->

In the median window the attribution order beats random order and a randomised network's order,
and it beats plain input deviation on deletion but not on insertion. Read the maps as input
deviation weighted by the model: what the prediction depends on, not a diagnosis of the engine.
The limits of these checks:

- completeness holds by construction, so it only checks the numerical integration;
- the deletion and insertion curves splice baseline readings into real windows, which produces
  inputs unlike any the model was trained on;
- the windows come from the replay engines of one subset, for one network seed, with no
  confidence intervals, and the medians hide a poor tail;
- correlated sensors share credit, and the maps change with the baseline;
- nothing here is a ground truth for which readings matter, or shows that a sensor causes the
  predicted wear;
- `attribution-checks.json` records neither a commit nor a source hash, and it is not in the
  manifest, because the checks ran after evaluation.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/attribution-checks-dark.png">
  <img alt="Deletion and insertion curves for integrated gradients, input deviation, a randomised network and random order, and rank correlations between attribution maps under each change." src="figures/attribution-checks-light.png">
</picture>

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
| This project | 125 | 90%, cross-fitted in 5 folds by engine; 10% held out for the replay | mean of 3 repeats of a 5-model average | last 30–60 cycles, chosen per fold on inner-validation engines | last cycle of each test engine | see [Full results](#full-results) | see [Full results](#full-results) | see [Full results](#full-results) | see [Full results](#full-results) |
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

## Corrections from the pre-release methodology review

Before release, the code and artifacts were reviewed again against every claim in these documents.
The review confirmed the engine separation, the CV+ implementation and the provenance of the
training and evaluation code. It also found wording that went beyond the evidence, corrected here:

- **The CV+ guarantee** was stated as 1 − 2α = 80%, which is the jackknife+ bound. The five-fold
  bound is lower, and the test sets do not meet its assumptions ([above](#what-the-conformal-guarantee-covers)).
- **"First alert more than 30 cycles early"** counted first alerts with more than 30 cycles left,
  that is, any first alert before the horizon. The tables and figures now say so.
- **"Served five-model average"** described the test evaluation inexactly: each repeat is evaluated
  as a five-fold ensemble with a CV+ interval, and only the first repeat's LSTM is served.
- **The attribution checks** were summarised by their medians as showing faithful attributions.
  The tail is poor, and the attributions do not beat plain input deviation on insertion.
- **The shared preprocessing**, fitted on every cross-fitted engine's readings, and the
  **summaries defined after the test labels were read** were not disclosed. Both are now listed
  above.
- **The ± in the tables** was not said to be training-seed spread rather than precision, and the
  bootstrap intervals were not said to be uncorrected for multiple comparisons.

## Open issues that need separate work

These go beyond presentation. They need new evaluation design, a retraining, or a decision about
publishing.

- **The calibration population.** The truncation points match the test sets' trajectory lengths
  but not their distribution of remaining life. That produces over-coverage on the test sets and
  under-coverage in mid-life. Calibrating conditionally on predicted RUL is the natural next step,
  and it needs retraining and re-evaluation. A guarantee that applies to the test sets would also
  need a scaler fitted per fold and folds fixed independently of life length.
- **The served alert probability.** Its temperature was fitted for single fold models out of fold.
  Calibrating the averaged probability needs engines that none of the five fold models saw, which
  means retraining.
- **Uncertainty from training.** The difference intervals cover test-engine sampling only. Seed
  variation would need more repeats and a two-level bootstrap.
- **Alert definitions.** Splitting switch-offs at the horizon, or a stricter definition of a missed
  engine, would be new definitions; the last-30-cycle count above is one, and it is post hoc.
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
