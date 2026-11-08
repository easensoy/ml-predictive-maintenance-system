# Model card: predmaint remaining-useful-life models (v2.1)

## Model details

- **What it predicts.** From one turbofan engine's recent sensor history:
  - the remaining useful life (RUL) in cycles;
  - a nominal 90% interval around it;
  - the probability that the engine fails within the next 30 cycles.

  Each prediction also comes with an integrated-gradients attribution over the readings in the
  model's input window.
- **Architecture.** Five 2-layer LSTM networks with attention pooling, one per cross-fitting fold.
  - Network k was trained without fold k's engines, on the input window (30–60 cycles) that won
    on inner-validation engines.
  - Each network predicts the RUL point (MSE), its 5th and 95th percentiles (pinball loss) and the
    alert logit (cross-entropy). The served prediction averages the five.
  - The interval is a CV+ interval on conformalized-quantile-regression scores, with nominal 90%
    coverage; no finite-sample coverage guarantee is claimed for the test sets. The alert
    probability averages the five members' temperature-scaled probabilities.
- **Target.** RUL capped at 125 cycles. A point estimate near 125 means "at least about 125 cycles
  left". An interval that reaches 125 is open-ended and reads "at least the lower end".
- **Baseline.** LightGBM on four statistics per sensor (last value, mean, standard deviation,
  slope), with the same folds, window choice, interval calibration and alert calibration. It
  too is evaluated as a five-fold ensemble; only the LSTM is served.
- **Versions.** Each subset was trained three times (repeats), with different fold assignments,
  inner splits, initialisations and calibration truncations. The dashboard, API and Docker image
  serve the first repeat of FD001.
- **Not part of the model.** The optional maintenance-note harness (`predmaint.notes`) is not used
  by the API or dashboard, and its language-model side has never been run.
- **Provenance.** Each `metrics.json` records the git commit the training ran from, the dataset
  file hashes and the software versions; `manifest.json` adds the SHA-256 of every artifact,
  which the audit checks. The published runs were trained from a clean checkout of one commit.
- **Author and licence.** Emin Alper Şensoy, 2026. Code under the [MIT licence](LICENSE).

## Intended use

- A worked, inspectable example of RUL prediction with conformal prediction intervals, a failure
  alert and attributions on a public simulated benchmark, for research, teaching and code review.
- The API and dashboard show how the model behaves, cycle by cycle, on engines it did not train on.

## Out of scope

- **Any decision about a real engine, aircraft or other machine.** The training data is
  simulated, the model has never seen field data, and nothing here is certified or validated for
  safety-critical use.
- Engines, operating conditions or fault modes outside the subset a model was trained on. The
  FD001 model knows one operating condition and one fault mode (HPC degradation).
- Predictions whose response carries `"out_of_distribution": {"flag": true}`. Those are
  extrapolations.

## Data

- **Source.** NASA's C-MAPSS turbofan degradation simulation (Saxena, Goebel, Simon and Eklund,
  PHM08, 2008, [doi:10.1109/PHM.2008.4711414](https://doi.org/10.1109/PHM.2008.4711414)), subsets
  FD001 to FD004. Training trajectories run to failure. Test trajectories stop some time before
  failure, and NASA gives the true RUL at that point.
- **Terms.** NASA's data portal does not specify a licence for this dataset. The raw files are
  therefore not redistributed in git: `python -m predmaint.data download` fetches them from NASA
  and checks their SHA-256.
- **NASA rows and labels that are in the repository.**
  - `artifacts/FD001/replay.csv` holds the raw rows of the FD001 replay engines, taken from
    `train_FD001.txt`, so the dashboard can replay them without the download.
  - `docs/example-request.json` holds 70 cycles of test engine 3 from `test_FD001.txt`.
  - The prediction files under `artifacts/*/predictions/` hold NASA's true RUL for each engine
    they score.
  - `docs/notes-eval/cases.json` is derived from the FD001 replay engines.
- **Splits.** All splits are by engine, never by row:
  - 10% of the training engines are held out for the replay;
  - the rest are cross-fitted in five folds stratified by life length;
  - each fold's training engines are split 85/15 into inner-train and inner-validation.
- **What touched the test labels.** In the final pipeline, only `predmaint.evaluate`, once per
  subset; the [experiment ledger](predictive-maintenance/docs/experiment-ledger.md) records when.
  The labels are then stored in the committed prediction files, which the reports, figures and
  audit read. The version 2 development runs listed in the ledger also printed test scores, so
  the test sets are not an untouched holdout
  ([test-set exposure](predictive-maintenance/docs/evaluation.md#test-set-exposure)).

## Metrics

Generated by `python -m predmaint.report` and checked by `python -m predmaint.audit`.

<!-- results:start -->
**Official test endpoints.** One prediction per test engine, at its last observed cycle. Each repeat of each model is evaluated as its five-fold ensemble: the point prediction is the mean of the five fold models and the interval is their CV+ interval. The API and dashboard serve the first repeat's LSTM ensemble. Entries are mean ± sample sd over 3 repeats on the same test engines; the ± is repeat variability (fold assignment, inner-validation splits, initialisation and calibration truncation; the replay holdout and the preprocessing are fixed), not the precision of the estimate. Bold marks the better model in each subset. RMSE is against the true RUL capped at 125 cycles; RMSE against the uncapped RUL scores the same capped predictions, so it includes the error the cap imposes on engines with more than 125 cycles left. Coverage is the empirical, marginal share of test engines whose capped RUL lies in the interval, an open-ended interval counting as [lower, ∞); its 95% Wilson interval takes n = test engines at the repeat-mean rate and covers engine sampling only. Width truncates open-ended intervals at the cap, a summary defined after the test labels were read; the evaluation notes give two other width statistics.

| Subset | Model | RMSE | PHM08 score | RMSE vs uncapped RUL | Empirical coverage, nominal 90% (95% CI) | Open-ended intervals | Mean width, open-ended truncated at the cap | Alert AP | Alert ECE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FD001 (100 engines) | LSTM | 12.94 ± 0.11 | 306 ± 17 | 14.28 ± 0.14 | 95.3 ± 0.6% (89–98%) | 45 ± 2% | 41.5 ± 1.4 cycles | 0.984 ± 0.004 | 0.045 ± 0.007 |
| FD001 (100 engines) | LightGBM | **12.11 ± 0.24** | **244 ± 3** | **13.36 ± 0.21** | 95.0 ± 1.0% (89–98%) | 39 ± 3% | 46.1 ± 2.7 cycles | **0.984 ± 0.002** | 0.044 ± 0.003 |
| FD002 (259 engines) | LSTM | 12.53 ± 0.11 | 643 ± 43 | 25.27 ± 0.37 | 96.4 ± 0.2% (94–98%) | 43 ± 2% | 39.7 ± 0.8 cycles | 0.996 ± 0.002 | 0.029 ± 0.000 |
| FD002 (259 engines) | LightGBM | **11.47 ± 0.10** | **571 ± 13** | **24.04 ± 0.08** | 95.6 ± 1.2% (93–98%) | 40 ± 4% | 41.6 ± 2.5 cycles | **0.999 ± 0.000** | 0.021 ± 0.002 |
| FD003 (100 engines) | LSTM | 13.22 ± 0.26 | 610 ± 55 | 14.90 ± 0.15 | 94.7 ± 3.1% (89–98%) | 42 ± 1% | 37.6 ± 2.9 cycles | **0.987 ± 0.001** | 0.031 ± 0.010 |
| FD003 (100 engines) | LightGBM | **12.03 ± 0.15** | **281 ± 12** | **13.63 ± 0.16** | 95.0 ± 3.0% (89–98%) | 57 ± 15% | 58.0 ± 11.3 cycles | 0.986 ± 0.006 | 0.031 ± 0.007 |
| FD004 (248 engines) | LSTM | **12.82 ± 0.29** | **737 ± 48** | 25.75 ± 0.06 | 95.8 ± 1.3% (93–98%) | 50 ± 0% | 43.1 ± 1.8 cycles | **0.985 ± 0.002** | 0.025 ± 0.004 |
| FD004 (248 engines) | LightGBM | 13.11 ± 0.15 | 905 ± 34 | **25.41 ± 0.11** | 94.2 ± 0.2% (91–97%) | 62 ± 5% | 56.0 ± 3.5 cycles | 0.984 ± 0.002 | 0.012 ± 0.004 |

**Paired differences, LSTM minus LightGBM, on the same test engines.** Repeat-averaged difference with a 95% percentile interval from a paired engine-level bootstrap: 2,000 resamples of the test engines, each applied to both models and every repeat. It reflects which engines are in the test set, for the trained models; it excludes repeat variability, and the 16 intervals are not corrected for multiple comparisons.

| Subset | RMSE, LSTM minus LightGBM | PHM08 score, LSTM minus LightGBM | Coverage, LSTM minus LightGBM (points) | Interval width, LSTM minus LightGBM |
|---|---:|---:|---:|---:|
| FD001 | +0.83 (-0.70 to +2.39) | +62 (-22 to +180) | +0.3 (-2.3 to +3.0) | -4.6 (-7.7 to -1.8) |
| FD002 | +1.06 (+0.26 to +1.93) | +72 (-42 to +190) | +0.8 (-1.4 to +3.1) | -1.9 (-3.4 to -0.5) |
| FD003 | +1.19 (-0.50 to +3.25) | +329 (-53 to +1048) | -0.3 (-4.0 to +3.7) | -20.4 (-26.5 to -14.9) |
| FD004 | -0.30 (-1.04 to +0.46) | -168 (-438 to +14) | +1.6 (-0.9 to +4.2) | -12.9 (-16.0 to -10.1) |

**Alerts over whole lives, out of fold.** Every cycle of every cross-fitted training engine, each predicted by the single fold model that never saw it, not by an ensemble. The alert fires when the probability of failure within 30 cycles reaches 0.5. Engine counts are means ± sd over repeats; the ECE interval is the repeat-averaged 95% bootstrap interval over engines.

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

**The ensembles over whole lives: replay holdout.** Every cycle of the held-out replay engines, which no network, preprocessing or calibration step used, predicted by each repeat's five-fold ensemble, the form the API serves (only the first repeat's LSTM is deployed). Definitions as above; RMSE and coverage weight engines equally. With 10 to 26 engines per subset, the alert counts are small.

| Subset | Model | Engines | RMSE, every cycle | Empirical coverage, every cycle | Open-ended intervals | First alert, median cycles before failure | Missed | First alert with more than 30 cycles left | Switch-offs before failure per engine |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FD001 | LSTM | 10 | 11.22 ± 0.28 | 98.3 ± 1.5% | 57 ± 1% | 30 ± 1 | 0.0 ± 0.0 | 4.7 ± 0.6 | 0.00 ± 0.00 |
| FD001 | LightGBM | 10 | 8.82 ± 0.24 | 94.4 ± 1.6% | 53 ± 2% | 30 ± 0 | 0.0 ± 0.0 | 5.0 ± 0.0 | 0.17 ± 0.12 |
| FD002 | LSTM | 26 | 14.53 ± 0.20 | 93.5 ± 0.5% | 56 ± 1% | 31 ± 1 | 0.0 ± 0.0 | 13.0 ± 2.6 | 0.13 ± 0.08 |
| FD002 | LightGBM | 26 | 13.29 ± 0.09 | 90.3 ± 2.2% | 53 ± 4% | 30 ± 0 | 0.0 ± 0.0 | 11.3 ± 0.6 | 0.31 ± 0.07 |
| FD003 | LSTM | 10 | 11.47 ± 0.61 | 96.7 ± 1.7% | 63 ± 2% | 30 ± 1 | 0.0 ± 0.0 | 5.0 ± 1.0 | 0.00 ± 0.00 |
| FD003 | LightGBM | 10 | 11.69 ± 0.26 | 92.9 ± 3.0% | 73 ± 10% | 30 ± 1 | 0.0 ± 0.0 | 4.0 ± 1.0 | 0.17 ± 0.15 |
| FD004 | LSTM | 25 | 12.41 ± 0.15 | 96.7 ± 0.4% | 65 ± 1% | 30 ± 2 | 0.0 ± 0.0 | 10.3 ± 3.5 | 0.09 ± 0.08 |
| FD004 | LightGBM | 25 | 11.80 ± 0.09 | 94.4 ± 0.7% | 74 ± 3% | 29 ± 1 | 0.0 ± 0.0 | 11.0 ± 0.0 | 0.27 ± 0.05 |
<!-- results:end -->

## Uncertainty and calibration: what holds and what does not

- **Intervals.** Five-fold CV+ would guarantee less than the nominal 90% even if its conditions
  held, and they do not hold here: the calibration points are not exchangeable with the test
  endpoints, the preprocessing is fitted on the calibration engines' readings, and the folds
  depend on the data. The intervals therefore have nominal coverage, and their evidence is
  empirical coverage
  ([conditions](predictive-maintenance/docs/evaluation.md#what-the-conformal-guarantee-covers)).
  That coverage is marginal: above 90% on every test set, but about 90% (LSTM) or below it
  (LightGBM) for engines in mid-life.
- **The calibration population.** Calibration points are drawn from the test trajectory lengths,
  but their mix of remaining life differs from the test endpoints' (more engines at the cap),
  and the test labels could not be used to match it.
- **The cap.** Upper ends at the cap are open-ended. Under that reading, coverage of the capped
  truth equals coverage of the actual remaining life; read as closed at the cap, the same
  intervals would cover the actual remaining life far less often.
- **Width.** Comparisons of width use three statistics, which treat open-ended intervals
  differently ([interval width](predictive-maintenance/docs/evaluation.md#interval-width)).
- **Alert probability.**
  - It was temperature-scaled on out-of-fold truncation points, for single fold models; the
    served probability averages five members scaled that way.
  - Its expected calibration error is reported on the test endpoints (the ensemble, one point
    per engine) and out of fold (single fold models, every cycle, with intervals that resample
    whole engines). Most probabilities are near 0, so ECE says little about calibration near the
    threshold; the reliability diagrams show each bin with its count.
  - The alert fires at a probability of 0.5, a threshold fixed in advance and not tuned to any
    cost of missed or false alerts.

## Attributions

<!-- attribution:start -->
**Scope.** Integrated gradients (IG) of the served FD001 ensemble's point prediction, relative to a reference window in which every scaled reading equals its operating regime's training mean, with 48 integration steps. 40 windows: the 10 FD001 replay engines at 100, 60, 30 and 10 cycles before failure. One trained ensemble from one seed; each statistic is a median or a count over windows, with no confidence interval, and there is no ground truth for which readings matter.

- **Completeness, a numerical check.** IG satisfies completeness by construction, so the gap between the attributions' sum and the prediction's change from the reference tests only the path-integral approximation: median 0.04 cycles against a median change of 69 cycles, with a worst window of 5.72 cycles (0.19 with 256 steps).
- **Deletion and insertion, perturbation sensitivity.** Readings are replaced by their reference values in order of attribution magnitude, and the prediction's normalised departure from the reference is averaged over the replacement fractions (lower is better for deletion, higher for insertion). In the median window IG beats random order (deletion area 0.097 against 0.557; better in 32 of 40 windows) and a re-initialised network's IG (0.190; better in 32 of 40).
- **The tail.** The medians hide poor windows: 5 of 40 have a deletion area above 1, where partial replacement moves the prediction further from the reference than the full window does, and the mean deletion area is 0.591 against 0.565 for random order. At 100 cycles before failure, where the median change is 7.7 cycles, IG loses to random order in 7 of 10 windows.
- **Input deviation, a negative comparison.** Ranking readings by their distance from the reference alone gives a median deletion area of 0.132 and insertion area of 0.923, against IG's 0.097 and 0.935. IG beats it in 26 of 40 windows on deletion and 18 of 40 on insertion, and the two maps have a median rank correlation of 0.68.
- **Weight randomisation.** Re-initialising every weight, or only the output heads, gives maps with median rank correlations of 0.55 and 0.56 to the trained maps.
- **Reference sensitivity.** With each engine's own mean over its first cycles as the reference instead of the regime mean, the maps change: median rank correlation 0.54, and 2 of the top 3 sensors kept.
<!-- attribution:end -->

Integrated gradients attribute the prediction relative to a reference input, a window at the
operating regime's training mean, and the maps change when the reference changes. They describe
what the prediction is sensitive to and largely track how far readings are from that reference.
Completeness holds by construction, so it is not evidence of faithfulness, and the maps are not a
physical or causal diagnosis.

## Known failure modes

- **Engines with long remaining life.** Engines far from failure get open-ended intervals: the
  model cannot say how much longer than 125 cycles such an engine will run.
- **Mid-life engines.** Errors are largest for engines about 50 to 100 cycles from failure, and
  there both models overestimate the remaining life, the dangerous direction. Most of the worst
  interval misses are such engines given an open-ended interval.
- **Alert timing.** About half of first alerts come before the 30-cycle horizon begins, false
  alarms at that moment by the label's definition, nearly all of them with at most 60 cycles
  left. A few engines on the six-condition subsets, FD002 and FD004, get a first alert with more
  than 60 cycles left, and out of fold the LSTM's single fold models miss a few FD004 engines
  outright.
- **The out-of-distribution flag** only checks that each scaled sensor and the operating
  settings are within the training range (plus 10% of that range). A new fault mode that keeps
  every sensor in range would not be flagged.
- **The worst individual misses** are listed in
  [`failure-cases.md`](predictive-maintenance/docs/failure-cases.md).

## Latency

<!-- latency:start -->
Measured in process on an Intel Core Ultra 7 165H with 16 PyTorch threads, CPU only, for the served 5-network FD001 model:

| Operation | Median (p95) |
|---|---|
| Prediction | 11 ms (p95 13 ms) |
| Prediction with integrated gradients | 255 ms (p95 274 ms) |
| Replay tick, whole 10-engine fleet, no attributions | 61 ms (p95 64 ms) |
| Attribution refresh for one watched engine | 243 ms (p95 278 ms) |

The replay ticks every 1000 ms. At the 95th percentile, the fleet step plus 2 attribution refreshes takes 620 ms, and one more refresh would take it to 898 ms. So each tick refreshes at most `PM_EXPLAIN_PER_TICK` watched engines (2 by default), taking turns when more are watched.
<!-- latency:end -->

## Citation

Cite the dataset as Saxena et al. (2008) above. [CITATION.cff](CITATION.cff) describes this
software.
