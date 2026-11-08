# Turbofan remaining useful life on NASA C-MAPSS: LSTM vs. LightGBM

[![CI](https://github.com/easensoy/ml-predictive-maintenance-system/actions/workflows/ci.yml/badge.svg?branch=master&event=push)](https://github.com/easensoy/ml-predictive-maintenance-system/actions/workflows/ci.yml?query=branch%3Amaster)
![Python 3.15](https://img.shields.io/badge/python-3.15-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

<!-- headline:start -->
This project predicts how many cycles a simulated turbofan engine has left, with a nominal 90% prediction interval and an alert for failure within 30 cycles, and compares a five-network LSTM with a LightGBM baseline on NASA's C-MAPSS data under one cross-fitted protocol. LightGBM is at least as accurate (lower test RMSE on 3 of 4 subsets, clearly only on FD002), while the LSTM gives narrower intervals at similar coverage and an alert that switches off before failure less often.
<!-- headline:end -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="predictive-maintenance/docs/figures/hero-dark.png">
  <img alt="One held-out engine over its whole life: the predicted remaining useful life with its nominal 90% interval, the true RUL capped at 125 cycles and the actual remaining life above the cap, the cycles where the truth leaves the interval, and the probability of failure within 30 cycles crossing the alert threshold." src="predictive-maintenance/docs/figures/hero-light.png">
</picture>

<sub>A training engine held out of every training and calibration step, replayed cycle by cycle through the served model. It was picked by a fixed rule, not by eye: among the held-out engines whose interval misses the truth at least once, the one with the median first-alert lead. The target is capped at 125 cycles, so above the cap the solid line (the target) stays flat while the dashed line (the engine's actual remaining life) keeps rising, and the interval becomes open-ended: "at least the lower end".</sub>

<!-- findings:start -->
- **Accuracy.** LightGBM's test RMSE is lower on 3 of 4 subsets, but the paired LSTM − LightGBM difference excludes 0 only on FD002: +1.06 cycles (95% CI +0.26 to +1.93). The LSTM is lower on FD004: -0.30 cycles (95% CI -1.04 to +0.46).
- **Uncertainty.** Pooled over all 707 test engines, the nominal 90% intervals cover 95.8% (LSTM) and 95.0% (LightGBM), both 95% CIs above 90%, but only 91% (LSTM, 76–100 cycles left) and 81% (LightGBM, 51–75 cycles left) in their weakest band of remaining life. 39 to 62% of intervals are open-ended (“at least N cycles”). The LSTM's are narrower on all 4 subsets under each of three width definitions: by 2 to 20 cycles with open-ended intervals counted up to the 125-cycle cap, by 7 to 39 on the engines where both are closed, and by 6 to 35 over each model's own closed intervals, the only one of the three fixed before the test labels were read.
- **Alerts.** Out of fold, over whole lives, both alerts first fire a median 30 to 31 cycles before failure. LightGBM's switches off again before failure 0.33 to 0.49 times per engine and the LSTM's 0.02 to 0.15, or 0.16 to 0.24 against 0.01 to 0.07 counting only switch-offs in the last 30 cycles, when the alert should stay on. The LSTM misses 1.7 of 224 FD004 engines per repeat; LightGBM misses none.
<!-- findings:end -->

All of this is on NASA's simulated run-to-failure data, against a target capped at 125 cycles. The
official test sets were also seen during an earlier development round (every run is in the
[experiment ledger](predictive-maintenance/docs/experiment-ledger.md)), so they are not an
untouched holdout.

**[Reproduce the numbers](#reproduce)** · **[Full evaluation](predictive-maintenance/docs/evaluation.md)** · **[Dashboard](#dashboard)** · **[Failure cases](predictive-maintenance/docs/failure-cases.md)**

## Results

<!-- summary-table:start -->
| Subset (test engines) | Test RMSE | RMSE difference, LSTM − LightGBM (95% CI) | Inside the nominal 90% interval | Mean interval width (cycles) | Open-ended intervals | Alert switch-offs per engine |
|---|---:|---:|---:|---:|---:|---:|
| FD001 (100) | 12.94&nbsp;/&nbsp;**12.11** | +0.83 (-0.70 to +2.39) | 95%&nbsp;/&nbsp;95% | **41.5**&nbsp;/&nbsp;46.1 | 45%&nbsp;/&nbsp;39% | **0.02**&nbsp;/&nbsp;0.39 |
| FD002 (259) | 12.53&nbsp;/&nbsp;**11.47** | +1.06 (+0.26 to +1.93) | 96%&nbsp;/&nbsp;96% | **39.7**&nbsp;/&nbsp;41.6 | 43%&nbsp;/&nbsp;40% | **0.15**&nbsp;/&nbsp;0.36 |
| FD003 (100) | 13.22&nbsp;/&nbsp;**12.03** | +1.19 (-0.50 to +3.25) | 95%&nbsp;/&nbsp;95% | **37.6**&nbsp;/&nbsp;58.0 | 42%&nbsp;/&nbsp;57% | **0.11**&nbsp;/&nbsp;0.33 |
| FD004 (248) | **12.82**&nbsp;/&nbsp;13.11 | -0.30 (-1.04 to +0.46) | 96%&nbsp;/&nbsp;94% | **43.1**&nbsp;/&nbsp;56.0 | 50%&nbsp;/&nbsp;62% | **0.12**&nbsp;/&nbsp;0.49 |

Each cell is LSTM / LightGBM, the mean over 3 repeats of the whole pipeline; bold marks the lower value where lower is better. The test columns score one prediction per official test engine, at its last observed cycle, by each repeat's five-fold ensemble (the mean of the five fold models, with the CV+ interval over them). RMSE is against the true RUL capped at 125 cycles, and coverage is the share of test engines inside the interval. Width counts an open-ended interval up to the cap, a summary defined after the test labels were read. The difference is the paired one, with a 95% bootstrap interval over test engines. Switch-offs are counted out of fold, over every cycle of every cross-fitted training engine, for single fold models.

**Where each model is better**

- **LightGBM:** lower test RMSE on FD001, FD002 and FD003 (clearly only on FD002), lower PHM08 score on FD001, FD002 and FD003 (none clearly) and lower RMSE against the actual, uncapped remaining life on all 4 subsets.
- **LSTM:** lower test RMSE on FD004 (not clearly), lower PHM08 score on FD004 (not clearly), narrower intervals on all 4 subsets (clearly on 4) and fewer alert switch-offs on all 4 subsets.
- **Neither:** no coverage difference excludes 0 on any subset, and alert average precision is 0.984 to 0.999 for both.
<!-- summary-table:end -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="predictive-maintenance/docs/figures/comparison-dark.png">
  <img alt="Test RMSE per subset for the single fold models and the five-fold ensembles of the LSTM and LightGBM, and the paired LSTM-minus-LightGBM difference with 95% bootstrap intervals." src="predictive-maintenance/docs/figures/comparison-light.png">
</picture>

**Reading the comparison.** Each model is evaluated as a five-fold ensemble: the point prediction
averages the five fold models, one per cross-fitting fold. The whole pipeline ran three times with
different seeds.

- **Filled dots** are the three ensembles, one per repeat. Their spread is training-seed
  variability on fixed test engines; the table gives their mean. It is much smaller than the
  uncertainty from which engines happen to be in the test set.
- **Open circles** are the single fold models, five per repeat. They show how much averaging
  helps; they are not an error bar on the ensemble.
- **The right panel** is the paired difference on the same test engines. Its 95% interval comes
  from a bootstrap that resamples test engines and applies the same draw to both models and every
  repeat, so it measures how much the comparison depends on which engines are in the test set. It
  does not cover training variability beyond the three repeats it averages.
- **Multiple comparisons.** The full tables test 16 differences (4 subsets × 4 metrics) without
  correction, so one interval that only just excludes 0 is weak evidence on its own.
- **Published C-MAPSS results** use other caps, splits and test points. They are listed with their
  protocols in [the evaluation notes](predictive-maintenance/docs/evaluation.md#published-results),
  not plotted here.

The per-subset tables with PHM08 scores, RMSE against the uncapped truth, bootstrap intervals for
every difference, alert calibration and false-alert counts are in
[the evaluation notes](predictive-maintenance/docs/evaluation.md#full-results).

## What fails

- **Mid-life engines.** Both models make their largest errors on engines roughly 50 to 100 cycles
  from failure. There they mostly overestimate the remaining life, which is the dangerous
  direction, and coverage falls to about nominal for the LSTM and below it for LightGBM.
- **Open-ended intervals on engines that are already wearing.** Most of the worst interval misses
  are "at least N cycles" intervals for engines in that range that the model still read as
  healthy.
- **The six-condition subsets.** Every first alert that comes with more than 60 cycles left is on
  FD002 or FD004, and the only missed engines are the LSTM's, on FD004.
- **Uneven coverage.** The intervals over-cover the test sets as a whole, covering almost every
  engine close to failure or far from it, but they are tightest, or too tight, in mid-life
  (below). Out of fold, over whole lives, coverage is below nominal for LightGBM on FD001 and FD002
  and for the LSTM on FD003.

The [failure-case gallery](predictive-maintenance/docs/failure-cases.md) lists the worst misses and
the earliest, latest and missed alerts, engine by engine.

## How it was evaluated

The engine is the sampling unit: no split ever puts cycles of one engine on both sides.

- **Held out.** 10% of each subset's training engines are set aside for the dashboard replay. No
  network, scaler or calibration step sees them.
- **Cross-fitted.** The rest are split into five folds by engine, and each fold model is trained
  without its fold. Inside each fold's training engines, an 85/15 split chooses the input window
  (30 to 60 cycles) and the early-stopping point.
- **Calibrated out of fold.** No calibration score comes from a model that used that engine for
  training, early stopping or window choice: each engine's conformity score comes from the one
  fold model that never saw it, at one truncation point per engine drawn from the test sets'
  trajectory lengths. The intervals are CV+ intervals over the five fold models. The alert
  probability is temperature-scaled out of fold, and its 0.5 threshold was fixed in advance.
- **Shared preprocessing.** The feature scaling and the operating-regime clusters are fitted once,
  on the sensor readings (never the labels) of all cross-fitted engines.
- **Test labels read once.** Training never reads NASA's test labels. `predmaint.evaluate` scored
  all three repeats of both models once per subset and recorded it in the
  [experiment ledger](predictive-maintenance/docs/experiment-ledger.md).

[The evaluation notes](predictive-maintenance/docs/evaluation.md) give the full protocol, the
assumptions behind the intervals, the three populations that are scored and the published results.

### Uncertainty

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="predictive-maintenance/docs/figures/uncertainty-dark.png">
  <img alt="Measured coverage of the nominal 90% intervals per subset on the test sets and out of fold, mean interval widths, the share of open-ended intervals, and test coverage by true remaining life for both models." src="predictive-maintenance/docs/figures/uncertainty-light.png">
</picture>

- **Two populations.** The test sets score one point per engine, at its last observed cycle, with
  the CV+ interval of the five-fold ensemble. Out of fold scores every cycle of every cross-fitted
  engine, with engines weighted equally, using the single fold model that never saw the engine and
  that model's own conformal margin. The two contain different mixes of remaining life, so their
  coverages differ.
- **Open-ended intervals.** An interval that reaches the 125-cycle cap is reported as "at least the
  lower end", in the tables, the API and the dashboard. Under that reading, covering the capped
  target and covering the actual remaining life are the same event. Such an interval covers almost
  every engine far from failure, so coverage is really tested on engines nearer failure, which the
  lower panel separates. The right-hand column gives the share of open-ended intervals.
- **Width** counts an open-ended interval up to the cap. The findings above give two other
  definitions, including the one fixed before the test labels were read; all three put the LSTM's
  intervals below LightGBM's on every subset.
- **Nominal, not guaranteed.** 90% is the target the calibration aims at. Even in theory, five-fold
  CV+ guarantees less than that, and only for engines exchangeable with the calibration engines,
  which the test sets are not. [The evaluation notes](predictive-maintenance/docs/evaluation.md#what-the-conformal-guarantee-covers)
  give the bound and why it does not apply, so the measured coverage is the evidence.

### Alerts

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="predictive-maintenance/docs/figures/alerts-dark.png">
  <img alt="Cycles left at the first alert for every cross-fitted engine, with medians and the 30-cycle horizon; the number of times the alert switches off again before failure per engine; and counts of engines, missed engines and first alerts with more than 60 cycles left." src="predictive-maintenance/docs/figures/alerts-light.png">
</picture>

- **The alert** fires when the probability of failure within 30 cycles reaches 0.5. The threshold
  was fixed in advance and is not tuned to the cost of a missed or a false alert.
- **Out of fold** means every cycle of every cross-fitted training engine, each predicted by the
  single fold model that never saw it, not by the five-fold ensemble. The dot strip and the
  counts are the served repeat; the switch-offs are the mean over three repeats.
- **A switch-off** is the alert turning off again before failure. Some of them end an alert that
  fired early, which does no harm, so the findings above also count only the switch-offs in the
  last 30 cycles. "Missed" means no alert while at least one cycle was left. About half of first
  alerts come with more than 30 cycles left, most of them within ten cycles of the horizon.
- **Ranking and calibration** are measured separately: average precision against its base rate,
  and reliability diagrams with cycles per bin, are in
  [the evaluation notes](predictive-maintenance/docs/evaluation.md#alert-ranking-and-calibration).

### Attributions

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="predictive-maintenance/docs/figures/attribution-dark.png">
  <img alt="Integrated-gradients attribution over sensors and the cycles of the model's input window for the hero engine at its first alert, and the net effect per sensor." src="predictive-maintenance/docs/figures/attribution-light.png">
</picture>

The hero engine at the cycle its alert first fires. Each cell is how much one reading moved the
predicted RUL, in cycles, relative to a baseline window in which every scaled reading sits at its
operating regime's training mean. The cells sum to the prediction minus the baseline's prediction.
Left of the dashed line are cycles that only the networks with the longest window read.

[The attribution checks](predictive-maintenance/docs/evaluation.md#attribution-checks) test these
maps on held-out engines. In the median window, deleting readings in attribution order moves the
prediction to the baseline faster than random order or a randomised network's order, but not in
every window. Ranking readings simply by their distance from the baseline does about as well. So
read the maps as input deviation weighted by the model: what the prediction depends on, not a
diagnosis of the engine.

## Dashboard

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="predictive-maintenance/docs/figures/dashboard-dark.png">
  <img alt="Dashboard panel for the hero engine at the cycle its alert first fires: predicted RUL with its nominal 90% interval against the true RUL, and the probability of failure within 30 cycles crossing the alert threshold." src="predictive-maintenance/docs/figures/dashboard-light.png">
</picture>

<!-- dashboard:start -->
Both captures show the same replay state: FD001 replay engine 81 at cycle 207 of 240, the cycle at which its alert first fires (the hero figure shows its whole life). The dashboard predicts 32 cycles (nominal 90% interval 18 to 51) and a 51% probability of failure within 30 cycles; the true RUL, known because the simulated engine ran to failure, is 33. These are the numbers in that engine's row of `replay-holdout.csv.gz`, which `predmaint.audit` checks.
<!-- dashboard:end -->

The dashboard replays simulated engines held out of training, one recorded cycle per tick. It is
not a live feed and not field telemetry. Run it with `uv run python -m predmaint.serve`. The REST
API, a real example response, the serving latency, Docker and the architecture are in
[`docs/serving.md`](predictive-maintenance/docs/serving.md).

## Reproduce

From `predictive-maintenance/`, with [uv](https://docs.astral.sh/uv/) (tested with 0.13.7) and
Python 3.15, which uv can install.

**Check the published results.** These commands are read-only, need no download and no GPU, and
are what CI runs:

```bash
uv sync
uv run python -m predmaint.audit                  # recompute every table, interval and generated block
uv run python -m predmaint.figures --out /tmp/f   # rebuild every figure from the committed predictions
uv run pytest                                     # synthetic-data tests, plus the audit
```

**Regenerate the documents.** These rewrite files in `docs/` and the generated blocks in the
README, the model card, the evaluation notes and the serving page:

```bash
uv run python -m predmaint.report        # findings, tables and the API example
uv run python -m predmaint.figures       # docs/figures
uv run python -m predmaint.gallery       # docs/failure-cases.md
uv run python -m predmaint.screenshots   # dashboard images; needs Edge, Chrome or Chromium
```

**Retrain.** This replaces the published results:

```bash
uv run python -m predmaint.data download          # 12 MB from NASA, checksum-verified
uv run python -m predmaint.train --subset FD001   # and FD002, FD003, FD004
uv run python -m predmaint.evaluate               # reads the test labels; the ledger records it
uv run python -m predmaint.attribution_checks
uv run python -m predmaint.benchmark
```

`predmaint.train` refuses to run from uncommitted sources, so every `metrics.json` names the commit
that produced it. The published runs were trained on a laptop GPU (NVIDIA RTX 1000 Ada, CUDA build
of PyTorch 2.15). The locked environment installs CPU PyTorch, which runs the same pipeline roughly
ten times slower. GPU runs are not bit-for-bit reproducible across devices, so a rerun reproduces
the protocol, not the exact numbers.

## Documentation

- [Evaluation notes](predictive-maintenance/docs/evaluation.md): the protocol, the assumptions
  behind the intervals, the three scored populations, the full per-subset tables, alert ranking and
  calibration, the attribution checks and the published results.
- [Experiment ledger](predictive-maintenance/docs/experiment-ledger.md): every training and
  evaluation run, including each time the test sets were viewed.
- [Failure-case gallery](predictive-maintenance/docs/failure-cases.md): the worst interval misses
  and the earliest, latest and missed alerts.
- [Model card](MODEL_CARD.md): intended and out-of-scope uses, data and caveats.
- [Attribution checks](predictive-maintenance/docs/evaluation.md#attribution-checks):
  completeness, deletion and insertion tests, a randomised network and a changed baseline.
- [Serving](predictive-maintenance/docs/serving.md): the API, latency, Docker and architecture.

## Data, licence and history

- **Data.** A. Saxena, K. Goebel, D. Simon and N. Eklund, "Damage Propagation Modeling for Aircraft
  Engine Run-to-Failure Simulation", *PHM08*, 2008,
  [doi:10.1109/PHM.2008.4711414](https://doi.org/10.1109/PHM.2008.4711414), from the NASA
  Prognostics Center of Excellence. NASA does not state a licence for it, so the raw files are not
  redistributed here; `predmaint.data download` fetches them. The rows this repository does
  contain are listed in the [model card](MODEL_CARD.md#data).
- **Code.** [MIT licence](LICENSE). To cite this software, use [CITATION.cff](CITATION.cff).
- **Methods.** R. F. Barber, E. J. Candès, A. Ramdas and R. J. Tibshirani, "Predictive inference
  with the jackknife+", *Annals of Statistics* 49(1), 2021; Y. Romano, E. Patterson and E. Candès,
  "Conformalized Quantile Regression", *NeurIPS*, 2019; C. Guo, G. Pleiss, Y. Sun and K. Q.
  Weinberger, "On Calibration of Modern Neural Networks", *ICML*, 2017; M. Sundararajan, A. Taly
  and Q. Yan, "Axiomatic Attribution for Deep Networks", *ICML*, 2017; J. Adebayo, J. Gilmer, M.
  Muelly, I. Goodfellow, M. Hardt and B. Kim, "Sanity Checks for Saliency Maps", *NeurIPS*, 2018.
- **History.** Version 1 (2025) described a "production-ready" system. An audit in November 2026
  found that its data were synthetic, its model could not serve a prediction, its dashboard metrics
  were hard-coded and its "live sensors" were random numbers. Version 2 rebuilt it on C-MAPSS. An
  outside review of version 2 found that calibration reused the engines that had chosen the model,
  that intervals were read as closed at the cap, and that some claims went beyond the evidence.
  Version 2.1, this repository, fixed those and reran the evaluation. Every version is in the git
  history.
