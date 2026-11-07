# Experiment ledger

Every model configuration trained for this repository, its validation scores, and whether its
test-set scores were looked at. A configuration counts as "test viewed" if any test-set metric
was printed or saved while it was under consideration, whether or not it was kept.

## v2 development, 2026-11-06 (test set viewed for every run)

These rows are reconstructed from the run folders kept from development: each folder's
`metrics.json` (configuration, device, test metrics) and `history.json` (validation RMSE per
epoch). Nothing was typed in by hand.

Common settings: engines split 70/15/15 into train, validation and holdout (split seed 42), the
target capped at 125 cycles, a 2-layer LSTM with 64 hidden units and dropout 0.2, Adam at
learning rate 1e-3, batch 256, early stopping with patience 12. "Validation RMSE per seed" is the
best epoch's RMSE over every window of the validation engines. The training command printed the
test RMSE, test coverage and alert AP at the end of every run, so the test set was seen for all
of them.

| Started | Run | Subset | Window | EWMA α | Max epochs | Seeds | Device | Validation RMSE per seed | Mean | Test RMSE | Test coverage | Test alert AP | LightGBM test RMSE |
|---|---|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|
| 2026-11-06 20:25 | smoke_FD001 | FD001 | 30 | – | 3 | 0 | not recorded | 18.06 | 18.06 | 16.10 | 92% | 0.950 | 12.23 |
| 2026-11-06 20:28 | single_FD001 | FD001 | 30 | – | 80 | 0 | not recorded | 13.91 | 13.91 | 13.96 | 87% | 0.990 | 12.23 |
| 2026-11-06 20:42 | exp_ewma | FD001 | 30 | 0.3 | 80 | 0,1 | not recorded | 13.18 / 13.11 | 13.15 | 13.19 | 87% | 0.986 | 12.59 |
| 2026-11-06 20:42 | exp_default | FD001 | 30 | – | 80 | 0,1 | not recorded | 13.91 / 13.49 | 13.70 | 14.08 | 90% | 0.992 | 12.23 |
| 2026-11-06 20:42 | exp_w40 | FD001 | 40 | – | 80 | 0,1 | not recorded | 11.82 / 11.53 | 11.68 | 12.47 | 88% | 0.985 | 12.87 |
| 2026-11-06 20:43 | exp_w40_ewma | FD001 | 40 | 0.3 | 80 | 0,1 | not recorded | 11.60 / 11.72 | 11.66 | 12.88 | 88% | 0.988 | 12.96 |
| 2026-11-06 20:58 | exp_fd002 | FD002 | 40 | – | 80 | 0 | not recorded | 14.05 | 14.05 | 14.35 | 88% | 0.992 | 12.22 |
| 2026-11-06 20:58 | exp_w60 | FD001 | 60 | – | 80 | 0,1 | not recorded | 10.57 / 10.51 | 10.54 | 12.71 | 84% | 0.984 | 12.52 |
| 2026-11-06 20:59 | exp_w50 | FD001 | 50 | – | 80 | 0,1 | not recorded | 11.02 / 11.37 | 11.20 | 13.18 | 85% | 0.992 | 13.35 |
| 2026-11-06 21:00 | exp_fd004 | FD004 | 40 | – | 80 | 0 | not recorded | 15.41 | 15.41 | 14.43 | 85% | 0.969 | 13.99 |
| 2026-11-06 21:12 | sweep FD001_w30 | FD001 | 30 | – | 80 | 0,1 | GPU | 14.25 / 13.89 | 14.07 | 13.24 | 85% | 0.988 | – |
| 2026-11-06 21:12 | sweep FD003_w30 | FD003 | 30 | – | 80 | 0,1 | GPU | 10.92 / 11.14 | 11.03 | 13.14 | 82% | 0.990 | – |
| 2026-11-06 21:13 | sweep FD002_w30 | FD002 | 30 | – | 80 | 0,1 | GPU | 15.56 / 15.51 | 15.54 | 14.64 | 87% | 0.990 | – |
| 2026-11-06 21:13 | sweep FD001_w40 | FD001 | 40 | – | 80 | 0,1 | GPU | 12.30 / 11.95 | 12.13 | 12.42 | 90% | 0.983 | – |
| 2026-11-06 21:14 | sweep FD004_w30 | FD004 | 30 | – | 80 | 0,1 | GPU | 15.44 / 15.87 | 15.65 | 14.71 | 84% | 0.975 | – |
| 2026-11-06 21:14 | sweep FD003_w40 | FD003 | 40 | – | 80 | 0,1 | GPU | 10.70 / 10.52 | 10.61 | 13.04 | 84% | 0.995 | – |
| 2026-11-06 21:15 | sweep FD001_w50 | FD001 | 50 | – | 80 | 0,1 | GPU | 11.69 / 11.30 | 11.50 | 14.09 | 83% | 0.992 | – |
| 2026-11-06 21:17 | sweep FD003_w50 | FD003 | 50 | – | 80 | 0,1 | GPU | 10.37 / 9.76 | 10.07 | 13.12 | 88% | 0.998 | – |
| 2026-11-06 21:17 | sweep FD002_w40 | FD002 | 40 | – | 80 | 0,1 | GPU | 13.99 / 13.83 | 13.91 | 13.48 | 91% | 0.996 | – |
| 2026-11-06 21:17 | sweep FD001_w60 | FD001 | 60 | – | 80 | 0,1 | GPU | 11.16 / 11.09 | 11.12 | 12.79 | 87% | 0.982 | – |
| 2026-11-06 21:18 | sweep FD004_w40 | FD004 | 40 | – | 80 | 0,1 | GPU | 14.84 / 15.19 | 15.01 | 13.29 | 83% | 0.970 | – |
| 2026-11-06 21:19 | sweep FD003_w60 | FD003 | 60 | – | 80 | 0,1 | GPU | 10.32 / 10.40 | 10.36 | 13.00 | 84% | 0.998 | – |
| 2026-11-06 21:20 | sweep FD002_w50 | FD002 | 50 | – | 80 | 0,1 | GPU | 13.66 / 13.33 | 13.50 | 13.58 | 86% | 0.998 | – |
| 2026-11-06 21:22 | sweep FD004_w50 | FD004 | 50 | – | 80 | 0,1 | GPU | 14.82 / 15.02 | 14.92 | 13.31 | 85% | 0.982 | – |
| 2026-11-06 21:23 | sweep FD002_w60 | FD002 | 60 | – | 80 | 0,1 | GPU | 13.64 / 12.90 | 13.27 | 12.70 | 88% | 0.996 | – |
| 2026-11-06 21:25 | sweep FD004_w60 | FD004 | 60 | – | 80 | 0,1 | GPU | 15.14 / 14.90 | 15.02 | 13.30 | 88% | 0.985 | – |
| 2026-11-06 21:30 | published FD001 | FD001 | 60 | – | 80 | 0,1,2,3,4 | GPU | 11.16 / 11.09 / 10.89 / 10.64 / 10.46 | 10.85 | 12.89 | 86% | 0.985 | 12.52 |
| 2026-11-06 21:31 | published FD003 | FD003 | 50 | – | 80 | 0,1,2,3,4 | GPU | 10.37 / 9.76 / 10.23 / 10.43 / 10.30 | 10.22 | 13.19 | 83% | 0.998 | 12.47 |
| 2026-11-06 21:32 | published FD002 | FD002 | 60 | – | 80 | 0,1,2,3,4 | GPU | 13.64 / 12.90 / 12.70 / 13.46 / 13.50 | 13.24 | 12.48 | 91% | 0.996 | 11.54 |
| 2026-11-06 21:33 | published FD004 | FD004 | 50 | – | 80 | 0,1,2,3,4 | GPU | 14.82 / 15.02 / 14.98 / 14.88 / 15.02 | 14.94 | 13.06 | 82% | 0.981 | 13.48 |

How the published configuration was chosen:

- The window length (30, 40, 50 or 60 cycles) was picked per subset by the lowest mean validation
  RMSE over seeds 0 and 1 in the GPU sweep: 60 for FD001 and FD002, 50 for FD003 and FD004. The
  test RMSE of every candidate was printed next to it.
- Causal input smoothing (EWMA, α = 0.3) was tried on FD001 at windows 30 and 40 and not kept.
- Two calibration variants were tried during development and dropped: conformal margins fitted
  separately by stage of life (Mondrian bins), and clipping the conformal margin at zero. Their
  test coverage was looked at; the exact numbers were not logged.
- The architecture and optimiser settings above were set before these runs and not tuned.
- The validation engines chose the window, stopped training and fitted the conformal margin and
  the alert temperature, so calibration reused the model-selection engines.

Because of all this, the published v2 results are development-exposed: treat their test scores
as optimistic.
