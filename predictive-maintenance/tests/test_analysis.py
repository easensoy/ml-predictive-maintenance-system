"""The analysis tools that read a finished run: attribution checks, the failure gallery, the
latency benchmark and the hero figure's engine choice."""

import pandas as pd

from predmaint import attribution_checks, benchmark, gallery
from predmaint.report import hero_engine


def test_attribution_checks_report_every_check(artifacts_dir):
    report = attribution_checks.run(artifacts_dir / "FD001")
    summary = report["summary"]
    assert report["windows"] > 0
    for key in (
        "completeness_gap_cycles",
        "spearman_vs_full_randomisation",
        "spearman_vs_input_deviation",
        "spearman_vs_own_baseline",
    ):
        assert summary[key]["n"] == report["windows"]
    # Integrated gradients is exact up to the Riemann sum: more steps, smaller gap.
    assert (
        summary["completeness_gap_256_steps"]["median"]
        <= summary["completeness_gap_cycles"]["median"] + 1e-6
    )
    curves = report["curves"]
    if report["curve_windows"]:
        assert curves["deletion_ig"][0] == 1.0
        assert abs(curves["deletion_ig"][-1]) < 1e-6


def test_randomised_copy_changes_the_weights_but_not_the_original(artifacts_dir):
    from predmaint.inference import Predictor

    model = Predictor(artifacts_dir / "FD001").model
    before = [p.detach().clone() for p in model.parameters()]
    clone = attribution_checks.randomized(model, seed=3)
    assert all((a == b).all() for a, b in zip(before, model.parameters(), strict=True))
    assert any((a != b).any() for a, b in zip(before, clone.parameters(), strict=True))


def test_gallery_writes_every_section(artifacts_dir, tmp_path):
    out = tmp_path / "failure-cases.md"
    gallery.write(artifacts_dir, out)
    text = out.read_text(encoding="utf-8")
    for heading in ("## Worst interval misses", "## Earliest, latest and missed", "## Where"):
        assert heading in text
    assert "| FD001 |" in text


def test_benchmark_reports_latency_and_hardware(artifacts_dir):
    report = benchmark.run(artifacts_dir / "FD001", repeats=2)
    assert set(report["latency"]) == {
        "predict_without_explanation",
        "predict_with_integrated_gradients",
        "replay_tick_whole_fleet",
        "engine_detail_with_attribution",
    }
    assert report["latency"]["predict_without_explanation"]["median_ms"] > 0
    assert report["hardware"]["logical_cpus"]


def test_hero_engine_rule_prefers_an_engine_with_a_miss():
    rows = []
    for unit, lead, miss in ((1, 40, False), (2, 30, True), (3, 20, True), (4, 10, True)):
        for rul in range(60, -1, -1):
            inside = not (miss and rul == 50)
            rows.append(
                {
                    "unit": unit,
                    "cycle": 61 - rul,
                    "rul": rul,
                    "lower": rul - 5 if inside else rul + 5,
                    "upper": rul + 5,
                    "open_ended": False,
                    "probability": 0.9 if rul <= lead else 0.1,
                }
            )
    # Engines 2-4 miss once; their first-alert leads are 30, 20 and 10: the median is engine 3.
    assert hero_engine(pd.DataFrame(rows), threshold=0.5, cap=125) == 3
