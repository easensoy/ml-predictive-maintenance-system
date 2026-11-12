"""Turn one structured prediction record into a short maintenance note, and measure how well
that works.

    python -m predmaint.notes cases            # build the frozen cases, docs/notes-eval/cases.json
    python -m predmaint.notes eval             # grade the template and every recorded response
    python -m predmaint.notes eval --claude --max-spend-usd USD   # also call Claude (needs a key)
    python -m predmaint.notes eval --regrade   # grade one configuration's recorded responses

A deterministic template is the baseline. Each note is graded on four properties:

1. **Numeric fidelity** (automatic): every number in the note can be read off the record.
2. **No unsupported causal claims** (rubric): attributions say what the model's output depends
   on, so a note may not call them causes, faults, wear or damage. A pattern rubric grades every
   note, and ``human-review-*.csv`` holds a fixed sample for a person to check the rubric.
3. **The cap**: an open-ended interval must read "at least N cycles", with no upper bound.
4. **Abstention**: when the record is out of distribution, incomplete, self-contradictory or too
   uncertain to plan on, the note must decline to give an estimate; otherwise it must not.

The Claude side needs the optional ``notes`` dependencies (``uv sync --extra notes``) and an API
key in ``ANTHROPIC_API_KEY``; CI, the API and the dashboard never need either. Before any call it
bounds the worst-case cost of the calls still to make and refuses to start above
``--max-spend-usd``; during the run it adds up the spend from each response's ``usage`` and stops
before a call could take it past the limit. Every response is appended to
``claude-responses.jsonl`` under a cache key, so a rerun makes only the missing calls, and grading
(plain ``eval``, or ``--regrade`` for one configuration) runs offline without a client.
"""

import argparse
import copy
import csv
import hashlib
import json
import os
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from predmaint.config import PROJECT_ROOT, Settings

EVAL_DIR = PROJECT_ROOT / "docs" / "notes-eval"
MODEL = "claude-opus-5-5"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
EFFORT = "medium"  # the default level on claude-opus-5-5, sent explicitly
# The ceiling on each response's output, thinking included: thinking is always on for MODEL and
# counts toward max_tokens. It bounds what a call can cost; the bill is usage.output_tokens.
# About 16,000 is the documented default for non-streaming requests such as these.
MAX_TOKENS = 16_000
REPEAT_CASES, REPEATS = 30, 3
FALLBACK_BETA = "server-side-fallback-2026-07-01"  # the header that fallbacks="default" needs
# USD per million tokens, copied from the Claude API reference bundled with Claude Code (models
# table cached 2026-10-06) and not re-checked against the pricing page: check it before a paid
# run. claude-opus-5 and claude-opus-4-8 are the fallback targets that reference expects for
# claude-opus-5-5. A model missing here cannot be run, because its cost cannot be bounded.
PRICE_PER_MTOK = {
    "claude-opus-5-5": {"input": 4.00, "output": 20.00},
    "claude-opus-5": {"input": 5.00, "output": 25.00},
    "claude-opus-4-8": {"input": 5.00, "output": 25.00},
}
# Cache writes bill at 1.25x (5-minute) or 2x (1-hour) base input and cache reads at a fraction of
# it. No request here sets cache_control, so both counts should stay zero; they are priced at the
# top of those ranges so that the tracked spend can only over-count.
CACHE_WRITE_FACTOR, CACHE_READ_FACTOR = 2.0, 1.0
# Added to one token per byte of the prompt for message framing and anything the API adds around
# it: an assumed allowance, not a documented figure.
FRAMING_TOKENS = 1_000
TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)
MAX_WIDTH = 80  # cycles: a wider closed interval is too uncertain to plan maintenance on
WORDS = 80
ABSTAIN_MARKER = "no maintenance estimate"
PROPERTIES = ("numeric_fidelity", "no_causal_claims", "cap_handling", "abstention")

SYSTEM_PROMPT = f"""You write maintenance notes for turbofan engines. Each request is one \
prediction record as JSON. Follow these rules exactly.

1. Use only facts in the record. Every number you write must appear in the record; you may round \
to whole numbers and write probabilities as percentages.
2. Give the remaining-life estimate (rul.point) and its interval. If rul.open_ended is true, write \
the interval as "at least <rul.lower> cycles" and give no upper bound: the model's target is \
capped at rul.capped_at cycles, so it cannot say how much longer the engine will run.
3. The interval is a nominal 90% interval. Say how often it held on test engines, using \
measured_coverage. Never call it guaranteed.
4. top_attributions describe what the model's prediction depends on. Never describe them as \
causes, faults, wear, damage or a diagnosis.
5. Set abstained to true and give no remaining-life estimate when out_of_distribution.flag is \
true; when rul.point, rul.lower or the failure probability is missing; when fields contradict \
each other (for example status "alert" while the probability is below the threshold, or rul.lower \
above rul.point); or when a closed interval is wider than {MAX_WIDTH} cycles. Say briefly why.
6. At most {WORDS} words of plain English, no markdown."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"note": {"type": "string"}, "abstained": {"type": "boolean"}},
    "required": ["note", "abstained"],
    "additionalProperties": False,
}

# Words that turn "what the model depends on" into a physical claim the record cannot support.
CAUSAL_PATTERNS = (
    r"\bbecause\b",
    r"\bcause[ds]?\b",
    r"\bdue to\b",
    r"\bfault",
    r"\bfailing\b",
    r"\bwear",
    r"\bworn\b",
    r"\bdamage",
    r"\bdegrad",
    r"\bleak",
    r"\bcrack",
    r"\berosion\b",
    r"\bfouling\b",
    r"\bdiagnos",
)
# The template's disclaimer names causes only to deny them; it is removed before grading.
DISCLAIMER = "this shows what the model uses, not a diagnosed cause"
NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")


# --------------------------------------------------------------------------------------------
# What a correct note must do


def expected(record: dict) -> dict:
    """The behaviour the record calls for: abstain (and why) or not, and open-ended or not."""
    rul, alert = record["rul"], record["failure_within_horizon"]
    point, lower, upper = rul.get("point"), rul.get("lower"), rul.get("upper")
    probability, threshold = alert.get("probability"), alert["threshold"]
    status = record.get("status")
    open_ended = bool(rul.get("open_ended"))
    reason = None
    if record["out_of_distribution"]["flag"]:
        reason = "out_of_distribution"
    elif point is None or lower is None or probability is None:
        reason = "missing_fields"
    elif (
        lower > point
        or (not open_ended and upper is not None and upper < point)
        or (status == "alert" and probability < threshold)
        or (status in {"healthy", "watch"} and probability >= threshold)
    ):
        reason = "conflicting_fields"
    elif not open_ended and upper is not None and upper - lower > MAX_WIDTH:
        reason = "too_uncertain"
    return {"abstain": reason is not None, "reason": reason, "open_ended": open_ended}


# --------------------------------------------------------------------------------------------
# The baseline

WHY = {
    "out_of_distribution": "its readings are outside the range the model was trained on",
    "missing_fields": "the prediction record is incomplete",
    "conflicting_fields": "the prediction record contradicts itself",
    "too_uncertain": "the model's interval is too wide to plan maintenance on",
}


def template_note(record: dict) -> dict:
    """Deterministic note that follows the same rules Claude is given."""
    e = expected(record)
    head = f"Engine {record['engine']} ({record['subset']}), cycle {record['cycle']}:"
    if e["abstain"]:
        return {
            "note": f"{head} {ABSTAIN_MARKER}: {WHY[e['reason']]}. Check the sensors and "
            "readings before relying on this engine's prediction.",
            "abstained": True,
        }
    rul, alert = record["rul"], record["failure_within_horizon"]
    horizon = alert["horizon_cycles"]
    if e["open_ended"]:
        interval = f"at least {rul['lower']:.0f} cycles"
    else:
        interval = f"between {rul['lower']:.0f} and {rul['upper']:.0f} cycles"
    action = {
        "alert": f"Plan maintenance within the next {horizon} cycles.",
        "watch": f"Monitor closely: failure within {horizon} cycles is possible.",
        "healthy": "No action needed now.",
    }.get(record.get("status") or "", "Review the engine's status.")
    drivers = [a["symbol"] for a in record.get("top_attributions") or []][:2]
    depends = (
        f" The estimate depends most on {' and '.join(drivers)}; {DISCLAIMER}." if drivers else ""
    )
    note = (
        f"{head} estimated remaining life {rul['point']:.0f} cycles, {interval} (a nominal 90% "
        f"interval that held for {100 * record['measured_coverage']:.0f}% of test engines). "
        f"Probability of failure within {horizon} cycles: {100 * alert['probability']:.0f}%."
        f"{depends} {action}"
    )
    return {"note": note, "abstained": False}


# --------------------------------------------------------------------------------------------
# Graders


def _allowed_numbers(record: dict) -> list[float]:
    rul, alert = record["rul"], record["failure_within_horizon"]
    probability = alert.get("probability")
    values = [
        record["engine"],
        record["cycle"],
        rul.get("capped_at"),
        90,
        alert["horizon_cycles"],
        rul.get("point"),
        rul.get("lower"),
        rul.get("upper"),
        None if probability is None else 100 * probability,
        100 * alert["threshold"],
        100 * record["measured_coverage"],
        MAX_WIDTH,
        *(abs(a["contribution_cycles"]) for a in record.get("top_attributions") or []),
    ]
    return [float(v) for v in values if v is not None]


def grade(note: dict, record: dict) -> dict:
    """Pass or fail on each property, with the evidence."""
    text = note["note"]
    lowered = text.lower()
    e = expected(record)
    allowed = _allowed_numbers(record)
    numbers = [float(m) for m in NUMBER.findall(re.sub(r"\bFD00\d\b", "", text))]
    unsupported = [
        x for x in numbers if not any(abs(x - a) <= max(0.51, 0.006 * abs(a)) for a in allowed)
    ]
    causal = [p for p in CAUSAL_PATTERNS if re.search(p, lowered.replace(DISCLAIMER, ""))]
    abstained = bool(note["abstained"])
    cap_ok = True
    if e["open_ended"] and not abstained:
        lower = f"{record['rul']['lower']:.0f}"
        says_at_least = bool(re.search(rf"(at least|≥|>=|no fewer than)\s*{lower}\b", lowered))
        gives_upper = bool(
            re.search(rf"\b{lower}\s*(to|-|–|and)\s*\d+", lowered)
            or re.search(r"\bat most\b|\bno more than\b|\bup to\b", lowered)
        )
        cap_ok = says_at_least and not gives_upper
    return {
        "numeric_fidelity": not unsupported,
        "unsupported_numbers": unsupported,
        "no_causal_claims": not causal,
        "causal_patterns": causal,
        "cap_handling": cap_ok,
        "abstention": abstained == e["abstain"],
        "words": len(text.split()),
    }


def evaluate(cases: list[dict], notes: dict[str, dict]) -> dict:
    graded = {c["case_id"]: grade(notes[c["case_id"]], c) for c in cases if c["case_id"] in notes}
    by_kind: dict[str, list[dict]] = {}
    for c in cases:
        if c["case_id"] in graded:
            by_kind.setdefault(c["kind"], []).append(graded[c["case_id"]])
    return {
        "cases": len(graded),
        "pass_rate": {p: float(np.mean([g[p] for g in graded.values()])) for p in PROPERTIES},
        "all_four": float(np.mean([all(g[p] for p in PROPERTIES) for g in graded.values()])),
        "by_kind": {
            kind: {p: float(np.mean([g[p] for g in rows])) for p in PROPERTIES} | {"n": len(rows)}
            for kind, rows in by_kind.items()
        },
        "graded": graded,
    }


# --------------------------------------------------------------------------------------------
# Claude


@dataclass(frozen=True)
class RunConfig:
    """One configuration of the Claude side. model, effort and max_tokens go into every request,
    and so into its cache key; repeat_cases, repeats and max_cases pick the (case, repetition)
    pairs that are called."""

    model: str = MODEL
    effort: str = EFFORT
    max_tokens: int = MAX_TOKENS
    repeat_cases: int = REPEAT_CASES
    repeats: int = REPEATS
    max_cases: int | None = None


def claude_request(
    record: dict, effort: str = EFFORT, model: str = MODEL, max_tokens: int = MAX_TOKENS
) -> dict[str, Any]:
    """The request sent for one record (separate, so tests can check it without a key).

    Structured output constrains the reply to the schema; the server-side fallback re-runs a
    declined request on a fallback model instead of returning the refusal. max_tokens caps
    thinking plus reply: it is the most a reply can be billed for, not what it is billed."""
    payload = {k: v for k, v in record.items() if k not in {"case_id", "kind"}}
    return {
        "model": model,
        "max_tokens": max_tokens,
        "betas": [FALLBACK_BETA],
        "fallbacks": "default",
        "system": SYSTEM_PROMPT,
        "output_config": {
            "effort": effort,
            "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
        },
        "messages": [{"role": "user", "content": json.dumps(payload, sort_keys=True)}],
    }


def cache_key(request: dict, case_id: str, repetition: int) -> str:
    """The identity of one call: a SHA-256 of the whole request (model, system prompt, output
    schema, record, effort, max_tokens, betas and fallbacks), the case and the repetition index.
    A response recorded under the key is reused instead of being paid for again."""
    blob = json.dumps(
        {"request": request, "case_id": case_id, "repetition": repetition},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def plan_calls(cases: list[dict], run: RunConfig) -> list[dict]:
    """Every call the configuration needs, in order: each case once, and the first
    ``repeat_cases`` cases ``repeats`` times."""
    calls: list[dict] = []
    for i, case in enumerate(cases[: run.max_cases]):
        request = claude_request(case, run.effort, run.model, run.max_tokens)
        for repetition in range(run.repeats if i < run.repeat_cases else 1):
            key = cache_key(request, case["case_id"], repetition)
            calls.append({"case": case, "repetition": repetition, "request": request, "key": key})
    return calls


def price_of(model: str) -> dict[str, float]:
    if model not in PRICE_PER_MTOK:
        raise SystemExit(
            f"No price for {model!r} in PRICE_PER_MTOK, so the cost of a run cannot be bounded. "
            "Add its input and output prices from the pricing page first."
        )
    return PRICE_PER_MTOK[model]


def input_token_bound(request: dict) -> int:
    """The request's input tokens, over-counted without an API call: one token per UTF-8 byte of
    the system prompt, output format and messages, plus FRAMING_TOKENS. The tokenizer is not
    documented, so this is an assumption rather than a guarantee, which is why a run also stops
    on the spend it records."""
    sent = request["system"] + json.dumps(request["output_config"])
    sent += json.dumps(request["messages"])
    return len(sent.encode("utf-8")) + FRAMING_TOKENS


def call_bound_usd(request: dict) -> float:
    """The most one call can cost, worked out before it is made: the over-counted input plus the
    full max_tokens of output at the requested model's price, and, with fallbacks on, the same
    again at the dearest price in PRICE_PER_MTOK for the attempt a fallback model can make. The
    declined attempt is billed too when it declined mid-reply or in a billed refusal category."""
    tokens_in, tokens_out = input_token_bound(request), request["max_tokens"]
    prices = [price_of(request["model"])]
    if request.get("fallbacks"):
        prices.append({k: max(p[k] for p in PRICE_PER_MTOK.values()) for k in ("input", "output")})
    return sum(tokens_in * p["input"] + tokens_out * p["output"] for p in prices) / 1e6


def claude_note(client: Any, record: dict, effort: str = "medium") -> dict:
    """One Claude note, with its latency, token usage and stop reason."""
    started = time.perf_counter()
    response = client.beta.messages.create(**claude_request(record, effort))
    result: dict[str, Any] = {
        "case_id": record["case_id"],
        "model": getattr(response, "model", MODEL),
        "stop_reason": response.stop_reason,
        "latency_s": round(time.perf_counter() - started, 3),
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    if response.stop_reason == "refusal":
        return result | {"note": "", "abstained": True, "refused": True}
    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        parsed = json.loads(text)
        return result | {"note": str(parsed["note"]), "abstained": bool(parsed["abstained"])}
    except json.JSONDecodeError, KeyError, TypeError:
        return result | {"note": text, "abstained": False, "malformed": True}


def make_client() -> Any:
    """The API client, or a clear message about what is missing."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "The Claude side needs an API key in ANTHROPIC_API_KEY. The template side runs "
            "without one: python -m predmaint.notes eval"
        )
    try:
        import anthropic
    except ImportError as exc:
        raise SystemExit("Install the optional dependencies first: uv sync --extra notes") from exc
    return anthropic.Anthropic()


def run_claude(
    cases: list[dict], out: Path, client: Any = None, repeat_cases: int = 30, repeats: int = 3
) -> None:
    """Call Claude once per case, and ``repeats`` times on the first ``repeat_cases`` cases,
    appending every response to ``out``."""
    client = client or make_client()
    with out.open("a", encoding="utf-8") as log:
        for i, case in enumerate(cases):
            for _ in range(repeats if i < repeat_cases else 1):
                log.write(json.dumps(claude_note(client, case)) + "\n")
                log.flush()


def load_recorded(path: Path) -> dict[str, list[dict]]:
    recorded: dict[str, list[dict]] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                recorded.setdefault(row["case_id"], []).append(row)
    return recorded


def claude_summary(cases: list[dict], recorded: dict[str, list[dict]]) -> dict:
    first = {case_id: runs[0] for case_id, runs in recorded.items()}
    rows = [r for runs in recorded.values() for r in runs]
    latencies = [r["latency_s"] for r in rows]
    cost = (
        sum(
            r["input_tokens"] * PRICE_PER_MTOK["input"]
            + r["output_tokens"] * PRICE_PER_MTOK["output"]
            for r in rows
        )
        / 1e6
    )
    repeated = [runs for runs in recorded.values() if len(runs) > 1]
    same_decision = [len({r["abstained"] for r in runs}) == 1 for runs in repeated]
    same_numbers = [len({tuple(NUMBER.findall(r["note"])) for r in runs}) == 1 for runs in repeated]
    return {
        "calls": len(rows),
        "latency_s": {
            "median": statistics.median(latencies),
            "p95": float(np.percentile(latencies, 95)),
        },
        "cost_usd": {"total": cost, "per_note": cost / len(rows)},
        "repeatability": {
            "cases": len(repeated),
            "same_abstention": float(np.mean(same_decision)) if repeated else None,
            "same_numbers": float(np.mean(same_numbers)) if repeated else None,
        },
        "refusals": sum(bool(r.get("refused")) for r in rows),
        "malformed": sum(bool(r.get("malformed")) for r in rows),
        "notes": first,
        "evaluation": evaluate(cases, first),
    }


# --------------------------------------------------------------------------------------------
# The frozen case set


def _record(predictor: Any, history: pd.DataFrame, unit: int, coverage: float) -> dict:
    result = predictor.predict_readings(history, explain=True)
    rul, alert = result["rul"], result["failure_within_horizon"]
    sensors = (result.get("explanation") or {}).get("sensors", [])[:3]
    ood = result["out_of_distribution"]
    return {
        "engine": unit,
        "subset": predictor.contract.subset,
        "cycle": int(history["cycle"].iloc[-1]),
        "rul": {k: rul[k] for k in ("point", "lower", "upper", "open_ended", "capped_at")},
        "failure_within_horizon": {
            "horizon_cycles": alert["horizon_cycles"],
            "probability": alert["probability"],
            "threshold": predictor.alert_threshold,
        },
        "status": result["status"],
        "top_attributions": [
            {k: s[k] for k in ("symbol", "description", "contribution_cycles")} for s in sensors
        ],
        "out_of_distribution": {
            "flag": ood["flag"],
            "sensors": [s["sensor"] for s in ood["sensors_outside_training_range"]],
        },
        "measured_coverage": coverage,
    }


def _conflict_alert(r: dict) -> None:
    r["status"] = "alert"
    r["failure_within_horizon"]["probability"] = 0.12


def _conflict_healthy(r: dict) -> None:
    r["status"] = "healthy"
    r["failure_within_horizon"]["probability"] = 0.81


def _conflict_bounds(r: dict) -> None:
    r["rul"]["lower"] = r["rul"]["point"] + 9


def _missing_point(r: dict) -> None:
    r["rul"]["point"] = None


def _missing_probability(r: dict) -> None:
    r["failure_within_horizon"]["probability"] = None


def _missing_attributions(r: dict) -> None:
    r["top_attributions"] = []


def _out_of_distribution(r: dict) -> None:
    r["out_of_distribution"] = {"flag": True, "sensors": ["sensor_11"]}


def _wide_interval(r: dict) -> None:
    r["rul"].update(point=55.0, lower=5.0, upper=110.0, open_ended=False)


def _open_ended(r: dict) -> None:
    r["rul"].update(point=121.3, lower=96.4, upper=125.0, open_ended=True)
    r["status"] = "healthy"
    r["failure_within_horizon"]["probability"] = 0.0


def _low_coverage(r: dict) -> None:
    r["measured_coverage"] = 0.71


def _at_threshold(r: dict) -> None:
    r["status"] = "alert"
    r["failure_within_horizon"]["probability"] = 0.5


def _decimals(r: dict) -> None:
    r["rul"]["point"] = round(max(r["rul"]["point"], r["rul"]["lower"]) + 0.46, 2)
    if not r["rul"]["open_ended"]:
        r["rul"]["upper"] = max(r["rul"]["upper"], r["rul"]["point"])


EDITS = (
    ("conflict-status", _conflict_alert),
    ("conflict-status", _conflict_healthy),
    ("conflict-bounds", _conflict_bounds),
    ("missing-point", _missing_point),
    ("missing-probability", _missing_probability),
    ("missing-attributions", _missing_attributions),
    ("out-of-distribution", _out_of_distribution),
    ("wide-interval", _wide_interval),
    ("open-ended", _open_ended),
    ("low-coverage", _low_coverage),
    ("at-threshold", _at_threshold),
    ("decimals", _decimals),
)


def build_cases(model_dir: Path, seed: int = 0, normal: int = 90, adversarial: int = 60) -> list:
    """Real records from the served model on replay engines (never trained on) at several
    stages of life, plus adversarial edits of further real records. Deterministic."""
    from predmaint.inference import Predictor

    predictor = Predictor(model_dir)
    metrics = json.loads((model_dir / "metrics.json").read_text())
    repeat = str(metrics["served"]["repeat"])
    coverage = metrics["test"]["models"]["lstm"]["per_repeat"][repeat]["coverage"]
    replay = pd.read_csv(model_dir / "replay.csv")
    records = []
    for _, engine in replay.groupby("unit"):
        engine = engine.sort_values("cycle").reset_index(drop=True)
        unit = int(engine["unit"].to_numpy()[0])
        for left in range(5, int(engine["rul"].max()) + 1, 7):
            rows = np.flatnonzero(engine["rul"].to_numpy() == left)
            if len(rows):
                records.append(_record(predictor, engine.iloc[: rows[0] + 1], unit, coverage))
    order = np.random.default_rng(seed).permutation(len(records))
    cases: list[dict] = []
    for i in order[:normal]:
        cases.append({"case_id": f"{len(cases):03d}-normal", "kind": "normal", **records[i]})
    pool = [records[i] for i in order[normal:]] or [records[i] for i in order]
    for i in range(adversarial):
        kind, edit = EDITS[i % len(EDITS)]
        record = copy.deepcopy(pool[i % len(pool)])
        edit(record)
        cases.append({"case_id": f"{len(cases):03d}-{kind}", "kind": kind, **record})
    return cases


# --------------------------------------------------------------------------------------------
# Reports


def write_report(cases: list[dict], template: dict, claude: dict | None, out_dir: Path) -> None:
    kinds = sorted({c["kind"] for c in cases})
    lines = [
        "# Maintenance notes: template vs Claude"
        if claude
        else "# Maintenance-note harness: template baseline only",
        "",
        "Generated by `python -m predmaint.notes eval` from `cases.json`: "
        f"{len(cases)} records, {sum(c['kind'] == 'normal' for c in cases)} straight from the "
        "served model on replay engines and the rest adversarial edits of real records.",
        "",
        "| Property | Template | Claude |" if claude else "| Property | Template |",
        "|---|---:|---:|" if claude else "|---|---:|",
    ]
    for p in (*PROPERTIES, "all_four"):
        t = template["all_four"] if p == "all_four" else template["pass_rate"][p]
        if claude:
            ev = claude["evaluation"]
            cell = f"{100 * (ev['all_four'] if p == 'all_four' else ev['pass_rate'][p]):.0f}%"
            lines.append(f"| {p.replace('_', ' ')} | {100 * t:.0f}% | {cell} |")
        else:
            lines.append(f"| {p.replace('_', ' ')} | {100 * t:.0f}% |")
    lines += ["", "| Case kind | Cases | Template, all four |", "|---|---:|---:|"]
    for kind in kinds:
        k = template["by_kind"][kind]
        all_four = np.mean(
            [
                all(template["graded"][c["case_id"]][p] for p in PROPERTIES)
                for c in cases
                if c["kind"] == kind
            ]
        )
        lines.append(f"| {kind} | {k['n']} | {100 * all_four:.0f}% |")
    if claude:
        r = claude["repeatability"]
        lines += [
            "",
            f"Claude ({MODEL}): {claude['calls']} calls, median latency "
            f"{claude['latency_s']['median']:.1f} s (p95 {claude['latency_s']['p95']:.1f} s), "
            f"about ${claude['cost_usd']['per_note']:.4f} per note at list prices; "
            f"{claude['refusals']} refusals and {claude['malformed']} malformed replies. On "
            f"{r['cases']} cases run three times, the abstention decision was the same "
            f"{100 * (r['same_abstention'] or 0):.0f}% of the time and the numbers "
            f"{100 * (r['same_numbers'] or 0):.0f}%.",
            "",
            "## Cases where Claude loses to the template",
            "",
            "| Case | Kind | Failed | Claude's note |",
            "|---|---|---|---|",
        ]
        for case in cases:
            case_id = case["case_id"]
            mine = claude["evaluation"]["graded"].get(case_id)
            if mine is None:
                continue
            lost = [p for p in PROPERTIES if template["graded"][case_id][p] and not mine[p]]
            if lost:
                text = claude["notes"][case_id]["note"].replace("|", "\\|")
                lines.append(f"| {case_id} | {case['kind']} | {', '.join(lost)} | {text} |")
    else:
        lines += [
            "",
            "The template follows the grading rules by construction, so its pass rates check that "
            "the cases and the grader agree; they say nothing about the quality of model-written "
            "notes. The language-model side has not been run: no model call has been made, so no "
            "model fidelity, abstention, latency or cost figure exists. Running it needs an API "
            "key in `ANTHROPIC_API_KEY` and `uv sync --extra notes`, then "
            "`python -m predmaint.notes eval --claude`.",
        ]
    (out_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_human_review(cases: list[dict], notes: dict[str, dict], path: Path, n: int = 20):
    """A fixed sample for a person to check the causal-claim rubric against."""
    present = [c for c in cases if c["case_id"] in notes]
    sample = present[:: max(1, len(present) // n)][:n]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "kind", "note", "rubric_flags", "human_verdict"])
        for c in sample:
            flags = grade(notes[c["case_id"]], c)["causal_patterns"]
            writer.writerow(
                [c["case_id"], c["kind"], notes[c["case_id"]]["note"], ";".join(flags), ""]
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m predmaint.notes",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=["cases", "eval"])
    parser.add_argument("--claude", action="store_true", help="also call Claude (needs a key)")
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--out", type=Path, default=EVAL_DIR)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    cases_path = args.out / "cases.json"
    if args.command == "cases":
        cases = build_cases(Settings.from_env().artifacts_dir / args.subset)
        cases_path.write_text(json.dumps(cases, indent=1), encoding="utf-8")
        print(f"Wrote {len(cases)} cases to {cases_path}")
        return 0
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    template_notes = {c["case_id"]: template_note(c) for c in cases}
    template = evaluate(cases, template_notes)
    recorded_path = args.out / "claude-responses.jsonl"
    if args.claude:
        run_claude(cases, recorded_path)
    recorded = load_recorded(recorded_path)
    claude = claude_summary(cases, recorded) if recorded else None
    write_report(cases, template, claude, args.out)
    export_human_review(cases, template_notes, args.out / "human-review-template.csv")
    summary = {"template": {k: v for k, v in template.items() if k != "graded"}}
    if claude:
        export_human_review(cases, claude["notes"], args.out / "human-review-claude.csv")
        summary["claude"] = {k: v for k, v in claude.items() if k not in {"notes", "evaluation"}}
        summary["claude"]["evaluation"] = {
            k: v for k, v in claude["evaluation"].items() if k != "graded"
        }
    (args.out / "results.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print((args.out / "results.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
