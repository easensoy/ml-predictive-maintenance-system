"""The maintenance-note harness: graders, the template baseline and the Claude client path.

No test calls the API: the Claude path runs against a fake client that returns canned replies,
and building a real client fails in every test.
"""

import copy
import json
from types import SimpleNamespace

import pytest

from predmaint import notes

REAL_MAKE_CLIENT = notes.make_client


@pytest.fixture(autouse=True)
def _no_real_client(monkeypatch):
    def refuse():
        raise AssertionError("a test tried to build a real API client")

    monkeypatch.setattr(notes, "make_client", refuse)


def _record(**changes) -> dict:
    record = {
        "case_id": "001-normal",
        "kind": "normal",
        "engine": 17,
        "subset": "FD001",
        "cycle": 145,
        "rul": {"point": 38.2, "lower": 22.4, "upper": 55.1, "open_ended": False, "capped_at": 125},
        "failure_within_horizon": {"horizon_cycles": 30, "probability": 0.24, "threshold": 0.5},
        "status": "watch",
        "top_attributions": [
            {"symbol": "Ps30", "description": "Static pressure", "contribution_cycles": -8.1},
            {"symbol": "T50", "description": "LPT outlet temperature", "contribution_cycles": -4.2},
        ],
        "out_of_distribution": {"flag": False, "sensors": []},
        "measured_coverage": 0.88,
    }
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(record.get(key), dict):
            record[key] = {**record[key], **value}
        else:
            record[key] = value
    return record


OPEN = {"rul": {"point": 121.0, "lower": 96.0, "upper": 125.0, "open_ended": True}}


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({}, None),
        ({"out_of_distribution": {"flag": True, "sensors": ["sensor_11"]}}, "out_of_distribution"),
        ({"rul": {"point": None}}, "missing_fields"),
        ({"failure_within_horizon": {"probability": None}}, "missing_fields"),
        ({"status": "alert"}, "conflicting_fields"),
        ({"rul": {"lower": 40.0}}, "conflicting_fields"),
        ({"rul": {"lower": 5.0, "upper": 100.0}}, "too_uncertain"),
        ({"top_attributions": []}, None),
        (OPEN | {"status": "healthy"}, None),
    ],
)
def test_expected_behaviour(changes, reason):
    assert notes.expected(_record(**changes))["reason"] == reason


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"out_of_distribution": {"flag": True, "sensors": ["sensor_11"]}},
        {"status": "alert"},
        OPEN | {"status": "healthy"},
        {"top_attributions": []},
        {"measured_coverage": 0.71},
    ],
)
def test_template_passes_every_grader(changes):
    record = _record(**changes)
    graded = notes.grade(notes.template_note(record), record)
    assert all(graded[p] for p in notes.PROPERTIES), graded


def test_template_states_open_ended_intervals_as_a_lower_bound():
    record = _record(**OPEN, status="healthy")
    text = notes.template_note(record)["note"]
    assert "at least 96 cycles" in text
    assert "125" not in text


def test_grader_catches_an_invented_number():
    record = _record()
    note = {"note": "Engine 17: about 38 cycles left, maybe 60.", "abstained": False}
    graded = notes.grade(note, record)
    assert not graded["numeric_fidelity"]
    assert graded["unsupported_numbers"] == [60.0]


def test_grader_catches_causal_claims():
    note = {"note": "Ps30 is rising because the compressor is degrading.", "abstained": False}
    graded = notes.grade(note, _record())
    assert not graded["no_causal_claims"]


def test_grader_catches_an_upper_bound_on_an_open_ended_interval():
    record = _record(**OPEN, status="healthy")
    note = {"note": "Engine 17: 121 cycles left, 96 to 125 cycles.", "abstained": False}
    assert not notes.grade(note, record)["cap_handling"]


def test_grader_catches_wrong_abstention():
    record = _record(out_of_distribution={"flag": True, "sensors": ["sensor_11"]})
    note = {"note": "Engine 17: 38 cycles left.", "abstained": False}
    assert not notes.grade(note, record)["abstention"]


def test_claude_request_uses_structured_output_and_server_side_fallback():
    request = notes.claude_request(_record())
    assert request["model"] == notes.MODEL
    assert request["max_tokens"] == 16000
    assert request["fallbacks"] == "default"
    assert request["betas"] == ["server-side-fallback-2026-07-01"]
    assert request["output_config"]["format"]["schema"] == notes.OUTPUT_SCHEMA
    assert request["output_config"]["effort"] == "medium"
    assert "temperature" not in request
    payload = json.loads(request["messages"][0]["content"])
    assert "case_id" not in payload
    assert payload["rul"]["point"] == 38.2


def test_claude_request_takes_model_effort_and_max_tokens():
    request = notes.claude_request(_record(), "high", model="claude-opus-5", max_tokens=4000)
    assert (request["model"], request["max_tokens"]) == ("claude-opus-5", 4000)
    assert request["output_config"]["effort"] == "high"


def _key(record=None, repetition=0, **request_args):
    record = record or _record()
    request = notes.claude_request(record, **request_args)
    return notes.cache_key(request, record["case_id"], repetition)


def test_cache_key_changes_with_model_prompt_schema_record_and_budget(monkeypatch):
    base = _key()
    assert _key() == base
    changed = {
        "model": _key(model="claude-opus-5"),
        "effort": _key(effort="high"),
        "max_tokens": _key(max_tokens=8000),
        "record": _key(record=_record(cycle=146)),
        "case": _key(record=_record(case_id="002-normal")),
        "repetition": _key(repetition=1),
    }
    with monkeypatch.context() as m:
        m.setattr(notes, "SYSTEM_PROMPT", notes.SYSTEM_PROMPT + " Be brief.")
        changed["prompt"] = _key()
    with monkeypatch.context() as m:
        m.setattr(notes, "OUTPUT_SCHEMA", {**notes.OUTPUT_SCHEMA, "required": ["note"]})
        changed["schema"] = _key()
    assert base not in changed.values()
    assert len(set(changed.values())) == len(changed)


def test_worst_case_bound_prices_the_whole_output_ceiling_and_a_fallback():
    request = notes.claude_request(_record())
    tokens_in = notes.input_token_bound(request)
    assert tokens_in > len(request["system"]) + len(request["messages"][0]["content"])
    # claude-opus-5-5 at $4 / $20 per million, plus a fallback attempt at the dearest listed
    # price, $5 / $25, each with the full 16,000-token output ceiling
    expected = (tokens_in * 4 + 16000 * 20 + tokens_in * 5 + 16000 * 25) / 1e6
    assert notes.call_bound_usd(request) == pytest.approx(expected)
    smaller = notes.claude_request(_record(), max_tokens=2000)
    assert notes.call_bound_usd(smaller) < notes.call_bound_usd(request)


class FakeMessages:
    """Stands in for client.beta.messages; replies come from a list of canned responses, each
    (stop_reason, text) or (stop_reason, text, usage)."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        stop_reason, text, *usage = self.replies.pop(0)
        return SimpleNamespace(
            model=kwargs["model"],
            stop_reason=stop_reason,
            content=[SimpleNamespace(type="text", text=text)],
            usage=usage[0] if usage else SimpleNamespace(input_tokens=600, output_tokens=90),
        )


def _client(*replies):
    return SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages(replies)))


GOOD = json.dumps(
    {"note": "Engine 17 (FD001), cycle 145: 38 cycles left, 22 to 55.", "abstained": False}
)


def _cases(n):
    return [_record(case_id=f"{i:03d}-normal") for i in range(n)]


def _bound(cases, run):
    return sum(notes.call_bound_usd(call["request"]) for call in notes.plan_calls(cases, run))


def test_claude_note_parses_structured_output_and_records_usage():
    reply = json.dumps({"note": "Engine 17: 38 cycles left.", "abstained": False})
    result = notes.claude_note(_client(("end_turn", reply)), _record())
    assert result["note"] == "Engine 17: 38 cycles left."
    assert result["abstained"] is False
    assert (result["input_tokens"], result["output_tokens"]) == (600, 90)


def test_claude_note_handles_refusals_and_malformed_replies():
    refused = notes.claude_note(_client(("refusal", "")), _record())
    assert refused["refused"] is True
    assert refused["abstained"] is True
    malformed = notes.claude_note(_client(("end_turn", "not json")), _record())
    assert malformed["malformed"] is True


def test_run_claude_records_every_call_and_summarises(tmp_path):
    cases = [_record(case_id=f"{i:03d}-normal") for i in range(4)]
    good = json.dumps(
        {"note": "Engine 17 (FD001), cycle 145: 38 cycles left, 22 to 55.", "abstained": False}
    )
    client = _client(*[("end_turn", good)] * 6)
    out = tmp_path / "responses.jsonl"
    notes.run_claude(cases, out, client=client, repeat_cases=1, repeats=3)
    recorded = notes.load_recorded(out)
    assert sum(len(runs) for runs in recorded.values()) == 6
    summary = notes.claude_summary(cases, recorded)
    assert summary["calls"] == 6
    assert summary["repeatability"]["cases"] == 1
    assert summary["repeatability"]["same_abstention"] == 1.0
    assert summary["cost_usd"]["per_note"] == pytest.approx((600 * 4 + 90 * 20) / 1e6)
    assert summary["evaluation"]["cases"] == 4


def test_make_client_explains_what_is_missing(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        REAL_MAKE_CLIENT()


def test_build_cases_mixes_real_and_adversarial_records(artifacts_dir):
    cases = notes.build_cases(artifacts_dir / "FD001", normal=10, adversarial=24)
    kinds = {c["kind"] for c in cases}
    assert len(cases) == 34
    assert {"normal", "out-of-distribution", "conflict-status", "open-ended"} <= kinds
    assert len({c["case_id"] for c in cases}) == len(cases)
    again = notes.build_cases(artifacts_dir / "FD001", normal=10, adversarial=24)
    assert again == cases  # frozen: the same model gives the same cases
    for case in cases:
        record = copy.deepcopy(case)
        graded = notes.grade(notes.template_note(record), record)
        assert graded["abstention"], case["case_id"]
