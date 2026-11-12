# Maintenance notes from prediction records

An evaluation harness for one narrow job: turning a structured prediction record (what
`/api/predict` returns, plus the subset's measured coverage) into a short maintenance note. It
is optional. The API, the dashboard and CI do not depend on it.

**Status: template-only: no model call has been made.** Only the deterministic template baseline
has been graded, so no model fidelity, abstention, latency or cost result exists. The cost
controls below have been exercised against fake clients in the tests, never against the API.

## What is graded

Every note, from either system, is graded on four properties (`predmaint.notes.grade`):

| Property | Pass when | How it is checked |
|---|---|---|
| Numeric fidelity | every number in the note can be read off the record, allowing rounding | automatic |
| No unsupported causal claims | the note never turns an attribution into a cause, fault, wear, damage or diagnosis | pattern rubric on every note, plus a fixed sample in `human-review-*.csv` for a person to check the rubric |
| The cap | an open-ended interval reads "at least N cycles", with no upper bound | automatic |
| Abstention | the note declines to estimate exactly when the record is out of distribution, incomplete, self-contradictory, or has a closed interval wider than 80 cycles | automatic |

The baseline is a deterministic template (`predmaint.notes.template_note`) that follows the same
rules the model is given. A template is hard to beat on these four properties by construction,
so the question the harness answers is narrow: does a language model match it on faithfulness
before its fluency is worth anything?

## The cases

`cases.json` is frozen. `python -m predmaint.notes cases` rebuilds it deterministically from the
served FD001 model and its replay engines, which no network trained on:

- real records at many stages of life, straight from the model;
- adversarial edits of further real records: conflicting status and probability, a lower bound
  above the point estimate, missing fields, out-of-distribution readings, very wide and
  open-ended intervals, low measured coverage, a probability exactly at the threshold, and
  unrounded decimals.

## Running it

```bash
uv run python -m predmaint.notes eval            # template and any recorded responses; writes results.md and results.json
```

The model side needs the optional dependencies, an API key in `ANTHROPIC_API_KEY` (read from
the environment only; never write it to a file in this repository) and a spending limit in US
dollars:

```bash
uv sync --extra notes
uv run python -m predmaint.notes eval --claude --max-spend-usd <USD>
```

That would make one call per case and two extra calls for each of the first 30 cases, to
measure repeatability. Each request asks for structured output and sets a server-side fallback
for refusals; neither has been exercised here. Every response would be appended to
`claude-responses.jsonl` with its latency and token counts, so that `results.md` (pass rates,
latency, cost, repeatability and every case where the model loses to the template) could be
regenerated offline.

## Status

`results.md` holds the template's grades only. The template follows the grading rules by
construction, so its pass rates show that the cases and the grader agree, not how well a
language model would do. The model side has not been run, `claude-responses.jsonl` does not
exist, and the human-review sample has not been filled in.
