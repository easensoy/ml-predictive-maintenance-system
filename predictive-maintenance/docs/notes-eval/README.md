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

With the defaults that would make one call per case and two extra calls for each of the first 30
cases, to measure repeatability: 210 calls on the 150 frozen cases. Each request asks for
structured output and sets a server-side fallback for refusals; neither has been exercised here.
Every response would be appended to `claude-responses.jsonl` with its cache key, latency and
token counts, so that `results.md` (pass rates, latency, cost, repeatability and every case where
the model loses to the template) could be regenerated offline.

| Flag | Default | Sets |
|---|---|---|
| `--max-spend-usd` | none; required with `--claude` | the most one run may spend |
| `--model` | `claude-opus-5-5` | the model, which needs a price in `PRICE_PER_MTOK` |
| `--effort` | `medium` | `output_config.effort`: `low`, `medium`, `high`, `xhigh` or `max` |
| `--max-tokens` | `16000` | `max_tokens`, the ceiling on each reply's thinking plus text |
| `--repeat-cases` | `30` | how many of the first cases are called more than once |
| `--repeats` | `3` | calls for each of those cases |
| `--max-cases` | every case | only the first N cases, for a pilot run |

### Resuming and grading offline

Every request has a cache key: a SHA-256 of the whole request (model, system prompt, output
schema, the record's JSON, effort, `max_tokens`, beta header and fallback setting), the case ID
and the repetition index. Each line of `claude-responses.jsonl` stores its key. Rerunning the same
command makes only the calls whose keys are not recorded yet, so a run that stopped at its limit,
or was interrupted, carries on where it left off, and raising `--repeats` or `--max-cases` makes
only the new calls. Changing anything that goes into a request (model, effort, `max_tokens`, the
prompt, the schema or a case) gives new keys, so those calls are made again.

Grading needs no API call, key or `notes` dependency:

```bash
uv run python -m predmaint.notes eval                           # every recorded response
uv run python -m predmaint.notes eval --regrade                 # the default configuration's responses only
uv run python -m predmaint.notes eval --regrade --effort high   # another configuration's
```

`--regrade` takes the same settings as `--claude`, picks the recorded lines whose keys that
configuration produces, and prints how many of its planned calls are recorded and the most the
missing ones could cost, so it also shows the bound before anything is spent. Lines written
before cache keys existed are still read and graded by plain `eval`; they match no key, so
`--regrade` leaves them out and a `--claude` run makes those calls again.

## Cost controls

- **A worst-case bound before any call.** For each call not yet recorded, the bound takes the
  input tokens, over-counted as one token per UTF-8 byte of the system prompt, output format and
  record plus 1,000 for framing, and the full `max_tokens` of output, at the requested model's
  prices. A server-side fallback can bill both the declined attempt and the fallback attempt, so
  each call also carries a second attempt of the same size at the dearest price in
  `PRICE_PER_MTOK`. The run refuses to start when the sum exceeds `--max-spend-usd`; the refusal
  and `--regrade` both print the figure.
- **A stop during the run.** The spend is added up from each response's usage counts, and the run
  stops before any call whose worst case could take it past the limit. It also stops at once if
  an attempt ran on a model with no price, since the spend can then no longer be tracked.
- **Billing from usage, not from `max_tokens`.** `max_tokens` is a ceiling on each reply,
  thinking included. It sets the bound, but a call is billed for the tokens it used: the spend
  comes from `usage.input_tokens` and `usage.output_tokens` (which includes thinking) and, when a
  fallback ran, from every entry of `usage.iterations` at that attempt's model's price, because
  the top-level counts then cover only the attempt that answered. Refusals in categories that
  are not billed are counted anyway, and no request sets `cache_control`, so cache counts should
  be zero; any that appear are priced at the top of their documented range. The tracked spend can
  therefore only over-count.
- **Prices.** `PRICE_PER_MTOK` holds only the verified prices below. A model without a price is
  refused, because its cost cannot be bounded.
- **The output budget was not lowered.** `max_tokens` stays at 16,000, about the documented
  default for non-streaming requests. Thinking is always on for Claude Opus 5.5 and counts toward
  the ceiling, and the documentation gives no figure for how much it thinks at `medium` on a task
  like this, so a smaller ceiling could cut replies off. Each response records `output_tokens` and
  `thinking_tokens`, and the report gives the largest reply, so a pilot run (`--max-cases`) can
  show whether a lower `--max-tokens` is safe. A reply cut off at the ceiling is recorded as
  truncated and graded as a failure.

### Verified facts

Checked on 2026-11-08 against the Claude API reference bundled with Claude Code 2.1.317 (its
models table is dated 2026-10-06), the source of the `anthropic` SDK at 2.6.0 (the version
`uv.lock` pins) and, where the bundled reference was silent, the official pages linked.

| Fact | Source |
|---|---|
| `claude-opus-5-5` costs $4 per million input tokens and $20 per million output tokens; `claude-opus-5` and `claude-opus-4-8` cost $5 and $25 | bundled `claude-api` skill: models table in `SKILL.md`, `shared/models.md`, `shared/model-migration.md` (Migrating to Claude Opus 5.5). The [pricing page](https://platform.claude.com/docs/en/about-claude/pricing.md) was not fetched |
| Cache writes bill at 1.25x (5-minute) or 2x (1-hour) base input and cache reads at a fraction of it | bundled `shared/cost-optimization.md` (Step 0) |
| `max_tokens` is a hard limit on total output, thinking plus reply text; on Claude Opus 5.5 thinking is always on and effort, `medium` by default, is the main control of how much it reasons and what a request costs | [Effort](https://platform.claude.com/docs/en/build-with-claude/effort.md) (Claude Opus 5.5); bundled `shared/model-migration.md` |
| Thinking is billed as output: `usage.output_tokens` is the inclusive total used for billing and `output_tokens_details.thinking_tokens` the part that was reasoning; `thinking.display` changes visibility, not billing | [Extended thinking](https://platform.claude.com/docs/en/build-with-claude/extended-thinking.md); `anthropic` 2.6.0 `types/beta/beta_usage.py`; bundled `shared/model-migration.md` |
| With server-side fallback, `usage.iterations` is the per-attempt billing record, each attempt at its own model's rates, and top-level `usage` covers only the attempt that answered; a declined attempt is billed too when it declined mid-reply or in the `bio`, `frontier_llm` or `reasoning_extraction` categories | [Refusals and fallback](https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback.md) |
| `fallbacks: "default"` needs the beta header `server-side-fallback-2026-07-01`, and retries a declined request on the model Anthropic recommends for its refusal category | Refusals and fallback; bundled `shared/model-migration.md` (Migrating to Claude Opus 5) and `python/claude-api/README.md` |
| `client.beta.messages.create` takes `betas`, `fallbacks="default"`, a required `max_tokens` and `output_config={"effort": ..., "format": {"type": "json_schema", "schema": ...}}`, with effort one of `low` to `max` | `anthropic` 2.6.0: `types/beta/message_create_params.py`, `beta_output_config_param.py`, `beta_json_output_format_param.py`; bundled `python/claude-api/sdk-upgrade.md` and `python/claude-api/tool-use.md` |
| About 16,000 is the recommended `max_tokens` for non-streaming requests | bundled `SKILL.md` (Common pitfalls) |

### Not verified

- **The input estimate.** One token per byte plus 1,000 is an assumption: the tokenizer is not
  documented, and no `count_tokens` request is made, since that is an API call too. The stop on
  recorded spend is the backstop.
- **The fallback model.** Default routing is not published. The bound assumes a fallback attempt
  costs no more than the dearest listed price, $5 and $25 (Claude Opus 5 and Claude Opus 4.8, the
  targets the bundled reference expects for Claude Opus 5.5), and that a declined request gets one
  fallback attempt, as the documentation describes; a run stops if an attempt shows up on an
  unlisted model.
- **SDK retries.** The bound counts one request per attempt. The SDK retries connection errors,
  timeouts, 408, 409, 429 and 5xx responses twice by default; whether a request that timed out
  after being processed is billed was not checked.
- **Prices change.** The prices come from the bundled reference's table dated 2026-10-06; check
  the pricing page before a paid run.
- **Nothing here has met the API.** The request shape matches the SDK source and the
  documentation, but no request has been sent.

## Status

`results.md` holds the template's grades only. The template follows the grading rules by
construction, so its pass rates show that the cases and the grader agree, not how well a
language model would do. The model side has not been run, `claude-responses.jsonl` does not
exist, and the human-review sample has not been filled in.
