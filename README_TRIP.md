# TripSupervisor

A 6-agent MAS built to deliberately induce MAST's 14 failure modes, two
concurrency (FC4) failure modes, and a hallucination point — all with a
single model (`qwen3:8b` by default).

## Setup

```bash
ollama serve                # if not already running
ollama pull qwen3:8b        # if not already pulled
pip install aiohttp
```

## Files

| File | Purpose |
|---|---|
| `agent.py`, `blackboard.py`, `workflow.py` | Core harness (unchanged from earlier steps): Ollama-calling `Agent`, the racy/safe `Blackboard`, and the DAG `WorkflowEngine`. |
| `trip_mock_apis.py` | Mock Flight/Hotel/Activity APIs with realistic quirks (an undisclosed `loyalty_id` requirement, an undisclosed activity price cap) that create natural failure opportunities with zero injection. |
| `trip_failure_config.py` | `FailureConfig` — one boolean flag per MAST mode (+ blackboard mode + 2 hallucination flags). Whatever's `True` for a run is your silver ground-truth label. |
| `trip_json.py` | Defensive JSON extraction from model free-text (qwen3:8b doesn't always return clean JSON). |
| `trip_hallucination.py` | Diffs the Trip Digest's text against the ground-truth `final_record` (dollar amounts, confirmation numbers) — no LLM judge needed for this check. |
| `trip_workflow.py` | Builds the 7-node DAG. All prompts/callbacks are closures parameterized by `FailureConfig`, so one call to `build_trip_workflow(cfg)` wires up whichever failure modes you want for that run. |
| `run_trip_supervisor.py` | CLI entry point. |

## The DAG

```
Supervisor (sequential root)
     |
  -----+------+-----
  |          |         |
Flight     Hotel     Activity      <- PARALLEL, all racing on "budget_remaining"
  |          |         |
  -----+------+-----
       |
  Budget Verifier (sequential: waits for all three)
       |
    Checkout (sequential: assembles ground-truth final_record)
       |
   Trip Digest (sequential: the hallucination point)
```

## Running it

```bash
# Clean run, unsafe blackboard (races possible)
python run_trip_supervisor.py

# Safe blackboard as a control (no races possible)
python run_trip_supervisor.py --blackboard-mode safe

# Induce specific MAST modes + a hallucination point, comma-separated
python run_trip_supervisor.py --enable fm_2_4_information_withholding,fm_3_2_incomplete_verification,hallucinate_prompt_pressure
```

Available flags (see `trip_failure_config.py` for the mechanism behind each):
`fm_1_1_disobey_task_spec`, `fm_1_2_disobey_role_spec`, `fm_1_4_loss_of_history`,
`fm_1_5_unaware_termination`, `fm_2_2_fail_to_clarify`, `fm_2_3_task_derailment`,
`fm_2_4_information_withholding`, `fm_2_5_ignored_other_agent_input`,
`fm_2_6_reasoning_action_mismatch`, `fm_3_1_premature_termination`,
`fm_3_2_incomplete_verification`, `fm_3_3_incorrect_verification`,
`hallucinate_drop_context`, `hallucinate_prompt_pressure`.

Not every MAST mode has a flag here: **1.3 (Step Repetition), 2.1 (Conversation
Reset), and 3.3's sibling checks** are easier to induce by running the same
task many times and watching for *emergent* repeats/resets from qwen3:8b
itself (especially under time/token pressure) rather than by scripting them —
worth tracking as natural-failure baseline rates rather than injected ones.

## Output

Each run prints:
1. Per-agent status (success/timeout/error/malformed) — malformed is what
   you'll see if qwen3:8b's JSON doesn't parse even after one retry
   (`trip_json.parse_json_response` + `Agent`'s `validate_fn` retry).
2. The `final_record` — ground truth for everything downstream.
3. The Trip Digest text.
4. The hallucination check (`unsupported_amounts`, `unsupported_confirmations`).
5. The budget accumulator race check — see the note below, this is the
   part most worth understanding before you scale up runs.

Logs land in `logs/trip_workflow_<id>.jsonl` (per-agent calls) and
`logs/trip_blackboard_<id>.jsonl` (every blackboard read/write, versioned).

## Important: two different race-detection strategies for two different key shapes

`blackboard.py` now has **two** race-detection methods, and they're not
interchangeable:

- **`detect_lost_updates(key)`** — correct for a key where each write fully
  *replaces* the value (e.g. a shared text draft). There, the last writer
  really does discard everyone else's edits, and comparing consecutive
  writes' agent IDs correctly flags that.
- **`verify_accumulator(key, expected_value)`** — correct for a key that
  multiple agents *build on* (like `budget_remaining`, where three agents
  each subtract their own cost). Here, every write is *supposed* to
  overwrite the last one — that's not a bug by itself. The actual race
  signature is a **wrong final value**, because some agent's subtraction
  was computed from a stale base. `verify_accumulator` checks exactly that
  by comparing the final stored value against what it should be if every
  update had correctly built on the one before it.

Using `detect_lost_updates` on `budget_remaining` gives a **false positive
in both safe and unsafe mode** (three different agents always write to it
once each, so it always looks "raced" by that method's logic, whether or
not an actual race occurred). This was caught by the test suite (see
`/tmp/smoke_trip.py`-style tests) before it could contaminate a dataset —
worth keeping a similar smoke test in your own pipeline as you extend this,
since a race-detector bug produces silent false-positive "failures" in your
silver labels, which is exactly the kind of harness-vs-genuine-failure
confound your research needs to avoid.

## A note on the default delays

With the default `read_modify_write` delays in `trip_workflow.py`
(flight=0.4s, hotel=0.1s, activity=0.25s), **the unsafe-mode race is
deterministic, not probabilistic** — all three agents' reads happen before
any of their writes land, so every unsafe run loses two of the three budget
subtractions. If you want a mix of racy and clean outcomes across many
runs (closer to how real concurrency bugs behave — sometimes they hit,
sometimes they don't), randomize these delays per run, e.g. drawing each
from `random.uniform(0.05, 0.5)` inside each agent's `on_success` callback
in `trip_workflow.py`.

## Suggested next steps

1. Run a batch (10-20 runs) with no injection to establish a natural
   failure/hallucination baseline for qwen3:8b on this task before adding
   injected modes.
2. Run the same batch with `--blackboard-mode safe` as a matched control.
3. Sweep the `--enable` flags one at a time, then in combinations, to see
   which MAST modes qwen3:8b induces reliably vs. rarely at this model
   size — this itself is a finding (MAST's own paper found weaker/smaller
   models fail differently than strong ones).
4. Feed `logs/trip_workflow_*.jsonl` + `logs/trip_blackboard_*.jsonl` into
   the LLM-as-a-Judge annotator (next build step) to test whether it can
   recover which flags were set, purely from the trace.
