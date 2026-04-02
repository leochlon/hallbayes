# Objective Optimization Verification Skill

Use this when the task is to improve or satisfy a **well-defined objective**.

Examples:
- lower a benchmark metric
- increase a pass rate
- reduce latency or cost
- improve answer quality against a fixed evaluator
- satisfy a binary acceptance test while keeping guardrails intact

**Goal:** turn “try things until it feels better” into a persisted loop of baseline -> hypothesis -> smallest experiment -> measurement -> keep/revert -> repeat.

The loop:
1. Start or load a run.
2. Define the objective card.
3. Measure and store the baseline.
4. Pick one bottleneck or hypothesis.
5. Run the smallest next experiment.
6. Store evidence and record the attempt.
7. Audit the objective claims.
8. Run `detect_hallucination` on the cited result summary.
9. Keep or revert, then repeat.

---

## Mandatory external state
- Start or load a run before substantive work.
- Store all measurements, benchmarks, logs, diffs, repros, and evaluator outputs as real spans.
- Treat `run.json`, `evidence.tsv`, and `attempts.tsv` as working memory.
- Re-read `list_spans` and `list_attempts` before each iteration.
- Record every baseline, experiment, audit result, keep/discard decision, and revert as an attempt row.
- If code or config changes are possible, note git state before each attempt and record it in the ledger.

## Invalid states
- Optimizing before the objective, evaluator, and guardrails are operationally defined.
- Claiming improvement without a stored baseline and a stored post-change measurement.
- Re-running `audit_trace_budget` with no new evidence and no claim downgrade.
- Keeping a failed or non-improving change when git can revert/reset it.
- Stopping because you feel stuck before re-reading the ledgers and choosing the next smallest experiment.
- Changing the objective or guardrails mid-run without recording the change as evidence.

## Objective card
Make these explicit before looping:
- **OBJECTIVE_NAME** — what is being improved or satisfied.
- **OPTIMIZATION_DIRECTION** — minimize / maximize / satisfy / keep-below / keep-above.
- **EVALUATOR** — the command, benchmark, test, rubric, or measurement that decides success.
- **GUARDRAILS** — what must not regress.
- **ALLOWED_LEVERS** — what can be changed.
- **STOP_RULE** — when to stop.

## Minimum verified claims
Drive these to supported status:
- **OBJECTIVE_DEFINED** — the objective and evaluator are correctly specified.
- **BASELINE_MEASURED** — the current state has been measured and stored.
- **EXPERIMENT_MECHANISM** — each attempt says what changed and why it should move the objective.
- **CURRENT_BEST_RESULT** — the best retained attempt actually improves the objective or satisfies the acceptance test.
- **GUARDRAILS_OK** — regressions were checked, or residual risk is explicitly bounded.

## Objective loop
LOOP UNTIL THE STOP CONDITIONS ARE TRUE:
1. Read the current evidence ledger and attempt ledger.
2. Choose ONE unresolved bottleneck, constraint, or hypothesis.
3. Plan the SMALLEST next experiment.
4. Note git state if applicable, run the experiment, and store all outputs as spans.
5. Record an attempt row with claim id, action, git state, objective metric/value, result summary, and next step.
6. Run `audit_trace_budget` on the affected objective claims.
7. If flagged, gather more evidence or downgrade the claim and continue.
8. If the attempt fails guardrails or does not improve the objective, revert/reset it when possible, record the discard, and continue.
9. If the attempt improves the objective while respecting guardrails, keep it as the new current best and continue.
10. Run `detect_hallucination` on the cited summary of the current best result.
11. If flagged, revise or gather more evidence and continue.

## Stop conditions
Stop only when ALL of the following are true:
- a baseline exists
- the current best retained attempt is stored with supporting evidence
- objective-critical claims are not flagged on the current evidence
- guardrails have been checked or residual risk is explicitly bounded
- the final cited summary is not flagged by `detect_hallucination`, or flagged text has been removed/downgraded
If any item is false, continue iterating.

## Output format
### Run state
- run_id / run_dir
- current git state if available

### Objective card
- objective, evaluator, direction, guardrails, allowed levers, stop rule

### Evidence ledger summary
- Key spans and what each one proved.

### Attempt ledger summary
- Major experiments with keep / discard / revert / crash / continue decisions.

### Hypotheses considered
- H1, H2, ... with status: refuted / plausible / retained.

### Current best result
- Baseline vs best retained attempt, with citations.

### Kept changes / rejected changes
- What was advanced, what was reverted, and why.

### Remaining risks / next hypotheses
- Explicit unresolved bottlenecks and the next experiment to run.

### Verification
- `audit_trace_budget` summary for the objective claims.
- `detect_hallucination` summary for the final cited result summary.

---

## Worked example: reduce p95 latency without changing behavior

### Objective
Reduce endpoint `POST /events` p95 latency while keeping the response schema and success rate unchanged.

### Evidence to capture
- benchmark baseline output
- endpoint handler code span
- DB query plan or profiling output
- response schema contract
- post-change benchmark output

### Attempt pattern
1. Baseline benchmark shows p95 = 180 ms.
2. Hypothesis: synchronous dedup query dominates latency.
3. Smallest next experiment: move dedup write behind a queue-like in-process worker while keeping the same response contract.
4. Store benchmark output and diff as spans.
5. Record attempt row with objective metric `p95_ms`, baseline `180`, attempt `125`, decision `keep`.
6. Audit claims such as “the kept change reduces p95 latency” and “response schema is unchanged.”
7. Run `detect_hallucination` on the cited result summary.
8. Continue until no higher-leverage bottleneck remains or the stop rule is satisfied.
