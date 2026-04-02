# RCA + Fix Loop Report

This report is the output of a loop, not a substitute for the loop.

Use it only after running an evidence-backed RCA cycle with persisted spans and recorded attempts.

---

## Mandatory loop contract

Before claiming root cause or fix success:
1. Start or load a run.
2. Store all evidence as real spans.
3. Use the run directory as working memory:
   - `run.json`
   - `evidence.tsv`
   - `attempts.tsv`
4. Re-read `list_spans` and `list_attempts` before each iteration.
5. Record every experiment, patch attempt, audit result, and revert decision in the attempt ledger.

**Rule:** if evidence is not in the run, it does not count.

---

## Invalid states

These are workflow failures:
- declaring root cause without a passing ROOT_CAUSE audit on the current spans
- re-running `audit_trace_budget` with no new evidence and no claim downgrade
- omitting contradictory or negative experiment output
- stopping because you are stuck before re-reading the ledgers and choosing the next smallest discriminating action
- keeping a failed code change when git can revert/reset it

---

## RCA + Fix loop

LOOP UNTIL THE STOP CONDITIONS ARE TRUE:

1. Read the evidence ledger and attempt ledger.
2. Choose **one** unresolved claim or hypothesis that materially affects the fix.
3. Plan the smallest next action that could change the state of that claim.
   - inspect one file
   - run one repro slice
   - add one targeted patch
   - run one targeted test
4. Prefer actions that finish within ~5 minutes.
   - If not possible, choose the smallest faster slice and record why.
5. Run the action and store **all** outputs as spans.
6. Record an attempt row with:
   - `claim_id`
   - hypothesis
   - action
   - budget
   - input spans
   - output spans
   - audit status
   - keep / revert / continue decision
   - next step
7. Audit the affected claim trace with `audit_trace_budget`.
8. If flagged, gather more evidence or downgrade the claim and continue.
9. If a code change was made, keep it only if the evidence advances toward the stop conditions; otherwise revert/reset and continue.

After the loop starts, do **not** ask whether to continue unless blocked by permissions, missing execution capability, destructive ambiguity, or an explicit user stop.

---

## Stop conditions

Stop only when **all** of the following are true:
- the original repro now passes
- regression checks have run and their outcomes are stored as spans
- ROOT_CAUSE, FIX_MECHANISM, FIX_VERIFIED, and NO_NEW_FAILURES are not flagged on the current evidence
- any remaining uncertainty is explicitly marked as bounded risk or follow-up

If any item is false, continue iterating.

---

## Report

## Run state
- run_id:
- run_dir:
- current git state / baseline commit (if available):

## Problem statement
- What is broken?
- Expected vs observed behavior.

## Baseline repro
- Repro command(s):
- Baseline result span(s):
- Closest implicated code / config / docs span(s):

## Evidence ledger summary
> Summarize the key spans you used and what each one proved.

## Attempt ledger summary
> Summarize each major attempt with keep / revert / continue decisions.

## Hypotheses considered
- H1: ... (status: refuted / plausible / confirmed)
  - For:
  - Against:
  - Experiment(s):
- H2: ...

## Root cause (verified)
- **PRIMARY CLAIM:** The issue is because of ROOT_CAUSE. [S?]
- Supporting sub-claims:
  - ... [S?]

## Fix mechanism (verified)
- Intended change:
- Files changed:
- Why this should fix it (mechanism): [S?]

## Patch summary
- What changed (high level):
- Keep / revert decision history:

## Test plan
### Tests executed
- ... [S?]

### Results
- Original repro now passes: [S?]
- Regression suite passes: [S?]

## Additional failure modes considered
- FM1: ...
  - Check run: ... [S?]
  - Status: mitigated / unknown

## Remaining uncertainties / follow-ups
- Explicit bounded risks only.

## Verification
- `audit_trace_budget` summary:
- Flagged claims that were downgraded or left open:
