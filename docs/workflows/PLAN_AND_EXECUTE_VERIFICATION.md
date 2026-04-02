# Plan and Execute Verification Skill

Use this when you need a verified change plan and, after approval, an autonomous execution loop that keeps gathering evidence, testing, and revising until the stop conditions are true.

**Goal:** reproduce the good properties of `program.md` for repo changes:
- evidence must be persisted, not narrated
- each iteration must leave behind machine-readable state
- every failed attempt must drive a back-edge into more evidence gathering or a revert
- after approval, the agent should continue the loop without asking whether to continue

---

## Mandatory external state

Before substantive work:
1. Start or load a run.
2. Store evidence as real spans in the active run.
3. Use the run directory as working memory:
   - `run.json`
   - `evidence.tsv`
   - `attempts.tsv`
4. Re-read `list_spans` and `list_attempts` before each iteration.
5. Record every planning iteration and every execution attempt in the attempt ledger.

**Rule:** if evidence is not in the run, it does not count.

---

## Invalid states

These are workflow failures, not stylistic issues:
- returning plan steps that are unsupported by current spans unless they are explicitly marked **Unknown** or **Assumption**
- re-running `audit_trace_budget` with no new evidence and no downgraded claim
- executing edits before approval
- stopping because “I’m stuck” before re-reading the evidence and attempt ledgers and choosing the next smallest unresolved claim
- keeping a failed patch when git can revert/reset it
- presenting a completed change without stored test output and a passing audit on the changed claims

---

## Loop A — Plan-critical evidence gathering (before approval)

LOOP UNTIL EVERY PLAN-CRITICAL CLAIM IS SUPPORTED OR EXPLICITLY UNKNOWN:

1. Read the evidence ledger and attempt ledger.
2. Pick **one** unresolved claim that materially affects:
   - files to change
   - tests to run
   - rollout / migration safety
   - rollback behavior
3. Gather the smallest next evidence item for that claim:
   - repo excerpt
   - config / schema / contract
   - failing test output
   - command output
   - documentation snippet
4. Update:
   - **Facts (cited)**
   - **Decisions**
   - **Assumptions / Unknowns**
   - the plan-step trace
5. Run `audit_trace_budget` on the affected plan claims.
6. Record an attempt row with:
   - `claim_id`
   - `action`
   - supporting/input spans
   - output spans
   - audit result
   - next evidence to gather
7. If a claim is flagged, gather more evidence or downgrade the step before continuing.

Ask for approval **exactly once**, only after the plan-critical claims have been pushed to **supported** or explicit **unknown** status.

---

## Loop B — Execute (only after approval)

After approval, the loop becomes autonomous.

**Do not ask whether to continue** unless blocked by permissions, missing execution capability, destructive ambiguity, or an explicit user stop.

LOOP UNTIL THE STOP CONDITIONS ARE TRUE:

1. Re-read the evidence ledger, attempt ledger, and current git state.
2. Choose the smallest next patch mapped to **one** cited plan step.
3. Apply the patch and note the starting git state first.
4. Run the smallest discriminating unit / integration test or validation command for that step.
   - Prefer checks that finish within ~5 minutes.
   - If not possible, use the smallest faster slice and record why.
5. Store **all** outputs as spans.
6. Record an attempt row with:
   - `keep` / `revert` / `continue`
   - the relevant spans
   - the next step
7. Audit the changed claims with `audit_trace_budget`.
8. If tests fail or evidence contradicts the step:
   - revert/reset that patch
   - go back to **Loop A** with a revised claim or hypothesis
9. If the change advances toward the deliverable, keep it and continue the loop.

**Negative test results stay in the ledger.** They are evidence.

---

## Behavior rules

- Facts are proven constraints only.
- Decisions are chosen approaches.
- Assumptions / Unknowns are open gaps.
- Never blur those categories.
- Every executed patch must map back to a cited plan step.
- If you get stuck, read the ledgers and pick the next smallest unresolved claim instead of asking to stop.

---

## Stop conditions

Stop only when **all** of the following are true:
- deliverable-critical plan claims are supported by current evidence
- approved edits have been implemented or explicitly deferred
- planned unit / integration tests or equivalent validations have been run and their outputs are stored as spans
- changed claims are not flagged by `audit_trace_budget`
- any remaining unknowns are explicit and bounded

If any item is false, continue iterating.

---

## Output format

### Run state
- run_id / run_dir
- approval status
- current git state if available

### Evidence ledger summary
- Key spans and what each one proves.

### Attempt ledger summary
- Major planning / execution attempts with keep / revert / continue decisions.

### Repo understanding (cited)
- Short cited summary of relevant architecture and constraints.

### Facts (cited)
- Only proven constraints / requirements.

### Decisions
- Chosen approach and tradeoffs.

### Assumptions / unknowns
- Open gaps that remain explicit.

### Plan (exact file changes)
- Step-by-step plan including file paths, unit tests, integration tests, and rollback notes.

### Approval status
- If approval is not yet granted, ask for it here and stop before execution.
- If approval is granted, report execution progress and current keep / revert decisions instead of asking again.

### Verification trace (JSON)
- JSON array of `{idx, claim, cites}` for the active plan or executed changes.

### Audit result
- Paste the `audit_trace_budget` summary + any flagged claims.
