# Search & Learn Verification Skill

Use this when answering questions about unfamiliar code, APIs, configs, logs, or behavior.

**Goal:** resolve the question through a persisted evidence loop, not a plausible narrative.

The loop:
1. Start or load a run.
2. Store real evidence as spans.
3. Pick one unresolved claim.
4. Gather the smallest next evidence item or experiment.
5. Record the attempt.
6. Audit the claim trace.
7. Run `detect_hallucination` on the cited answer draft.
8. Repeat until answer-critical claims are supported or explicit unknowns.

---

## Mandatory external state
- Start or load a run before substantive work.
- Store all evidence as real spans in the active run.
- Treat `run.json`, `evidence.tsv`, and `attempts.tsv` as working memory.
- Re-read `list_spans` and `list_attempts` before each iteration.
- Record every evidence move, experiment, audit result, and downgrade as an attempt row.

## Invalid states
- Final answer contains factual claims not grounded in current spans.
- `audit_trace_budget` is re-run with no new evidence and no claim downgrade.
- A cited span id does not exist in the active run.
- The agent stops because it feels stuck before re-reading the ledgers and choosing the next smallest action.
- A guess is presented as fact instead of **Unknown** or **Assumption**.

## Minimum verified claims
Drive these to supported or explicit unknown status:
- **QUESTION_INTERPRETATION** — what the user is actually asking.
- **PRIMARY_ANSWER** — the main answer.
- **KEY_SUPPORT** — the most important supporting details, locations, commands, or caveats.
- **OPEN_GAPS** — remaining unknowns are explicit and paired with the next evidence to gather.

## Question-resolution loop
LOOP UNTIL THE STOP CONDITIONS ARE TRUE:
1. Read the current evidence ledger and attempt ledger.
2. Choose ONE unresolved claim or subquestion.
3. Gather the SMALLEST next evidence item or experiment for that claim.
4. Store the output as new spans.
5. Record an attempt row with claim id, action, supporting spans, audit status, result summary, and next step.
6. Run `audit_trace_budget` on the affected claim trace.
7. If flagged, gather more evidence or downgrade the claim and continue.
8. When the cited answer draft is ready, run `detect_hallucination` on that cited draft against the current spans.
9. If flagged, revise or gather more evidence and continue.

## Behavior rules
- Prefer the smallest discriminating action over broad unfocused search.
- Repo source, official docs, and experiment output beat paraphrase.
- If you run a command or experiment, store the output as a span.
- Do not ask whether to continue unless blocked by permissions, missing access, or an explicit user stop.
- If the same claim has been audited 3 times without progress, return the passed claims plus the unresolved gaps and the next evidence to gather.

## Stop conditions
Stop only when ALL of the following are true:
- answer-critical claims are supported by current evidence or explicitly marked Unknown
- the final cited answer is not flagged by `detect_hallucination`, or flagged claims have been removed/downgraded
- remaining gaps are explicit and paired with the next evidence to collect
If any item is false, continue iterating.

## Output format
### Run state
- run_id / run_dir

### Evidence ledger summary
- Key spans and what each one proved.

### Attempt ledger summary
- Major claim-resolution attempts and their decisions.

### Supported answer (cited)
- Short factual sentences, each cited.

### Unknowns / assumptions
- Explicitly separate what is not yet proven.

### Next evidence to gather
- Exact file paths, commands, or URLs that would close remaining gaps.

### Verification
- `audit_trace_budget` summary for the key claims.
- `detect_hallucination` summary for the final cited answer.

---

## Worked example: “Does auth validate JWT aud+iss?”

### Without Strawberry
**User:** Does this repo’s auth validate JWT `aud` and `iss`? Where is it configured?

**Assistant (uncited):**
- “Yes — the middleware validates issuer and audience via `jsonwebtoken.verify()` using `JWT_ISSUER` and `JWT_AUDIENCE` env vars.”
- “It’s configured in `config/auth.ts` and loaded from `.env`.”
- “To disable audience validation, set `audience: undefined`.”

### With Strawberry

#### 1) Start the run and collect baseline spans
Create a run, then store the relevant middleware/config/doc excerpts as spans.

**S0 — middleware code**
```text
// src/auth/middleware.ts
export function auth(req, res, next) {
  const token = getBearer(req);
  const { payload } = jwtVerify(token, getKey(), {
    issuer: process.env.JWT_ISSUER,
  });
  req.user = payload.sub;
  next();
}
```

**S1 — configuration docs**
```text
# README.md
JWT_ISSUER is required. No other JWT settings are currently supported.
```

**S2 — env example**
```text
# .env.example
JWT_ISSUER=
```

#### 2) Record the claim-resolution attempts
- `QUESTION_INTERPRETATION`: determine whether the user is asking about both issuer and audience validation.
- `PRIMARY_ANSWER`: inspect the middleware and config evidence.
- `OPEN_GAPS`: identify what additional span would be needed if audience validation exists elsewhere.

#### 3) Audit the key claims
Use `audit_trace_budget` on a short trace such as:
- issuer is validated via the `issuer` option in `jwtVerify` `[S0]`
- `JWT_ISSUER` is the configured source `[S0][S1][S2]`
- audience validation is not evidenced in the provided middleware `[S0]`

#### 4) Run `detect_hallucination` on the cited answer draft
Draft:
- Issuer is validated and configured via `JWT_ISSUER`. [S0][S1][S2]
- Audience (`aud`) validation is **not evidenced** in the provided spans. [S0]
- If audience validation exists elsewhere, it is not shown here. [S0]

#### 5) Final answer
Issuer is validated and configured via `JWT_ISSUER`. [S0][S1][S2]
Audience (`aud`) validation is **not evidenced** in the provided spans. [S0]
If audience validation exists elsewhere, the next evidence to collect is any additional auth wrapper or token verification helper that calls `jwtVerify(...)`. [S0]
