# Greenfield Prototyping Verification Skill

Use this when prototyping with incomplete requirements.

**Goal:** move fast without pretending assumptions are facts.

The loop:
1. Start or load a run.
2. Store real requirements, constraints, repo context, benchmarks, and spike output as spans.
3. Pick one unresolved fact or one high-impact assumption.
4. Gather the smallest next evidence item or spike.
5. Record the attempt.
6. Audit the Facts trace.
7. Run `detect_hallucination` on the cited prototype summary.
8. Repeat until prototype-critical facts are supported or explicit assumptions with validation paths.

---

## Mandatory external state
- Start or load a run before substantive work.
- Store all requirements, constraints, repo context, benchmarks, and spike output as real spans.
- Treat `run.json`, `evidence.tsv`, and `attempts.tsv` as working memory.
- Re-read `list_spans` and `list_attempts` before each iteration.
- Record every evidence move, spike, benchmark, architectural hypothesis, and downgrade as an attempt row.

## Invalid states
- Putting a claim into **Facts** without current span support.
- Re-running `audit_trace_budget` with no new evidence and no fact downgrade.
- Treating an architecture choice as a fact when it is really a Decision or Assumption.
- Stopping because you feel stuck before re-reading the ledgers and choosing the next smallest evidence action or spike.
- Shipping a prototype plan where material unknowns have no validation path.

## Minimum verified claims
Drive these to supported or explicit assumption status:
- **PROTOTYPE_GOAL** — what is being built and what “good enough” means.
- **FACTS** — hard requirements, constraints, and environmental truths.
- **DECISION_SUPPORT** — high-impact design decisions are either constrained by facts or clearly labeled as choices.
- **OPEN_ASSUMPTIONS** — major unknowns remain explicit and paired with the quickest validation experiment.

## Prototype-evidence loop
LOOP UNTIL THE STOP CONDITIONS ARE TRUE:
1. Read the current evidence ledger and attempt ledger.
2. Choose ONE unresolved fact or ONE high-impact assumption.
3. Gather the SMALLEST next evidence action or spike.
4. Store the output as new spans.
5. Record an attempt row with claim id, action, supporting spans, audit status, result summary, and next step.
6. Run `audit_trace_budget` on the affected Facts trace.
7. If flagged, gather more evidence or demote the claim to **Assumption** and continue.
8. When the prototype summary is ready, run `detect_hallucination` on the cited **Facts + Decisions + Assumptions** summary.
9. If flagged, revise or gather more evidence and continue.

## Behavior rules
- Facts are proven constraints only.
- Decisions are chosen approaches; they can be motivated by evidence without becoming facts.
- Assumptions are allowed, but every material assumption must have a validation path.
- A spike or benchmark is a hypothesis test; store the output even when it refutes your preferred design.
- Do not ask whether to continue unless blocked by permissions, missing context, or an explicit user stop.

## Stop conditions
Stop only when ALL of the following are true:
- prototype-critical facts are supported by current evidence or explicitly labeled Assumption/Unknown
- high-impact assumptions have a concrete validation experiment or evidence source
- the final cited prototype summary is not flagged by `detect_hallucination`, or flagged text has been removed/downgraded
- remaining risk is explicit and bounded
If any item is false, continue iterating.

## Output format
### Run state
- run_id / run_dir

### Evidence ledger summary
- Key spans and what each one proved.

### Attempt ledger summary
- Major evidence/spike attempts and their decisions.

### Goal
- What are we building and why?

### Facts (cited)
- Only proven requirements and constraints.

### Decisions
- Chosen architecture, interface, or implementation moves.

### Assumptions / unknowns
- Explicit unresolved items plus the next validation step.

### Prototype plan
- Minimal milestones, file areas, or components to build.

### Validation plan
- Fastest experiment or evidence source for each material assumption.

### Verification
- `audit_trace_budget` summary for the Facts trace.
- `detect_hallucination` summary for the final cited prototype summary.

---

## Worked example: prototype an events ingestion API

### Without Strawberry
**User:** Prototype an events ingestion API.

**Assistant (uncited):**
- “Exactly-once delivery”
- “Kafka pipeline”
- “99p < 50ms”
- “GDPR compliant by default”
- “RBAC and audit logs”

### With Strawberry

#### 1) Evidence pack
**S0 — requirements**
```text
We need POST /events. Return 202 quickly. Store events in Postgres.
```

**S1 — constraint**
```text
No message broker in v1. Deploy as a single service.
```

**S2 — privacy**
```text
Events may contain PII. Must support deletion by user_id.
```

#### 2) Record attempts against prototype-critical claims
- `PROTOTYPE_GOAL`: clarify endpoint behavior and “quickly” semantics.
- `FACTS`: verify Postgres storage, no broker in v1, single-service deployment, and deletion by `user_id`.
- `OPEN_ASSUMPTIONS`: throughput target and latency SLO remain assumptions until measured or provided.

#### 3) Facts / Decisions / Assumptions
**Facts**
- The API must expose `POST /events` and return `202` quickly. [S0]
- Events are stored in Postgres. [S0]
- v1 must not use a message broker and is deployed as a single service. [S1]
- Events may contain PII and deletion by `user_id` must be supported. [S2]

**Decisions**
- Use an `events_raw` table and a background worker inside the same service. [S0][S1]
- Add request validation and schema versioning. (Decision motivated by S0, not a fact.) [S0]

**Assumptions**
- Throughput target: 1k req/s. (Assumption)
- Exactly-once semantics. (Assumption)
- Latency SLO. (Assumption)

#### 4) Audit + hallucination pass
- Run `audit_trace_budget` on the Facts trace.
- Run `detect_hallucination` on the cited prototype summary.
- If throughput or latency statements are flagged, keep them in Assumptions until you have real evidence.
