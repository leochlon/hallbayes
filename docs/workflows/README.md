# Workflow verification playbooks

These playbooks show how to use Strawberry (Berry's hallucination detector) as a verification step inside a real loop, not just as a final decoration.

## Shared loop primitives

All workflow skills share the same skeleton:

1. **Start or load a run**
2. **Store evidence as real spans**
3. **Read the persisted ledgers** (`evidence.tsv`, `attempts.tsv`)
4. **Pick one unresolved claim / bottleneck / assumption**
5. **Run the smallest next evidence action or experiment**
6. **Record an attempt row**
7. **Audit the affected claims with `audit_trace_budget`**
8. **Run `detect_hallucination` on the cited final synthesis when applicable**
9. **Keep / downgrade / revert / continue**
10. **Repeat until explicit stop conditions hold**

Progress is externalized, failed attempts leave a trace, and the next step depends on real output rather than narrative.

## Client adherence

These playbooks are written as skills: start run, gather smallest evidence, record attempt, audit, revise/continue.

MCP clients vary in how strictly they follow that sequence.

- **Codex**: best adherence. Follows the skill end-to-end without deviating.
- **Claude**: start in `/plan` mode and ask it to plan the exact workflow skill, then execute step-by-step. Require the Strawberry tool call before the final answer.
- **Other clients**: may skip tool calls, drop citations, or drift into speculative answers. Pin the "Copy/paste prompt" block from each playbook as a system instruction.

**Claude `/plan` starter (copy/paste):**
```
/plan
Create a plan to execute the Strawberry-assisted workflow skill I'm using (e.g., `rca-fix-agent`, `search-and-learn`, etc.).
Your plan must include: (1) run start/load, (2) evidence collection steps and stored spans, (3) attempt-ledger updates, (4) a Strawberry verifier tool call, (5) a revision/back-edge if anything is flagged, and (6) explicit stop conditions.
Then execute the plan step-by-step, and do not produce a final answer until the verifier has run.

If you don't have direct access to collect evidence (repo browsing, web search, or running experiments), your plan must explicitly stop and ask the user to paste the missing spans before you proceed.
```

**Core verifier tools:**
- `audit_trace_budget` -- verify a cited reasoning trace (claims + cites)
- `detect_hallucination` -- verify a cited natural-language synthesis against spans

---

## Pick your workflow

1) **Search & Learn** -- `SEARCH_AND_LEARN_VERIFICATION.md`
   Q&A / repo exploration / API understanding. Uses persisted ledgers plus `audit_trace_budget` and `detect_hallucination`.

2) **Generate Boilerplate/Content** -- `GENERATE_BOILERPLATE_VERIFICATION.md`
   Tests/docs/migrations/configs. Uses `audit_trace_budget` on the trace to verify constraints and decisions.

3) **Inline Completions** -- `INLINE_COMPLETION_VERIFICATION.md`
   Spot-check high-impact tab-complete. Uses `audit_trace_budget` on a 3-6 step micro-trace.

4) **Greenfield Prototyping** -- `GREENFIELD_PROTOTYPE_VERIFICATION.md`
   Move fast with Facts vs Decisions vs Assumptions, inside a persisted evidence loop.

5) **Objective Optimization Agent** -- `OBJECTIVE_OPTIMIZATION_VERIFICATION.md`
   Program-style loop for any well-defined objective: baseline, hypothesis, smallest experiment, measurement, keep/revert, verify.

6) **Plan and Execute** -- `PLAN_AND_EXECUTE_VERIFICATION.md`
   Repo understanding + verified planning loop + post-approval execution loop.

7) **RCA Fix Agent** -- MCP prompt `rca_fix_agent` + `RCA_FIX_REPORT_TEMPLATE.md`
   Full debugging loop: baseline, hypotheses, experiments, keep/revert decisions, verify ROOT_CAUSE/FIX_MECHANISM/FIX_VERIFIED/NO_NEW_FAILURES.

---

## Worked examples
Each playbook includes a before/after comparison:
- **Without Strawberry** -- a plausible, confident answer that is easy to hallucinate
- **With Strawberry** -- persisted evidence, attempt rows, verifier calls, and an answer you can inspect
