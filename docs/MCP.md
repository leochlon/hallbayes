# Berry MCP server

Berry ships a single MCP surface: **classic**.

Run it via:

```bash
berry mcp --server classic
```

Older configs may still pass `--server science`/`--server forge`; those values are treated as aliases for `classic` for compatibility.

## What Berry exposes to MCP clients

### Tools

Berry exposes a focused set of tools for evidence collection, attempt ledgers, and hallucination detection.

#### Run management

- `start_run(problem_statement, deliverable, run_id?)` — create a new run directory with a problem statement + immutable deliverable anchor.
- `load_run(run_id)` — resume an existing run (loads from disk if necessary) and set it active.
- `get_deliverable(run_id?)` — get the immutable deliverable anchor for the active run.

#### Evidence spans

- `add_span(text, source?, run_id?, meta?)` — add evidence from text.
- `add_file_span(path, start_line, end_line, source?, run_id?, meta?)` — capture lines from a local file (path + line range) as evidence.
- `list_spans(run_id?, limit?)` — list all spans (metadata only).
- `get_span(sid, run_id?)` — fetch full span text.
- `search_spans(query, run_id?, limit?)` — search over span texts (lightweight token match scoring).
- `distill_span(parent_sid, pattern, run_id?, source?, flags?, max_lines?)` — extract key lines from a large span (regex-based), creating a new span.

#### Attempt ledger

- `record_attempt(...)` — append a structured attempt row for the current run.
  - Supports claim/action basics plus optional `git_state`, `objective_metric`, `objective_value`, and `result_summary` fields.
- `list_attempts(run_id?, limit?)` — list recorded attempts for the active run.

Berry persists the run state in `~/.berry/runs/<run_id>/` and also writes machine-readable ledgers:
- `run.json`
- `evidence.tsv`
- `attempts.tsv`

#### Verification

- `detect_hallucination(answer, spans, verifier_model?, default_target?, max_claims?, claim_split?, require_citations?, context_mode?, include_prompts?, max_prompt_chars?, top_logprobs?, min_log_odds_gain?, use_cache?, group_claims?, max_group_size?, max_group_prompt_chars?)` — information-budget diagnostic per claim.
  - `require_citations=true` will flag claims that have no citations even if they could be supported by the overall context.
  - `context_mode="cited"` is the default and verifies each claim only against its cited spans.
  - `group_claims=true` is the default. Claims with the same verifier context are scored in one multi-claim prompt using one Y/N/U answer line per claim. If the grouped output cannot be parsed into exactly one answer distribution per claim, Berry automatically retries that group with legacy single-claim prompts.
  - `max_group_size` defaults to `8`; `max_group_prompt_chars` defaults to `24000` and causes oversized chunks to split or fall back to single prompts.
  - `include_prompts=true` returns the exact verifier prompt used. For grouped claims, this is the grouped prompt shared by the claims in that verifier call.
- `audit_trace_budget(steps, spans, verifier_model?, default_target?, require_citations?, context_mode?, include_prompts?, max_prompt_chars?, top_logprobs?, min_log_odds_gain?, use_cache?, group_claims?, max_group_size?, max_group_prompt_chars?)` — score explicit (claim, cites) steps.
  - The verifier is target-directed: cited context must push the posterior YES probability to `default_target`, and redacted context must not already support the claim.
  - Deterministic preflight statuses such as `missing_citations`, `unknown_citations`, `empty_context`, and `no_spans` are returned without calling the verifier.
  - Responses include both logical `verifier_calls` and estimated physical `verifier_api_calls_planned`; the latter reflects grouped prompt fanout and grouped-fallback retries.

### Prompts (workflows)

> **Client adherence note:** Prompt/skill support varies across MCP clients.
> - **Codex** is the most reliable at following workflow prompts end-to-end (citations + required Strawberry verifier tool calls + ledgered looping).
> - In **Claude**, using **`/plan` mode** and asking it to produce a plan for the workflow skill (then executing that plan) makes it much more likely to stay on-plan and run the verifier autonomously.
> - Other clients may treat prompts as suggestions.
>
> If you see drift, pin the playbook prompt text as a system instruction and insist the verifier tools are called before final answers.

- Search & Learn (loop-enforced)
- Generate Boilerplate/Content (verified)
- Inline completion guard (verified)
- Greenfield prototyping (loop-enforced)
- Objective Optimization Agent
- Plan and Execute (verified)
- **RCA Fix Agent** — full debugging loop with evidence-backed root cause, verified fix, and test plan

## Transports

Default transport is `stdio`.

Optional transports:

```bash
berry mcp --transport sse --host 127.0.0.1 --port 8000
berry mcp --transport streamable-http --host 127.0.0.1 --port 8000
```

## Project root resolution

By default, `berry mcp` uses `--project-root` if provided; otherwise it walks up from the current working
directory to find a `.git` directory and uses that as the project root. If no `.git` is found, Berry fails
closed unless you set `BERRY_ALLOW_NON_GIT_ROOT=1` (which treats the current directory as the project scope).

## Common errors

- If you see `MCP SDK not installed`, install `mcp[cli]` (it's a dependency of Berry but can be missing in some dev setups).
