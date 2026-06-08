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
- `export_run_ledger(run_id?)` — regenerate full JSON/TSV inspection exports from the authoritative SQLite ledger.

#### Evidence spans

- `add_span(text, source?, run_id?, meta?, kind?, source_type?, trust?, sensitivity?, tags?)` — add immutable text evidence to the run ledger.
- `add_file_span(path, start_line, end_line, source?, run_id?, meta?, read_mode?, kind?, trust?, sensitivity?, tags?)` — capture lines from a local file as evidence with path, byte-offset, file-hash, git, and trust provenance. `read_mode` defaults to `baseline` and falls back to worktree with explicit provenance when no baseline object is available.
- `list_spans(run_id?, limit?, kinds?, source_types?, trust?, status?)` — list span metadata, including `eid`, `kind`, `trust`, `status`, `sensitivity`, hashes, locator, parents, and tags.
- `get_span(sid, run_id?, include_sensitive_text?)` — fetch a span. Sensitive text is redacted unless explicitly requested.
- `mark_span(sid, run_id?, status?, sensitivity?, tags_add?, tags_remove?, trust?)` — update mutable annotations without mutating span text/provenance.
- `search_spans(query, run_id?, limit?)` — compatibility lexical search over active non-derived spans.
- `query_evidence(query, run_id?, limit?, kinds?, source_types?, trust?, status?, include_derived?, include_stale?)` — filterable evidence retrieval.
- `extract_span(parent_sid, selector, run_id?, reason?, source?, max_lines?, tags?)` — create an offset-preserving derived span from a regex or line-range selector. No-match creates no evidence span.
- `distill_span(parent_sid, pattern, run_id?, source?, flags?, max_lines?)` — compatibility wrapper around regex extraction.
- `get_evidence_pack(sids, run_id?, max_chars?, allow_sensitive?, include_stale?, allow_untrusted?)` — resolve run-owned SIDs into the exact verifier-safe evidence pack.

#### Attempt ledger

- `record_attempt(...)` — append a structured attempt row for the current run.
  - Supports claim/action basics plus optional `git_state`, `objective_metric`, `objective_value`, and `result_summary` fields.
- `list_attempts(run_id?, limit?)` — list recorded attempts for the active run.

Berry persists the authoritative run state in `~/.berry/runs/<run_id>/run.sqlite`. The SQLite ledger uses normalized runs/spans/attempts/claims/links/audits tables plus a tamper-evident `ledger_events` table. Hot writes are incremental: only changed rows are upserted, each row carries a payload hash, and each commit appends a hash-chained state event.

Full JSON/TSV inspection exports are no longer regenerated on every span append by default. Use `export_run_ledger(run_id?)` or set `BERRY_LEDGER_EXPORT_MODE=sync` to write `run.json`, `evidence.tsv`, `attempts.tsv`, `claims.tsv`, `claim_evidence.tsv`, `audits.tsv`, and `ledger_events.jsonl` for legacy consumers. `BERRY_LEDGER_EXPORT_MODE=hot` writes lightweight head / append-only mirrors on each commit; the default is `off`. Legacy JSON-only runs still load and migrate into SQLite.

See `docs/SPANS.md` for the full span schema, evidence-pack policy, and trust model.

#### Claim/evidence graph

- `create_claim(text, run_id?, kind?, target?, status?, source?, tags?, meta?)` — create a structured claim node.
- `link_claim_evidence(cid, sid, run_id?, relation?, note?, audit_id?, meta?)` — attach a typed evidence edge. `supports` / `contradicts` require citable evidence spans; `background` may point at anchors.
- `list_claim_evidence(run_id?, cid?, sid?, relation?)` — list graph edges by claim, span, or relation.
- `list_claims(run_id?, limit?, status?, kinds?, include_evidence?)` — list claims and their evidence edges.
- `get_claim(cid, run_id?, include_evidence?, include_audits?)` — fetch one claim with graph context.
- `mark_claim(cid, run_id?, status?, kind?, target?, latest_audit_id?, tags_add?, tags_remove?, meta_update?, reason?)` — update mutable claim annotations without changing claim text.
- `audit_claims(claim_ids?, run_id?, evidence_relations?, verifier_model?, context_mode?, include_prompts?, max_prompt_chars?, top_logprobs?, min_log_odds_gain?, use_cache?, group_claims?, max_group_size?, max_group_prompt_chars?, timeout_s?, max_evidence_chars?, allow_sensitive?, include_stale?, allow_untrusted?, max_claims?, record_audit?)` — build verifier steps from graph-linked claims and evidence, then update claim statuses and audit-linked edges.
- `list_audits(run_id?, claim_id?, limit?)` — list verifier audit records.

`audit_claims` uses claim targets as verifier confidence thresholds, resolves all cited spans server-side, records a non-citable audit span, records an `AuditRecord`, and updates each audited claim to `supported`, `contradicted`, or `insufficient` according to the verifier detail status.

#### Verification

Prefer the run-scoped tools:

- `detect_hallucination_run(answer, run_id?, sid_whitelist?, verifier_model?, default_target?, max_claims?, claim_split?, require_citations?, context_mode?, include_prompts?, max_prompt_chars?, top_logprobs?, min_log_odds_gain?, use_cache?, group_claims?, max_group_size?, max_group_prompt_chars?, timeout_s?, max_evidence_chars?, allow_sensitive?, include_stale?, allow_untrusted?, record_audit?, record_claim_graph?)` — detect answer hallucinations by resolving `[S#]` citations from the server-owned run ledger.
- `audit_trace_budget_run(steps, run_id?, sid_whitelist?, verifier_model?, default_target?, require_citations?, context_mode?, include_prompts?, max_prompt_chars?, top_logprobs?, min_log_odds_gain?, use_cache?, group_claims?, max_group_size?, max_group_prompt_chars?, timeout_s?, max_evidence_chars?, allow_sensitive?, include_stale?, allow_untrusted?, record_audit?, record_claim_graph?, auto_create_claims?)` — audit explicit `(claim, cites)` steps using server-resolved evidence. Steps may include `claim_id` / `cid`; when present, Berry updates the corresponding claim graph node. `auto_create_claims=true` creates claim nodes for anonymous steps.

The run-scoped tools fail closed when citations reference unknown, stale, sensitive, non-citable, or over-budget spans. Reports include an `evidence_pack` summary containing `pack_id`, `text_sha256`, input SIDs, materialized SIDs, exclusions, truncation state, and policy. When `record_audit=true`, Berry records a non-citable `kind=audit` span summarizing the verifier call.

They use the hardened v2 verifier underneath: `context_mode="cited"` by default, target-directed posterior checks, prior-leak detection, deterministic preflight statuses, verifier caching, and grouped multi-claim prompts with safe fallback to single-claim prompts.

Legacy raw-span tools remain available for compatibility:

- `detect_hallucination(answer, spans, verifier_model?, default_target?, max_claims?, claim_split?, require_citations?, context_mode?, include_prompts?, max_prompt_chars?, top_logprobs?, min_log_odds_gain?, use_cache?, group_claims?, max_group_size?, max_group_prompt_chars?)` — information-budget diagnostic per claim using caller-supplied span text.
- `audit_trace_budget(steps, spans, verifier_model?, default_target?, require_citations?, context_mode?, include_prompts?, max_prompt_chars?, top_logprobs?, min_log_odds_gain?, use_cache?, group_claims?, max_group_size?, max_group_prompt_chars?)` — score explicit `(claim, cites)` steps using caller-supplied span text.

The legacy tools are lower-trust because the caller supplies the evidence text. New workflows should use the `_run` variants.


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
