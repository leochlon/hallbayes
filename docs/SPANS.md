# Span management v2

Berry spans are the local evidence substrate for verification. In v2, a span is no longer just a text snippet. It is a typed, immutable, provenance-bearing evidence record owned by the Berry server.

## Core model

Every span keeps the v1 fields for compatibility:

```text
sid, text, source, created_at, meta
```

It now also carries first-class policy and provenance fields:

```text
eid, kind, source_type, media_type, text_sha256
locator, snapshot, parents, transform
trust, status, sensitivity, tags
```

Important fields:

- `sid` is the human-facing run-local citation ID, for example `S17`.
- `eid` is a stable hash over text/provenance/lineage metadata.
- `kind` separates `anchor`, `evidence`, `observation`, `derived`, `assumption`, `decision`, and `audit` records.
- `locator` stores where the evidence came from, such as path, line range, byte offsets, and encoding.
- `snapshot` stores source-state facts such as file hash, git commit, git status, baseline reference, and read mode.
- `parents` and `transform` preserve lineage for extracted spans.
- `status` supports `active`, `stale`, `superseded`, `tombstoned`, `redacted`, and `quarantined`.
- `sensitivity` supports `normal`, `secret`, `pii`, and `unknown`.

Span text and provenance are immutable. Mutable operations, such as `mark_span`, only change annotations like status, sensitivity, trust, and tags.

## Anchors are not evidence

`start_run` records the problem statement and deliverable as `kind=anchor`. Anchors capture task intent; they do not prove that work was done.

Server-resolved evidence packs exclude anchors by default. A final answer should cite spans that actually observe the world: file excerpts, test output, source snippets, logs, extracted lines, or other evidence/observation spans.

## Server-resolved verification

Prefer these tools over the legacy raw-span verifier entry points:

```text
detect_hallucination_run(answer, run_id?, ...)
audit_trace_budget_run(steps, run_id?, ...)
```

The run-scoped verifier tools resolve `[S#]` citations from the server-owned run ledger. The agent no longer supplies arbitrary span text to the verifier.

The server rejects evidence packs that contain:

```text
unknown SIDs
anchor / assumption / decision / audit spans
stale, tombstoned, redacted, quarantined spans
secret or PII spans unless allow_sensitive=true
untrusted/quarantined spans unless allow_untrusted=true
non-extractive derived summaries
prompt-budget-exceeded spans
```

The verifier receives only the materialized `sid/text` pairs from the evidence pack. The report includes an `evidence_pack` summary with `pack_id`, `text_sha256`, input SIDs, materialized SIDs, exclusions, truncation state, and policy.

## Derived spans

Use `extract_span` to create citable derived spans from a parent span:

```text
extract_span(parent_sid, selector={"type": "regex", "pattern": "failed|error"})
extract_span(parent_sid, selector={"type": "line_range", "start_line": 20, "end_line": 35})
```

No-match extraction creates no span. It returns `matched=false` and `sid=null`; Berry does not create `[no lines matched]` as fake evidence.

Extractive derived spans are citable because they preserve the exact parent text and matched line numbers. Abstractive summary spans are not primary evidence; if cited, the evidence pack surfaces parent spans instead and records the summary as excluded.

`distill_span` remains as a compatibility wrapper around regex extraction.

## File provenance

`add_file_span` captures:

```text
path and relative path
line range and byte offsets
encoding
file SHA-256
git commit and git status
run baseline kind/reference
read mode
```

`add_file_span` defaults to `read_mode="baseline"`. In git repos, Berry reads exact bytes from the run baseline commit when available; otherwise it falls back to the worktree and records the fallback reason. Worktree captures are marked with `trust="worktree"` so they cannot be confused with immutable baseline evidence.

If the file is modified in the working tree and the span is captured from the worktree, Berry marks its trust as `worktree`. That is useful evidence, but it should be distinguished from clean baseline evidence. Baseline captures also record `worktree_drift_from_baseline=true` when the current file has diverged from the captured baseline object.

## Persistence and ledgers

Berry persists authoritative run state under:

```text
~/.berry/runs/<run_id>/run.sqlite
```

`run.sqlite` is the source of truth. It stores normalized tables for runs, spans, attempts, claims, claim/evidence links, audits, and a tamper-evident `ledger_events` table. Hot writes are incremental: a span append upserts only the changed row, stores that row's payload hash, and appends a hash-chained `state_committed` event containing the changed row hash plus the current run metadata hash. Load reconstructs the expected table state by replaying the event chain and fails closed if row payloads, row hashes, metadata hashes, table positions, event payloads, or event-chain links do not match.

Inspection exports are deliberately separated from the hot commit path. By default, Berry does not rewrite full JSON/TSV snapshots after every span because that makes long sessions O(n²). Use the MCP tool `export_run_ledger` or set `BERRY_LEDGER_EXPORT_MODE=sync` when a legacy consumer needs:

```text
~/.berry/runs/<run_id>/run.json
~/.berry/runs/<run_id>/evidence.tsv
~/.berry/runs/<run_id>/attempts.tsv
~/.berry/runs/<run_id>/claims.tsv
~/.berry/runs/<run_id>/claim_evidence.tsv
~/.berry/runs/<run_id>/audits.tsv
~/.berry/runs/<run_id>/ledger_events.jsonl
```

`BERRY_LEDGER_EXPORT_MODE=hot` writes lightweight head / append-only mirrors on each commit. `BERRY_LEDGER_EXPORT_MODE=off` is the default and keeps the SQLite ledger as the only authoritative hot-path artifact. Legacy JSON-only runs still load and are migrated into the v4 SQLite ledger.

Persistence failures fail closed with a visible error instead of being silently swallowed. SQLite writes use an IMMEDIATE transaction with WAL mode, foreign keys, row payload hashes, event-chain verification, cached per-process write connections, and dangling-edge validation.

## Claim/evidence graph

Spans are the evidence substrate; claims are the auditable units built on top of it.

Berry now tracks:

```text
ClaimRecord
  cid, text, kind, status, target, latest_audit_id, tags, meta

ClaimEvidenceLink
  link_id, cid, sid, relation, audit_id, note

AuditRecord
  audit_id, kind, claim_ids, input_sids, materialized_sids,
  evidence_pack_id, evidence_pack_hash, verifier_model, policy, result
```

Allowed claim statuses are:

```text
open, supported, contradicted, insufficient, downgraded, closed
```

Allowed claim/evidence relations are:

```text
supports, contradicts, background, insufficient
```

`supports` and `contradicts` edges must point at citable evidence spans. `background` edges may point at non-citable anchors, but they do not prove the claim. Verifier audit tools update claim statuses and create audit-linked edges only after the server has resolved a policy-compliant evidence pack.

Use the graph tools for long-running work:

```text
create_claim(text, target=0.95)
link_claim_evidence(cid, sid, relation="supports")
list_claim_evidence(cid=cid)
audit_claims(claim_ids=[...])
mark_claim(cid, status="downgraded", reason="target too broad")
list_claims(status=["open", "insufficient"])
get_claim(cid)
list_audits(claim_id=cid)
```

The intended v2 loop is:

```text
create claims -> gather spans -> link evidence -> audit claims -> update/downgrade claims -> answer from supported claims plus explicit unknowns
```

## Search and packing

`query_evidence` replaces raw token-count search with filterable lexical retrieval:

```text
query_evidence(query, kinds?, source_types?, trust?, status?, include_derived?, include_stale?)
```

`get_evidence_pack` materializes the exact policy-filtered pack that verifier tools use.

## Compatibility

The legacy tools remain available:

```text
detect_hallucination(answer, spans, ...)
audit_trace_budget(steps, spans, ...)
```

They are useful for local experiments and external callers that do not use Berry runs, but they are lower-trust because the caller supplies the span text. New agent workflows should use the `_run` variants.
