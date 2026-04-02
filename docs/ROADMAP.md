# Berry roadmap (verification-first)

Berry's core bet: you can't *prompt* LLMs into being reliable; you must **enforce** verification at the MCP tool boundary.

## Shipped (core verification toolkit)

- **Evidence-based runs**: `start_run` anchors a problem statement and deliverable; spans collect evidence throughout.
- **Evidence collection**: `add_span`, `add_file_span`, `distill_span` for building trusted evidence.
- **Span management**: `list_spans`, `get_span`, `search_spans` for navigating evidence.
- **Hallucination detection**: `detect_hallucination` scores answers with `[S#]` citations against evidence.
- **Trace budget auditing**: `audit_trace_budget` verifies explicit (claim, cites) steps.
- **Workflow prompts**: playbooks for RCA, PR prep, architecture summaries, etc.
- **Setup ergonomics**: `berry init` writes client configs; `berry integrate` registers globally.

## Next

- **Evidence authenticity**: add optional HMAC-signed spans so only server-minted spans are citeable.
- **Export tools**: `export_run_bundle` (spans + audits + decisions) for debugging and compliance.

## Later

- **Safe command capture**: allowlisted runners (e.g., `pytest`) that store stdout/stderr as trusted spans.
- **Policy packs**: per-repo guardrails (required evidence types, required tests) enforced server-side.
- **CI mode**: headless verifier runs that block merges when PR claims are under-evidenced.
- **Repo access tools**: verified read/write/search with citation requirements.
- **Web and exec tools**: verified web fetch and command execution with evidence capture.
