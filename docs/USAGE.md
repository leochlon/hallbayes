# Berry usage

## Workflow playbooks (evidence-gated)

If you're using Strawberry/Berry to verify AI output, start with the workflow playbooks:

- `docs/workflows/README.md`

Each includes a worked **without vs with** example and copy/paste prompts for the verifier tools.

This is the practical guide to using Berry with MCP clients.

## Install Berry (from repo)

Prereqs: Python 3.10+.

```bash
pipx install -e .
```

Fallback:

```bash
pip install -e .
```

## Start the MCP server

Default transport is stdio (used by most clients):

```bash
berry mcp  # runs classic
```

Pick a server surface explicitly:

```bash
berry mcp --server classic   # classic toolpack

Older configs may still pass `--server science`/`--server forge`; those values are treated as aliases for `classic`.
```

Optional transports (require licensing; see `MCP.md`):

```bash
berry mcp --transport sse --host 127.0.0.1 --port 8000
berry mcp --transport streamable-http --host 127.0.0.1 --port 8000
```

## Initialize a repo for clients

From the repo root:

```bash
berry init
```

To also enable strict verification gates for this repo (writes `./.berry/config.json`):

```bash
berry init --strict
```

This creates repo‑scoped config files:
- `.cursor/mcp.json`
- `.codex/config.toml`
- `.mcp.json` (Claude Code)
- `.gemini/settings.json`

If you need to overwrite existing files:

```bash
berry init --force
```

## Connect each client

Use the generated files above, or print configs on demand:

```bash
berry print-config cursor
berry print-config codex
berry print-config claude
berry print-config gemini
```

Cursor deep‑link (installs in Cursor directly):

```bash
berry deeplink cursor
```

Per‑client setup hints:

```bash
berry instructions
```

## Optional: global registration (CLI clients)

Some clients support global MCP registration (so Berry can appear without committing repo config files).

To register Berry globally (best‑effort):

```bash
berry integrate
```

This currently targets Claude Code and Codex via their CLIs (`claude mcp add ...`, `codex mcp add ...`).
Clients that are not installed are skipped.

## Enable write access (off by default)

Writes are blocked unless you explicitly enable them. Enable writes and (optionally) add extra roots:

```bash
berry config set allow_write true
berry config add-root /absolute/path/to/extra/root
```

Notes:
- Writes are always limited to the repo root and allowed roots.
- Reads are allowed inside the repo root by default.

## Enforce verification (optional)

To force agents to use Berry's verification gates before repo operations, enable enforcement:

```bash
berry config set enforce_verification true
```

You can also enforce per-process via env var:

```bash
export BERRY_ENFORCE_VERIFICATION=1
```

## Evidence-based verification flow (MCP)

Berry focuses on evidence collection and hallucination detection. A typical verified workflow is:

1) `start_run(problem_statement, deliverable)` — creates a run with anchor spans (`S0` problem, `S1` deliverable)
2) Gather evidence spans:
   - `add_span(text)` — add text as evidence
   - `add_file_span(path, start_line, end_line)` — capture file excerpts
   - `distill_span(parent_sid, pattern)` — extract key lines from large spans
3) Verify claims before presenting them:
   - `detect_hallucination(answer, spans)` — check that an answer with `[S#]` citations is supported by evidence
   - `audit_trace_budget(steps, spans)` — verify explicit (claim, cites) trace steps

If verification flags claims as unsupported, gather more evidence and retry.

## Hallucination / citation checking

Berry exposes two verification tools:

- `detect_hallucination` — scores an answer containing `[S#]` citations against provided spans.
- `audit_trace_budget` — scores an explicit list of `(claim, cites)` steps.

These tools require a verifier to be configured (see `CONFIGURATION.md`). The `openai` backend uses `OPENAI_API_KEY`; the `gemini` backend uses `GEMINI_API_KEY`; the `vertex` backend uses `VERTEX_ACCESS_TOKEN`. The easiest path is `berry setup`.

## Audit log export / prune

Export audit log:

```bash
berry audit export --out /tmp/berry-audit.json
```

Prune old entries (uses `audit_log_retention_days` in config):

```bash
berry audit prune
```

## Recipes

List built‑in recipes:

```bash
berry recipes list
```

Export built‑ins to a JSON file:

```bash
berry recipes export --out /tmp/berry-recipes.json
```

Install a built‑in recipe into the repo:

```bash
berry recipes install search-learn
```

Import a recipe JSON file into the repo:

```bash
berry recipes import /path/to/recipe.json
```

## Support bundle

```bash
berry support bundle
```

Or create a bundle and print an issue template:

```bash
berry support issue
```

## Verify a signed artifact

```bash
berry verify --artifact /path/to/file --signature /path/to/file.sig --public-key /path/to/key.pub
```
