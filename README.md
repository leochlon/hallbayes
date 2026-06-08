# Berry

## Workflow verification playbooks

Want the “with vs without hallucination detector” experience? Start here:

- `docs/workflows/README.md` — index of workflow playbooks
  - Search & Learn
  - Generate Boilerplate/Content
  - Inline Completions
  - Greenfield Prototyping
  - RCA Fix Agent

Each playbook includes a before/after worked example (uncited output vs evidence-backed + verifier).

Berry runs a local MCP server with a safe, repo‑scoped toolpack plus verification tools (`detect_hallucination`, `audit_trace_budget`).

Berry ships a single MCP surface: **classic**.

Classic includes:
- Verification tools (`detect_hallucination`, `audit_trace_budget`) plus run-scoped variants that resolve evidence from the server-owned span ledger
- Run & evidence notebook tools (start/load runs, add/list/query/extract/pack spans)
- A claim/evidence graph (`create_claim`, `link_claim_evidence`, `audit_claims`, `list_audits`) persisted in a SQLite ledger

See `docs/MCP.md`, `docs/SPANS.md`, and `docs/workflows/README.md`.

Berry integrates with Cursor, Codex, Claude Code, and Gemini CLI via config files committed to your repo.

## Supported verifier backends

Berry’s current verification method requires token logprobs (Chat Completions-style `logprobs` + `top_logprobs`).

Supported today:
- `openai` (default): OpenAI-compatible Chat Completions endpoints with logprobs (OpenAI, OpenRouter, local vLLM, or any compatible `base_url`)
- `gemini`: Gemini Developer API `generateContent` with token logprobs via `logprobsResult` (when enabled for the model)
- `vertex`: Vertex AI `generateContent` (Gemini models) with token logprobs via `logprobsResult`
- `dummy`: deterministic offline backend for tests/dev

Not supported yet:
- Anthropic (OpenAI-compat layer ignores `logprobs`)

## Quickstart

1) Install:

```bash
pipx install -e .
```

2) In each repo you want to use Berry:

```bash
berry init
```

This provisions a hosted API key, writes MCP configs for Cursor/Codex/Claude Code/Gemini CLI, and sets up a Claude Code skill file.

3) Reload MCP servers in your client.

For a platform-aware assistant installer with user/profile installs, platform
shortcuts, hooks, always-on instruction files, and embedded path refreshes, use:

```bash
berry install --platform codex
berry cursor install
berry install --project --platform gemini
```

See `docs/INSTALL.md` for the full platform table.

To use your own API key or a different backend instead of the hosted key:

```bash
berry setup
```

## Docs

- `docs/USAGE.md` — task‑oriented guides
- `docs/CLI.md` — command reference
- `docs/INSTALL.md` — multi-platform assistant installer
- `docs/CONFIGURATION.md` — config files, defaults, and env vars
- `docs/MCP.md` — tools/prompts and transport details
- `docs/SPANS.md` — span-ledger model, SQLite/event-log persistence, claim/evidence graph, evidence-pack policy, and server-resolved verification
- `docs/PACKAGING.md` — release pipeline (macOS pkg + Homebrew cask)

## Tests

```bash
pytest
```
