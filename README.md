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
- Verification tools (`detect_hallucination`, `audit_trace_budget`)
- Run & evidence notebook tools (start/load runs, add/list/search spans)

See `docs/MCP.md` and `docs/workflows/README.md`.

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

## Local Models

Berry verifies against any OpenAI-compatible Chat Completions endpoint that returns `logprobs` + `top_logprobs` — so locally hosted models (vLLM, llama.cpp, LM Studio) work as first-class verifier backends. Point Berry at the local `base_url` and pick a model with reliable logprob support; no real API key is needed for most local servers (any non-empty string works). Run `berry doctor` to validate connectivity, logprob availability, and model selection before you start verifying.

```jsonc
// ~/.berry/mcp_env.json
{
  "BERRY_VERIFIER_BACKEND": "local",
  "BERRY_VERIFIER_MODEL": "gpt-oss-20b",
  "BERRY_LOCAL_BASE_URL": "http://127.0.0.1:1234/v1",
  "OPENAI_API_KEY": "not-needed"
}
```

See `docs/LOCAL-MODELS.md` for the full setup, supported servers, and calibration guidance.

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

To use your own API key or a different backend instead of the hosted key:

```bash
berry setup
```

## Docs

- `docs/USAGE.md` — task‑oriented guides
- `docs/CLI.md` — command reference
- `docs/LOCAL-MODELS.md` — running Berry against locally hosted verifier models (incl. `berry doctor`)
- `docs/CONFIGURATION.md` — config files, defaults, and env vars
- `docs/MCP.md` — tools/prompts and transport details
- `docs/PACKAGING.md` — release pipeline (macOS pkg + Homebrew cask)

## Tests

```bash
pytest
```
