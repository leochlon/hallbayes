# Changelog

All notable changes to this project are documented here.

## v2.0.0 — Semantic (experimental) + Packaging cleanup
Release date: 2025-09-10

- Semantic engine v2 (formerly v2.4) is now the canonical module at `scripts/hallucination_toolkit_semantic.py`.
- Concurrency: parallel skeleton sampling (`max_workers_skeletons`) and optional item-level parallelism (`max_workers_items`).
- Safer decisioning: prior-sufficiency gate, posterior-dominance latch (require P* ≥ 1 − h*), optional LLR top-meaning budget, and KL smoothing for multi-class KL.
- Thread-safe entailment providers and caches.
- Web UI: engine toggle now shows “Classic” and “Semantic (experimental)” (v2). v2.3 removed.
- Packaging:
  - Offline backend build uses a local virtualenv (`.build-venv`) to avoid PEP 668 “externally managed” errors.
  - Electron bundle includes `scripts/**` so the semantic engine is available inside the DMG.
- Repo cleanup:
  - Removed legacy semantic modules `hallucination_toolkit_semantic_v2_1.py`, `v2_2.py`, and `v2_3.py`.
  - Moved integration example to `app/examples/test_semantic.py`.

## v1.x — Classic EDFL Toolkit
- Classic EDFL/B2T/ISR decisioning and evidence-erase skeletons live at `scripts/hallucination_toolkit.py`.
- Web UI and CLI flows continue to support Classic alongside the new Semantic engine.

