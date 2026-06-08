# Berry v2 benchmark: information sufficiency, not correctness

## What we are measuring (and what we are not)

Berry's verifier scores **information sufficiency**: given the cited evidence,
is there enough information to support the claim? It is the question a RAG/agent
stack needs answered ("is this answer grounded in the retrieved
context?"), and it is what the EDFL information-budget computes (how many bits
the claim's prediction depends on the cited evidence).

It is **not** a world-correctness fact-checker. A claim can be true yet
unsupported by the cited evidence (insufficient grounding); Berry should and does
flag that. This is why correctness benchmarks (e.g. HaluEval QA, whose negative is
a *factually wrong* answer) are the wrong target: they conflate "wrong" with
"ungrounded." We benchmark sufficiency directly.

## Design: controlled evidence ablation (ground-truth sufficiency labels)

Hold the claim fixed; change only the evidence:

- **Sufficient**: cite the gold supporting evidence -> Berry should PASS.
- **Insufficient**: cite distractor evidence with the key fact absent -> Berry should FLAG.

The label is sufficiency, established by construction, so detection metrics
isolate sufficiency from correctness. Datasets with this structure:

- **HotpotQA (distractor)**: self-contained: gold `supporting_facts` vs 8 distractor paragraphs.
- **FEVER**: `SUPPORTED` (evidence sufficient) vs `NotEnoughInfo` (insufficient by annotation).

## Metrics (and why)

| Metric | What it shows |
|---|---|
| **AUROC** of the budget-gap score | does the score rank insufficient above sufficient |
| **AURC** (area under risk-coverage) | deployment-facing: error among the claims Berry chooses to answer |
| **risk @ coverage** | the operating curve a single yes/no judge cannot offer |
| **precision / recall / F1** at the target | decision quality at the prespecified risk target |

AUROC/AURC + a calibration view are the standard pairing for selective-prediction
tools; global AUROC alone hides deployment behaviour, so we report the
risk-coverage/decision view alongside it.

## Result: HotpotQA distractor (pilot)

Model `gpt-4.1-mini` (single logprob pass), 398 claims (199 sufficient /
199 insufficient), evidence ablation, 93s on a 12-endpoint pool.

| | Berry v2 verifier |
|---|---|
| AUROC (sufficient vs insufficient) | **0.841** |
| AURC (lower better) | **0.225** |
| decision @ target 0.95 | acc 0.81 · prec 0.74 · **recall 0.975** · F1 0.84 |
| mean budget-gap | sufficient **6.4 bits** vs insufficient **36.2 bits** |

The budget-gap separates the two conditions cleanly, and at the default target
Berry catches 97.5% of insufficient-evidence claims (5 false negatives / 199).

Framing matters: the identical model/verifier/harness scored only AUROC 0.722 when
pointed at a *correctness* task (HaluEval QA). Measuring sufficiency, what the
tool does, lifts it to 0.841.

## Result: RAGTruth (realistic RAG faithfulness)

Real LLM responses (gpt-4-era) with the context given to the model and human
grounding labels; the deployment question "is this answer supported by the
retrieved context?" 400 Summary responses, base rate 0.24 ungrounded,
`gpt-4.1-mini`. Each response is decomposed into atomic claims, each verified
against the context (the proper RAGTruth scoring).

| | Berry v2 verifier |
|---|---|
| AUROC | **0.819** |
| AURC (lower better) | **0.097** |
| decision @ target 0.95 | acc 0.58 · prec 0.35 · **recall 0.884** · F1 0.50 |

Berry catches 88% of ungrounded responses (11 false negatives / 95), and the low
AURC means the responses it ranks most-grounded really are. At the default target
it over-flags grounded responses (precision 0.35), a recall-first gate; pick the
operating point from the risk-coverage curve, not the fixed 0.95. A sentence-level
split scores the same (AUROC 0.810), so the result is robust to decomposition
granularity.

The 11 misses are plausible fabricated additions in long outputs (9 to 15 atomic
claims), each with its worst atomic claim just inside the grounded side (budget
gap -1.2 to -1.6). Both the decomposition and the verification run on
gpt-4.1-mini, so the same mid-tier model that drafts the atomic claims also judges
them. A stronger reasoning model (Claude Opus, or an o-series thinking model)
should split these long responses more faithfully and judge plausible additions
more strictly; that is the expected recall lift and is not yet measured.

## Result: VitaminC (contrastive control, the rigor headline)

VitaminC is evidence-relative (the *same claim* gets a different label as the
evidence changes), so it cannot be passed by world-correctness or by the claim's
prior. We slice by axis: **SUPPORTS** (sufficient) vs **NOT ENOUGH INFO**
(insufficient) is the pure-sufficiency test; **REFUTES** (sufficient but
contradicting) is reported separately. 800 rows, `gpt-4.1-mini`.

| | Berry v2 verifier |
|---|---|
| **AUROC (SUPPORTS vs NEI)**: pure sufficiency | **0.929** |
| AUROC (SUPPORTS vs NEI+REFUTES), full grounding | 0.939 |
| per class (recall / mean gap) | SUPPORTS FP 0.24 (−1.3) · NEI 0.93 (34.1) · REFUTES 0.96 (35.3) |

**Contrastive control (the key figure):** hold the claim fixed, change only the
evidence. Across 390 paired claims, Berry's budget-gap **rises when the evidence
weakens in 95% of pairs** (mean Δ **+36.8 bits**). The claim is identical, so this
isolates the evidence: Berry tracks information sufficiency, not correctness and
not the claim's prior.

## Cross-dataset summary

| Dataset | what it tests | AUROC | recall | note |
|---|---|---|---|---|
| **VitaminC** (SUP vs NEI) | contrastive sufficiency control | **0.929** | NEI 0.93 | 95% contrastive (claim fixed) |
| HotpotQA (ablation) | constructed sufficiency | 0.841 | 0.975 | off-topic negatives (easy) |
| RAGTruth (atomic) | realistic RAG grounding | 0.819 | 0.884 | real LLM outputs, AURC 0.097 |

Consistent grounding/sufficiency detection (single logprob pass, `gpt-4.1-mini`)
across a contrastive control, a constructed ablation, and a realistic RAG set.
The score ranks grounding well everywhere; the shared caveat is precision at the
recall-first default target (worst on RAGTruth, 0.35; best on VitaminC, FP 0.24),
so pick the operating point from the risk-coverage curve.

## Cost

One logprob forward pass per claim (grouped), no sampling. Published intrinsic
detectors report semantic-entropy ~0.69 AUROC at K=5 passes and NLI ~0.58 at 0
extra passes (different datasets, so directional only). Berry sits at the cheap
end of the cost axis with competitive separation; a same-dataset head-to-head is
the next step.

## Capability pillars no scoring baseline offers

These are part of the adoption story and need no model to demonstrate:

- **Tamper-evidence**: edits to the run payload, a span row, or the event-hash
  chain all fail closed on load (3/3 detected vs 0/3 for the prior JSON store).
- **Server-resolved evidence + provenance**: the verifier scores spans resolved
  from the ledger, not caller-supplied text; every audit is reproducible.

## Honest caveats

- The pilot runs one mid-tier model (`gpt-4.1-mini`) for both atomic decomposition
  and verification. The RAGTruth misses are plausible additions in long responses
  that a stronger reasoning model (Claude Opus or an o-series thinking model)
  should decompose and judge better; not yet measured. HotpotQA distractor is a
  relatively clean separation; FEVER SUPPORTED-vs-NEI is the next dataset and needs
  gold Wikipedia evidence text.
- The default 0.95 target favours recall; precision (0.74) reflects some
  over-flagging; report the risk-coverage curve, not a single operating point.
- Numbers are a pilot (≈200 items/condition); scale to 1–2k with bootstrap CIs
  for a publishable figure.

## Reproduce

```bash
# 1. proxy: OpenAI-compatible -> Azure pool, round-robin concurrency
python bench/aoai_proxy.py --pool ~/Downloads/aiderB/aoai_pool.json --port 8900 &

# 2. data: HotpotQA distractor via HF datasets-server (no big download)
curl -s "https://datasets-server.huggingface.co/rows?dataset=hotpotqa/hotpot_qa&config=distractor&split=validation&offset=0&length=200" -o /tmp/hotpot.json

# 3. run the sufficiency ablation through Berry's real verifier
OPENAI_BASE_URL=http://127.0.0.1:8900/v1 OPENAI_API_KEY=proxy \
  python bench/run_suff.py --data /tmp/hotpot.json --out /tmp/suff.jsonl
```

## How to present it

1. One bar/scatter: budget-gap distribution, sufficient vs insufficient (the 6.4
   vs 36.2 separation), the most legible single result.
2. The risk-coverage curve with the chosen operating point marked.
3. A small table: AUROC/AURC/F1 across datasets (HotpotQA, then FEVER) and 1–2
   models, with the cost column (passes/claim) next to AUROC to make the
   cheap-and-grounded point.
4. The capability row: tamper-evidence + provenance, as the "and it's auditable"
   close.
