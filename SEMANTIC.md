# Deployment-Oriented Overview of the Hallucination Toolkit

A concise, deployment-oriented overview of the toolkit that clarifies (1) how masking/skeletons work, (2) how the classical and semantic decision heads differ, (3) how to get more (or less) abstention, and (4) how to activate a code/creative mode safely—without letting it answer make-believe factual questions.

## 1. How Masking and Skeletons Work

The toolkit builds an ensemble of "skeleton" prompts per item to estimate priors that are deliberately less informative than the full prompt. We use two families:

### A. Evidence-erase (structured evidence prompts)
If the prompt has explicit fields (e.g., "Evidence:", "Context:", "References:"), the skeletonizer replaces those fields with a mask token and keeps the task body. This is used when you intend to gate on provided evidence.

### B. Closed-book semantic masking (unstructured prompts)
When there is no explicit evidence, we apply probabilistic masking (per match, by "strength") and block permutation to disrupt position cues:

- Multi-word Title-Case entities (e.g., Carlos Santana, New York)
- Years (e.g., 1985) and numbers
- Quoted spans
- Then permute prompt blocks (optionally preserving the first role line)

This does not target single capitalized words like "Python" but will mask multi-word entities; masking is randomized by seed and level so the ensemble spans milder to stronger ablations. This is the default for "closed-book" items where you want the model to rely only on persistent knowledge.

**Why masking?** The skeletons approximate "what the model would predict if some specifics were hidden". The EDFL / Bits-to-Trust gate then asks: "Do we have enough information (Δ̄) to lift the prior to the target reliability?" If not, abstain.

## 2. Classical vs. Semantic Decision Heads

### Classical (binary-event) head

**What it models:** A single binary event "MODEL-ANSWERS" vs "REFUSES" on each skeleton and on the full prompt.

**Signals:**
- Posterior P_y for the chosen event
- Skeleton priors S_{k,y}
- Compute the information budget Δ̄ (clipped log-likelihood ratio)
- Bits-to-Trust B2T = KL(Ber(1-h*) ∥ Ber(q_lo))
- ISR = Δ̄/B2T

**Decision:** Answer iff ISR ≥ thr and Δ̄ ≥ B2T.

**Pros:** Very fast, robust; directly gates the act of answering.

**Cons:** "Answer"/"refuse" is a coarse proxy; it ignores what the model would answer.

### Semantic (meaning-level) head

**What it models:** Uncertainty over meanings, not strings. We sample multiple answers, cluster by bi-directional entailment, then compute a posterior over clusters of equivalent meaning and their semantic entropy or probabilities. The top-cluster's mass P* becomes a key posterior signal. (See Fig. 1 on page 626 for the meaning-first pipeline.)

**Signals added:**
- P* (top-meaning posterior under full prompt)
- Skeleton priors projected onto the same clusters
- Δ̄ computed either as multi-class KL (with smoothing) or a 1-D LLR for the top meaning (stable when cluster supports differ)

**Decision (v2.4):** All of the following must hold:
1. Posterior-dominance latch: P* ≥ 1 - h* (the model must be confident in its top meaning)
2. ISR ≥ thr and Δ̄ ≥ B2T (information sufficiency)
3. Top meaning is not refusal-like

This prevents "answering" when the posterior is diffuse across multiple meanings or the best meaning is a refusal.

### When to use which?

- **Use classical** when you simply need a safe yes/no gate on whether to proceed.
- **Use semantic** when free-form outputs matter (QA with phrasing variability, summaries, longer answers) and you want abstentions to track meaning-level uncertainty. Semantic entropy has been shown to better flag confabulations across models/datasets (AUROC ≈ 0.79 in aggregate).

## 3. How to Get More Flexibility (or Strictness) Around Abstaining

You can make abstentions looser/tighter without changing code paths by tuning:

- **Target reliability h*** → sets the bar 1 - h*
  - Strict factual QA: h* ∈ [0.02, 0.10] → posterior target 90–98%
  - Creative/code: h* ∈ [0.20, 0.40] → target 60–80% (see §4)

- **ISR threshold** (default 1.0): raise to be stricter on budget sufficiency

- **Margin** (extra bits): require Δ̄ ≥ B2T + margin for critical flows

- **Posterior-dominance latch** (semantic): keep it on, but its effective bar moves with h* (e.g., at h* = 0.30 the latch is P* ≥ 0.70)

- **Priors:** increase m (skeletons) or masking strength for a more conservative q_lo

- **Posterior samples:** increase M_full to reduce spurious multi-cluster splits and push P* up

In v2.4 we also added a **prior-sufficiency gate**: if the conservative prior q_lo and posterior P* already exceed the target, we set B2T → 0 and allow an answer (unless the top meaning looks like a refusal). This prevents needless abstention when the model already "knows" the answer.

## 4. "Code/Creative Mode": How to Activate It, and What It Does and Does Not Guarantee

**Goal:** Let good solutions through when outputs are diverse in form (and sometimes in approach), while keeping strong safety for factual claims.

### How to activate (no code changes needed)

1. **Set a looser h*:** e.g., `h_star=0.30` (target = 0.70).
   This relaxes both the posterior latch and the Bits-to-Trust bar so consistent but stylistically diverse outputs can pass.

2. **Constrain the output type** for the full-prompt samples to reduce spurious meaning splits:
   - Code: ask for code only (single `python ...` block, no prose)
   - Creative prose: ask for a single paragraph in the requested style
   (Doing so naturally concentrates the posterior into one or two "meanings".)

3. **(Optional)** Increase M_full to 16–20 for more stable P*

4. **(Optional)** Closed-book mask skip-list for developer tokens ("python", "regex", "prime numbers", "breadth first search") so skeletons don't remove the task intent. (Add a case-insensitive skip set in the masker.)

### What the guarantees mean here

You are **not** asserting high factual reliability. You are asserting:
- (i) The model's top meaning has posterior mass ≥ 1 - h* (e.g., ≥ 70%), and
- (ii) There is enough information budget relative to the prior (ISR ≥ threshold)

This is appropriate for code/snippets, refactoring, style rewrites, outlines, etc.—where the "right answer" may have several good forms.

**Hard rule: Do not use creative/code mode for real-world facts**

Even if the wording feels "story-like", if the prompt contains real entities + dates or numbers (e.g., "Carlos Santana … in March 1985 … met …"), treat it as factual QA: run with the strict profile (e.g., h* ≤ 0.10), keep the posterior latch, and preserve the normal closed-book masking of multi-word entities and years in skeletons. That combination will force abstention on fabricated claims unless the model's posterior is truly concentrated and supported by its priors.

In other words: the "code/creative" settings are a permission structure for stylistic diversity—not a license to answer ungrounded factual questions. The semantic path's bidirectional-entailment clustering is designed to detect meaning dispersion, but factuality must still inherit the EDFL/B2T guarantees and the posterior-dominance latch when claims about the world are present.

## 5. Quick Side-by-Side (Cheat Sheet)

| Aspect | Classical | Semantic |
|--------|-----------|----------|
| **What is modeled** | Binary event ("answer" vs "refuse") | Distribution over meanings (clusters) |
| **Sampling** | Few samples per skeleton & full prompt | Multiple answer samples → entailment clustering → cluster probs |
| **Key signals** | q_lo, Δ̄, B2T, ISR | P*, semantic entropy H_sem, Δ̄ (multi-class KL or 1-D LLR) |
| **Latch** | N/A | Posterior-dominance: P* ≥ 1 - h* |
| **Pros** | Fast, robust on gating | Better at detecting confabulations; respects meaning equivalence |
| **Cons** | Coarse; ignores content | Slightly heavier; needs samples & clustering |
| **Where to use** | Simple safety gate, MCQ, short facts | Free-form QA, summaries, code with stylistic variation |

Citations for the semantic pipeline and estimator are in the Nature paper (see Fig. 1, pages 626–627); our toolkit's prior-sufficiency gate and refusal-like handling are implemented in the v2.3+ planner; the closed-book masking and auto skeleton policy are in the Risk Calculator module.

## TL;DR

- **Masking:** evidence-erase when fields exist; otherwise closed-book probabilistic masking of multi-word entities, years, numbers, quoted spans, plus block permutation.

- **Classical vs. Semantic:** classical gates a binary act; semantic gates meaning, with an added posterior-dominance latch to block diffuse posteriors.

- **More/less abstention:** tune h*, ISR threshold, margin, masking strength, m/M. Keep the prior-sufficiency shortcut when priors already clear the bar.

- **Code/Creative mode:** use lower 1 - h* targets and output-type constraints for non-factual tasks; never apply it to prompts that name real people/places with dates or numbers—fallback to strict factual QA in those cases.


## License

This project is licensed under the MIT License — see the LICENSE file for details.

## Attribution

Developed by Hassana Labs (https://hassana.io).

This implementation follows the framework from the paper “Compression Failure in LLMs: Bayesian in Expectation, Not in Realization” and related EDFL/ISR/B2T methodology.
