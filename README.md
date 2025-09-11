# Hallucination Risk Calculator & Prompt Re-engineering Toolkit (OpenAI-only)

**Post-hoc calibration without retraining** for large language models. This toolkit turns a raw prompt into:  
1) a **bounded hallucination risk** using the Expectation-level Decompression Law (EDFL), and  
2) a **decision** to **ANSWER** or **REFUSE** under a target SLA, with transparent math (nats).

It supports two deployment modes:

- **Evidence-based:** prompts include *evidence/context*; rolling priors are built by erasing that evidence.
- **Closed-book:** prompts have *no evidence*; rolling priors are built by semantic masking of entities/numbers/titles.

All scoring relies **only** on the OpenAI Chat Completions API. No retraining required.

---

## Table of Contents
- [Install & Setup](#install--setup)
- [Core Mathematical Framework](#core-mathematical-framework)
- [Understanding System Behavior](#understanding-system-behavior)
- [Two Ways to Build Rolling Priors](#two-ways-to-build-rolling-priors)
- [API Surface](#api-surface)
- [Calibration & Validation](#calibration--validation)
- [Practical Considerations](#practical-considerations)
- [Project Layout](#project-layout)
- [Deployment Options](#deployment-options)

---

## Install & Setup

```bash
pip install --upgrade openai
export OPENAI_API_KEY=sk-...
```

> The module uses `openai>=1.0.0` and the Chat Completions API (e.g., `gpt-4o`, `gpt-4o-mini`).

---

## Core Mathematical Framework

### The EDFL Principle

Let the binary event $\mathcal{A}$ be the thing you want to guarantee (e.g., **Answer** in decision mode, or **Correct** for factual accuracy).  
Build an ensemble of **content-weakened prompts** (the *rolling priors*) $\{S_k\}_{k=1}^m$. For the realized label $y$, estimate:

- **Information budget:**  
  $$\bar{\Delta} = \tfrac{1}{m}\sum_k \mathrm{clip}_+(\log P(y) - \log S_k(y), B)$$
  (one-sided clipping; default $B=12$ nats to prevent outliers while maintaining conservative bounds).

- **Prior masses:** $q_k = S_k(\mathcal{A})$, with:
  - $\bar{q}=\tfrac{1}{m}\sum_k q_k$ (average prior for EDFL bound)
  - $q_{\text{lo}}=\min_k q_k$ (worst-case prior for SLA gating)

By EDFL, the achievable reliability is bounded by:  
$$\bar{\Delta} \ge \mathrm{KL}(\mathrm{Ber}(p) \| \mathrm{Ber}(\bar{q})) \Rightarrow p\le p_{\max}(\bar{\Delta},\bar{q})$$

Thus the **hallucination risk** (error) is bounded by $\overline{\mathrm{RoH}} \le 1 - p_{\max}$.

### Decision Rule (SLA Gating)

For target hallucination rate $h^*$:
- **Bits-to-Trust:** $\mathrm{B2T} = \mathrm{KL}(\mathrm{Ber}(1-h^*) \| \mathrm{Ber}(q_{\text{lo}}))$
- **Information Sufficiency Ratio:** $\mathrm{ISR} = \bar{\Delta}/\mathrm{B2T}$
- **ANSWER** iff $\mathrm{ISR}\ge 1$ *and* $\bar{\Delta} \ge \mathrm{B2T} + \text{margin}$ (default `margin≈0.2` nats)

> **Why two priors?** The gate uses **worst-case** $q_{\text{lo}}$ for strict SLA compliance. The RoH bound uses **average** $\bar{q}$ per EDFL theory. This dual approach ensures conservative safety while providing realistic risk bounds.

---

## Understanding System Behavior

### Expected Behavioral Patterns

The toolkit exhibits different behaviors across query types, which is **mathematically consistent** with the framework:

#### Simple Arithmetic Queries
**Observation:** May abstain despite apparent simplicity  
**Explanation:**
- Models often attempt answers even with masked numbers (pattern recognition)
- This yields **low information lift** $\bar{\Delta} \approx 0$ between full prompt and skeletons
- Despite potentially low EDFL risk bound, worst-case prior gate triggers **abstention** (ISR < 1)

#### Named-Entity Factoids
**Observation:** Generally answered with confidence  
**Explanation:**
- Masking entities/dates substantially reduces answer propensity in skeletons
- Restoring these yields **large** $\bar{\Delta}$ that clears B2T threshold
- System **answers** with tight EDFL risk bound

**This is not a bug but a feature**: The framework prioritizes safety through worst-case guarantees while providing realistic average-case bounds.

### Mitigation Strategies

1. **Switch Event Measurement**
   - Use **Correct/Incorrect** instead of Answer/Refuse for factual QA
   - Skeletons without key information rarely yield correct results → large $\bar{\Delta}$

2. **Enhance Skeleton Weakening**
   - Implement mask-aware decision head that refuses on redaction tokens
   - Ensures skeletons have strictly lower "Answer" mass than full prompt

3. **Calibration Adjustments**
   - Relax $h^*$ slightly (e.g., 0.10 instead of 0.05) for higher answer rates
   - Reduce margin for less conservative gating
   - Increase sampling ($n=7-10$) for stability

4. **Provide Evidence**
   - Adding compact, relevant evidence increases $\bar{\Delta}$ while preserving bounds

---

## Two Ways to Build Rolling Priors

### 1) Evidence-based (when you have context)

- **Prompt** contains a field like `Evidence:` (or JSON keys)
- **Skeletons** erase the evidence content but preserve structure and roles; then permute blocks deterministically (seeded)
- **Decision head**: "Answer only if the provided evidence is sufficient; otherwise refuse."

**Example**
```python
from scripts.hallucination_toolkit import OpenAIBackend, OpenAIItem, OpenAIPlanner

backend = OpenAIBackend(model="gpt-4o-mini")
prompt = (
    """Task: Answer strictly based on the evidence below.
Question: Who won the Nobel Prize in Physics in 2019?
Evidence:
- Nobel Prize press release (2019): James Peebles (1/2); Michel Mayor & Didier Queloz (1/2).
Constraints: If evidence is insufficient or conflicting, refuse.
"""
)
item = OpenAIItem(
    prompt=prompt, 
    n_samples=5, 
    m=6, 
    fields_to_erase=["Evidence"], 
    skeleton_policy="auto"
)
planner = OpenAIPlanner(backend, temperature=0.3)
metrics = planner.run(
    [item], 
    h_star=0.05, 
    isr_threshold=1.0, 
    margin_extra_bits=0.2, 
    B_clip=12.0, 
    clip_mode="one-sided"
)
for m in metrics: 
    print(f"Decision: {'ANSWER' if m.decision_answer else 'REFUSE'}")
    print(f"Rationale: {m.rationale}")
```

### 2) Closed-book (no evidence)

- **Prompt** has no evidence
- **Skeletons** apply **semantic masking** of:
  - Multi-word proper nouns (e.g., "James Peebles" → "[…]")
  - Years (e.g., "2019" → "[…]")
  - Numbers (e.g., "3.14" → "[…]")
  - Quoted spans (e.g., '"Nobel Prize"' → "[…]")
- **Masking strengths**: Progressive levels (0.25, 0.35, 0.5, 0.65, 0.8, 0.9) across skeleton ensemble
- **Mask-aware decision head** refuses if redaction tokens appear or key slots look missing

**Example**
```python
from scripts.hallucination_toolkit import OpenAIBackend, OpenAIItem, OpenAIPlanner

backend = OpenAIBackend(model="gpt-4o-mini")
item = OpenAIItem(
    prompt="Who won the 2019 Nobel Prize in Physics?",
    n_samples=7,  # More samples for stability
    m=6,          # Number of skeletons
    skeleton_policy="closed_book"
)
planner = OpenAIPlanner(backend, temperature=0.3, q_floor=None)
metrics = planner.run(
    [item], 
    h_star=0.05,           # Target max 5% hallucination
    isr_threshold=1.0,     # Standard ISR gate
    margin_extra_bits=0.2, # Safety margin in nats
    B_clip=12.0,          # Clipping bound
    clip_mode="one-sided" # Conservative clipping
)
for m in metrics: 
    print(f"Decision: {'ANSWER' if m.decision_answer else 'REFUSE'}")
    print(f"Δ̄={m.delta_bar:.4f}, B2T={m.b2t:.4f}, ISR={m.isr:.3f}")
    print(f"EDFL RoH bound={m.roh_bound:.3f}")
```

**Tuning knobs (closed-book):**
- `n_samples=5–7` and `temperature≈0.3` stabilize priors
- `q_floor` (Laplace by default: $1/(n+2)$) prevents worst-case prior collapse to 0
- Adjust masking strength levels if a task family remains too answerable under masking

---

## API Surface

### Core Classes

- `OpenAIBackend(model, api_key=None)` – wraps Chat Completions API
- `OpenAIItem(prompt, n_samples=5, m=6, fields_to_erase=None, skeleton_policy="auto")` – one evaluation item
- `OpenAIPlanner(backend, temperature=0.5, q_floor=None)` – runs evaluation:
  - `run(items, h_star, isr_threshold, margin_extra_bits, B_clip=12.0, clip_mode="one-sided") -> List[ItemMetrics]`
  - `aggregate(items, metrics, alpha=0.05, h_star, ...) -> AggregateReport`

### Helper Functions

- `make_sla_certificate(report, model_name)` – creates formal SLA certificate
- `save_sla_certificate_json(cert, path)` – exports certificate for audit
- `generate_answer_if_allowed(backend, item, metric)` – only emits answer if decision was ANSWER

### ItemMetrics Fields

Every `ItemMetrics` includes:
- `delta_bar`: Information budget (nats)
- `q_conservative`: Worst-case prior $q_{\text{lo}}$
- `q_avg`: Average prior $\bar{q}$
- `b2t`: Bits-to-Trust requirement
- `isr`: Information Sufficiency Ratio
- `roh_bound`: EDFL hallucination risk bound
- `decision_answer`: Boolean decision
- `rationale`: Human-readable explanation
- `meta`: Dict with `q_list`, `S_list_y`, `P_y`, `closed_book`, etc.

---

## Calibration & Validation

### Validation Set Calibration

On a labeled validation set:

1. **Sweep the margin** parameter from 0 to 1 nats
2. For each margin, compute:
   - Empirical hallucination rate among answered items
   - Wilson upper bound at 95% confidence
3. **Select smallest margin** where Wilson upper bound ≤ target $h^*$ (e.g., 5%)
4. **Freeze policy**: $(h^*, \tau, \text{margin}, B, \text{clip\_mode}, m, r, \text{skeleton\_policy})$

### Portfolio Reporting

The toolkit provides comprehensive metrics:
- Answer/abstention rates
- Empirical hallucination rate + Wilson bound
- Distribution of per-item EDFL RoH bounds
- Worst-case and median risk bounds
- Complete audit trail

---

## Practical Considerations

### Choosing the Right Event

The default event is the **decision** $\mathcal{A} = \{\text{Answer}\}$. However:

| Task Type | Recommended Event | Rationale |
|-----------|------------------|-----------|
| Factual QA | **Correct/Incorrect** | Directly measures hallucination |
| Decision Support | **Answer/Refuse** | Measures confidence to respond |
| Creative Writing | **Answer/Refuse** | Correctness often undefined |

For tasks where skeletons still trigger answers frequently (causing $\bar{\Delta}\approx0$), switching to **Correctness** event with task-specific grading dramatically improves performance.

### Common Issues & Solutions

#### Issue: $\bar{\Delta} = 0$ with $\overline{\mathrm{RoH}} \approx 0$
**Not a contradiction!** The gate uses worst-case $q_{\text{lo}}$; the bound uses average $\bar{q}$.
- **Solution**: Increase `n_samples`, lower decision temperature, ensure skeletons truly weaken the event

#### Issue: Hit a low $\bar{\Delta}$ ceiling
**Cause**: Clipping may be too aggressive
- **Solution**: Increase `B_clip` (default 12) and use `clip_mode="one-sided"`

#### Issue: Arithmetic still refuses
**Cause**: Pattern recognition allows answers even with masked numbers
- **Solutions**:
  - Switch to **Correctness** event
  - Reduce masking strength for numbers on subset of skeletons
  - Provide worked examples as evidence

#### Issue: Prior collapse ($q_{\text{lo}} \to 0$)
**Cause**: All skeletons strongly refuse
- **Solution**: Apply prior floor (default Laplace: $1/(n+2)$) or use quantile prior

### Performance Characteristics

| Metric | Typical Value | Notes |
|--------|--------------|-------|
| Latency per item | 2-5 seconds | 7 samples × 7 variants (1 full + 6 skeletons) |
| API calls | $(1+m) \times \lceil n/\text{batch}\rceil$ | Can be parallelized |
| Accuracy | Wilson-bounded at 95% | Empirically validated |
| Cost | ~$0.01-0.03 per item | Using gpt-4o-mini |

### Stability Guidelines

1. **Sampling parameters**:
   - Use $n \ge 5$ samples per variant
   - Keep temperature $\in [0.2, 0.5]$ for decision head
   - Lower temperature → more stable priors

2. **Skeleton ensemble**:
   - Use $m \ge 6$ skeletons
   - Ensure diversity in masking strengths
   - Verify skeletons are meaningfully weaker

3. **Clipping strategy**:
   - Always use one-sided clipping for conservative bounds
   - Set $B \ge 10$ nats to avoid artificial ceilings
   - Monitor clipping frequency in logs

---

## Project Layout

```
.
├── app/                    # Application entry points
│   ├── web/web_app.py     # Streamlit UI
│   ├── cli/frontend.py    # Interactive CLI
│   ├── examples/          # Example scripts
│   └── launcher/entry.py  # Unified launcher
├── scripts/               # Core module
│   ├── hallucination_toolkit.py
│   └── build_offline_backend.sh
├── electron/              # Desktop wrapper
├── launch/                # Platform launchers
├── release/              # Packaged artifacts
├── bin/                  # Offline backend binary
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Deployment Options

### 1. Direct Python Usage

```python
from scripts.hallucination_toolkit import (
    OpenAIBackend, OpenAIItem, OpenAIPlanner,
    make_sla_certificate, save_sla_certificate_json
)

# Configure and run
backend = OpenAIBackend(model="gpt-4o-mini")
items = [OpenAIItem(prompt="...", n_samples=7, m=6)]
planner = OpenAIPlanner(backend, temperature=0.3)
metrics = planner.run(items, h_star=0.05)

# Generate SLA certificate
report = planner.aggregate(items, metrics)
cert = make_sla_certificate(report, model_name="GPT-4o-mini")
save_sla_certificate_json(cert, "sla.json")
```

### 2. Web Interface (Streamlit)

```bash
streamlit run app/web/web_app.py
```

### 3. One-Click Launcher

- **Windows**: Double-click `launch/Launch App.bat`
- **macOS**: Double-click `launch/Launch App.command`
- **Linux**: Run `bash launch/launch.sh`

First run creates `.venv` and installs dependencies automatically.

### 4. Desktop App (Electron)

Development:
```bash
cd electron
npm install
npm run start
```

Build installers:
```bash
npm run build
```

### 5. Offline Backend (PyInstaller)

Build single-file executable:
```bash
# macOS/Linux
bash scripts/build_offline_backend.sh

# Windows
scripts\build_offline_backend.bat
```

Creates `bin/hallucination-backend[.exe]` with bundled Python, Streamlit, and dependencies.

---

## New Utility Tools

### Quick Evaluation CLI

For rapid testing and evaluation:

```bash
# Evaluate a single prompt
python scripts/quick_eval.py "What is the capital of France?"

# Interactive mode
python scripts/quick_eval.py --interactive

# Batch process from file
python scripts/quick_eval.py --file prompts.txt --output results.json

# Use permissive settings
python scripts/quick_eval.py --permissive "Complex question here"
```

### Batch Processing Example

See `app/examples/batch_example.py` for a comprehensive example of:
- Processing multiple prompts
- Comparing different configurations
- Generating detailed reports
- Exporting results to JSON/CSV

```bash
python app/examples/batch_example.py
```

### Utility Functions

The `scripts/utils.py` module provides helper functions:
- `print_summary()` - formatted result summaries
- `save_results_to_json/csv()` - export capabilities
- `create_comparison_report()` - compare different runs
- `validate_openai_setup()` - check API configuration
- `create_batch_from_prompts()` - easy batch creation

## Minimal End-to-End Example

```python
from scripts.hallucination_toolkit import (
    OpenAIBackend, OpenAIItem, OpenAIPlanner,
    make_sla_certificate, save_sla_certificate_json,
    generate_answer_if_allowed
)

# Setup
backend = OpenAIBackend(model="gpt-4o-mini")

# Prepare items
items = [
    OpenAIItem(
        prompt="Who won the 2019 Nobel Prize in Physics?",
        n_samples=7,
        m=6,
        skeleton_policy="closed_book"
    ),
    OpenAIItem(
        prompt="If James has 5 apples and eats 3, how many remain?",
        n_samples=7,
        m=6,
        skeleton_policy="closed_book"
    )
]

# Run evaluation
planner = OpenAIPlanner(backend, temperature=0.3)
metrics = planner.run(
    items,
    h_star=0.05,           # Target 5% hallucination max
    isr_threshold=1.0,     # Standard threshold
    margin_extra_bits=0.2, # Safety margin
    B_clip=12.0,          # Clipping bound
    clip_mode="one-sided" # Conservative mode
)

# Generate report and certificate
report = planner.aggregate(items, metrics, alpha=0.05, h_star=0.05)
cert = make_sla_certificate(report, model_name="GPT-4o-mini")
save_sla_certificate_json(cert, "sla_certificate.json")

# Show results
for item, m in zip(items, metrics):
    print(f"\nPrompt: {item.prompt[:50]}...")
    print(f"Decision: {'ANSWER' if m.decision_answer else 'REFUSE'}")
    print(f"Risk bound: {m.roh_bound:.3f}")
    print(f"Rationale: {m.rationale}")
    
    # Generate answer if allowed
    if m.decision_answer:
        answer = generate_answer_if_allowed(backend, item, m)
        print(f"Answer: {answer}")
```

## License

This project is licensed under the MIT License — see the LICENSE file for details.

## Attribution

Developed by Hassana Labs (https://hassana.io).

This implementation follows the framework from the paper “Compression Failure in LLMs: Bayesian in Expectation, Not in Realization” (NeurIPS 2024 preprint) and related EDFL/ISR/B2T methodology.
