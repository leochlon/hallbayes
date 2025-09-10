# Hallucination Risk Calculator & Toolkit

**Post-hoc calibration for large language models without retraining.** Mathematically-grounded hallucination detection using the Expectation-level Decompression Law (EDFL), providing bounded risk guarantees and controlled abstention.

## Overview

This toolkit turns raw LLM prompts into:
1. **Bounded hallucination risk** with transparent mathematical guarantees
2. **Automated decisions** to ANSWER or REFUSE based on confidence thresholds
3. **SLA certificates** for production deployments (Classic version)

No model retraining required - works directly with OpenAI's Chat Completions API.

## Available Versions

### Classic (`scripts/hallucination_toolkit.py`)
Full-featured production toolkit with binary decision model:
- ✅ Complete aggregation and portfolio reporting
- ✅ SLA certificate generation
- ✅ Answer generation with safety checks
- ✅ Evidence-based and closed-book modes
- Simple, fast, production-ready

### Semantic v2 - Experimental (`scripts/hallucination_toolkit_semantic.py`)
Advanced meaning-level analysis with semantic entropy:
- ✅ Bidirectional entailment clustering
- ✅ Concurrent skeleton processing (6x faster)
- ✅ Prior-sufficiency gate & posterior-dominance latch
- ✅ Creative/Code mode with relaxed thresholds
- ❌ No aggregation or SLA certificates
- Research-oriented, better confabulation detection

## Installation

```bash
pip install --upgrade openai
export OPENAI_API_KEY=sk-...
```

## Quick Start Examples

### Classic Version with SLA Certificate

```python
from scripts.hallucination_toolkit import (
    OpenAIBackend, OpenAIItem, OpenAIPlanner,
    make_sla_certificate, save_sla_certificate_json,
    generate_answer_if_allowed
)

# Initialize
backend = OpenAIBackend(model="gpt-4o-mini")
planner = OpenAIPlanner(backend, temperature=0.3)

# Prepare items
items = [
    OpenAIItem(
        prompt="Who won the 2019 Nobel Prize in Physics?",
        n_samples=7,
        m=6,
        skeleton_policy="closed_book"
    ),
    OpenAIItem(
        prompt="""Task: Answer based on the evidence.
        Question: What is the speed of light?
        Evidence: The speed of light in vacuum is 299,792,458 m/s.""",
        n_samples=5,
        m=6,
        fields_to_erase=["Evidence"],
        skeleton_policy="evidence_erase"
    )
]

# Run evaluation
metrics = planner.run(
    items,
    h_star=0.05,           # Target max 5% hallucination
    isr_threshold=1.0,     # Information sufficiency threshold
    margin_extra_bits=0.2, # Safety margin (nats)
    B_clip=12.0,          # Clipping bound
    clip_mode="one-sided" # Conservative clipping
)

# Generate report and SLA certificate
report = planner.aggregate(
    items, 
    metrics, 
    alpha=0.05,  # 95% confidence
    h_star=0.05
)

cert = make_sla_certificate(report, model_name="GPT-4o-mini")
save_sla_certificate_json(cert, "sla_certificate.json")

print(f"Answer rate: {report.answer_rate:.1%}")
print(f"Empirical hallucination: {report.empirical_hallucination_rate:.1%}" 
      if report.empirical_hallucination_rate else "No labels")
print(f"Worst-case RoH bound: {report.worst_item_roh_bound:.3f}")

# Generate answers for allowed items
for item, metric in zip(items, metrics):
    if metric.decision_answer:
        answer = generate_answer_if_allowed(backend, item, metric)
        print(f"\nQ: {item.prompt[:50]}...")
        print(f"A: {answer}")
    else:
        print(f"\n[REFUSED] {item.prompt[:50]}...")
```

### Semantic Version (Experimental)

```python
from scripts.hallucination_toolkit_semantic import (
    OpenAIChatBackend, SemanticPlanner, OpenAIItem
)

# Initialize with semantic features
backend = OpenAIChatBackend(model="gpt-4o-mini", debug=True)
planner = SemanticPlanner(
    backend,
    temperature=1.0,
    top_p=0.9,
    max_tokens_answer=48,
    use_llr_top_meaning_budget=True,  # Recommended
    max_workers_skeletons=6,           # Parallel processing
    kl_smoothing_eps=1e-2
)

# Standard factual query
factual_item = OpenAIItem(
    prompt="Who won the Nobel Prize in Physics in 2014?",
    M_full=10,
    m_skeletons=6,
    n_samples_per_skeleton=6
)

# Creative/Code mode example
code_item = OpenAIItem(
    prompt="Write a Python function to calculate fibonacci numbers",
    M_full=16,  # More samples for style variation
    m_skeletons=6,
    n_samples_per_skeleton=4
)

# Run evaluations
factual_metrics = planner.run([factual_item], h_star=0.05, isr_threshold=1.0)
code_metrics = planner.run([code_item], h_star=0.30, isr_threshold=0.8)  # Relaxed for code

# Display semantic analysis
for metric in factual_metrics:
    print(f"Decision: {'ANSWER' if metric.decision_answer else 'REFUSE'}")
    print(f"Top meaning: {metric.top_meaning}")
    print(f"Posterior confidence: {metric.P_full_top:.1%}")
    print(f"Semantic entropy: {metric.H_sem:.3f}")
    print(f"RoH bound: {metric.roh_bound:.3f}")
```

## Core Concepts

### Mathematical Framework

The toolkit implements the EDFL (Expectation-level Decompression Law) framework:

1. **Rolling Priors**: Create weakened versions of prompts (skeletons)
2. **Information Budget**: Δ̄ = average clipped log-likelihood ratio between full and skeleton prompts
3. **Bits-to-Trust**: B2T = KL(Ber(1-h*) || Ber(q_lo)) - information needed for target reliability
4. **Decision Gate**: Answer if ISR = Δ̄/B2T ≥ threshold AND Δ̄ ≥ B2T + margin

### Deployment Modes

#### Evidence-based
For prompts with explicit context/evidence fields:
```python
item = OpenAIItem(
    prompt="Question: X\nEvidence: Y\nAnswer based on evidence.",
    fields_to_erase=["Evidence"],
    skeleton_policy="evidence_erase"
)
```

#### Closed-book
For prompts without evidence, using semantic masking:
```python
item = OpenAIItem(
    prompt="What happened in Paris in 1889?",
    skeleton_policy="closed_book"  # Masks entities, dates, numbers
)
```

#### Creative/Code Mode (Semantic only)
Relaxed thresholds for non-factual tasks:
```python
# Set h_star=0.30 (70% confidence) instead of 0.05 (95%)
# Still blocks fabricated facts about real entities
metrics = planner.run(items, h_star=0.30, isr_threshold=0.8)
```

## Feature Comparison

| Feature | Classic | Semantic v2 |
|---------|---------|-------------|
| **Core Model** | Binary (Answer/Refuse) | Meaning clusters |
| **Semantic Entropy** | ❌ | ✅ |
| **Concurrent Processing** | ❌ | ✅ (6x faster) |
| **Aggregation Reports** | ✅ | ❌ |
| **SLA Certificates** | ✅ | ❌ |
| **Answer Generation** | ✅ | ❌ |
| **Creative/Code Mode** | Limited | ✅ Full support |
| **Prior-Sufficiency Gate** | ❌ | ✅ |
| **Best For** | Production, compliance | Research, complex QA |

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `h_star` | 0.05 | Target max hallucination rate (0.05 = 5%) |
| `isr_threshold` | 1.0 | Information sufficiency ratio threshold |
| `margin_extra_bits` | 0.2 | Additional safety margin (nats) |
| `temperature` | 0.3-1.0 | Sampling temperature (lower = more stable) |
| `m` / `m_skeletons` | 6 | Number of skeleton variants |
| `n_samples` | 5-7 | Samples per skeleton (Classic) |
| `M_full` | 10 | Full prompt samples (Semantic) |
| `B_clip` | 12.0 | Clipping bound for Δ̄ computation |

### Tuning Guidelines

**Strict Factual QA**: 
- `h_star=0.02-0.05` (98-95% reliability)
- `temperature=0.3`
- Higher sample counts

**Creative/Code Generation**:
- `h_star=0.20-0.40` (80-60% reliability)  
- `temperature=0.8-1.0`
- Constrain output format to reduce variation

**High Abstention Rate?**
- Increase `n_samples` for stability
- Reduce `margin_extra_bits`
- Check if skeletons are sufficiently weakened

## API Reference

### Classic Version

```python
# Backend
backend = OpenAIBackend(model="gpt-4o-mini", api_key=None)

# Item
item = OpenAIItem(
    prompt="...",
    n_samples=7,
    m=6,
    skeleton_policy="auto",  # "auto", "evidence_erase", "closed_book"
    fields_to_erase=["Evidence"],  # For evidence mode
)

# Planner
planner = OpenAIPlanner(backend, temperature=0.3, q_floor=None)
metrics = planner.run(items, h_star=0.05, isr_threshold=1.0)
report = planner.aggregate(items, metrics, alpha=0.05)

# SLA Certificate
cert = make_sla_certificate(report, model_name="gpt-4o-mini")
save_sla_certificate_json(cert, "path.json")

# Answer Generation
answer = generate_answer_if_allowed(backend, item, metric)
```

### Semantic Version

```python
# Backend with logprobs
backend = OpenAIChatBackend(
    model="gpt-4o-mini",
    debug=True,
    strict=False
)

# Planner with semantic features
planner = SemanticPlanner(
    backend,
    entailer=None,  # Auto-creates OpenAILLMEntailer
    temperature=1.0,
    use_llr_top_meaning_budget=True,
    max_workers_skeletons=6,
    max_entail_calls=120
)

# Item with semantic parameters
item = OpenAIItem(
    prompt="...",
    M_full=10,
    m_skeletons=6,
    n_samples_per_skeleton=6
)

# Run with parallel item processing
metrics = planner.run(items, h_star=0.05, max_workers_items=8)
```

## Output Metrics

Both versions return `ItemMetrics` with:
- `decision_answer`: Boolean (ANSWER/REFUSE)
- `delta_bar`: Information budget (nats)  
- `roh_bound`: EDFL hallucination risk bound
- `isr`: Information Sufficiency Ratio
- `q_conservative`: Worst-case prior
- `rationale`: Human-readable explanation

Semantic version adds:
- `H_sem`: Semantic entropy over meanings
- `P_full_top`: Top meaning posterior probability
- `top_meaning`: Representative answer for top cluster

## Performance Characteristics

| Metric | Classic | Semantic |
|--------|---------|----------|
| Latency/item | 2-3 sec | 2-5 sec |
| API calls | (1+m)×n | (1+m)×n + entailment |
| Cost/item | $0.01-0.02 | $0.02-0.04 |
| Parallelism | No | Yes (6x speedup) |

## Troubleshooting

**High abstention rate:**
- Classic: Increase `n_samples`, lower `temperature`
- Semantic: Increase `M_full`, check if `P_full_top` is low

**Prior collapse (q_lo → 0):**
- Apply Laplace smoothing: `q_floor=1/(n+2)`
- Reduce masking strength in closed-book mode

**Creative tasks abstaining:**
- Use Semantic version with `h_star=0.30`
- Constrain output format in prompt
- Add skip-list for technical terms

## Web Interface

```bash
streamlit run app/web/web_app.py
```

Provides:
- Engine selection (Classic vs Semantic)
- Interactive parameter tuning  
- Real-time risk visualization
- SLA certificate export (Classic only)

## Contributing

We welcome contributions, criticisms, and real-world deployment experiences. Areas of interest:
- Alternative clustering methods
- Multi-model support beyond OpenAI
- Optimized skeleton generation strategies
- Production deployment case studies

## License

MIT License

## Attribution

Developed by Hassana Labs (https://hassana.io).

This implementation follows the framework from the paper “Compression Failure in LLMs: Bayesian in Expectation, Not in Realization” and related EDFL/ISR/B2T methodology.

V2 contains an implementation based on the framework from "Detecting hallucinations in large language models using semantic entropy" (Nature, 2024).