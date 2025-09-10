# Hallucination Risk Calculator & Toolkit

**Post-hoc calibration for large language models without retraining.** This toolkit provides mathematically-grounded hallucination detection and controlled abstention using the Expectation-level Decompression Law (EDFL).

## Key Features

- **Bounded hallucination risk** with transparent mathematical guarantees
- **Two decision modes**: Classical (binary) and Semantic (meaning-level)
- **Two deployment modes**: Evidence-based and Closed-book
- **No retraining required** - works with OpenAI Chat Completions API
- **Concurrent processing** for improved performance

## Quick Start

```python
from hallucination_toolkit_semantic_v2_4 import (
    OpenAIChatBackend, SemanticPlanner, OpenAIItem
)

# Initialize backend and planner
backend = OpenAIChatBackend(model="gpt-4o-mini", debug=True)
planner = SemanticPlanner(
    backend,
    temperature=1.0,
    top_p=0.9,
    max_tokens_answer=48,
    use_llr_top_meaning_budget=True,  # Recommended for v2.4
    max_workers_skeletons=6,            # Parallel skeleton processing
)

# Evaluate a query
items = [OpenAIItem(
    prompt="Who won the Nobel Prize in Physics in 2014?",
    M_full=10,
    m_skeletons=6,
    n_samples_per_skeleton=6
)]

metrics = planner.run(items, h_star=0.05, isr_threshold=1.0)
print(f"Decision: {'ANSWER' if metrics[0].decision_answer else 'REFUSE'}")
print(f"Risk bound: {metrics[0].roh_bound:.3f}")
```

## Installation

```bash
pip install --upgrade openai
export OPENAI_API_KEY=sk-...
```

## Core Concepts

### Mathematical Framework

The toolkit uses the EDFL principle to bound hallucination risk:

1. **Build rolling priors**: Create content-weakened versions of prompts (skeletons)
2. **Compute information budget**: Δ̄ = average clipped log-likelihood ratio
3. **Apply EDFL bound**: Hallucination risk ≤ 1 - p_max(Δ̄, q̄)
4. **Gate decision**: Answer if ISR ≥ threshold and Δ̄ ≥ Bits-to-Trust

### Deployment Modes

#### Evidence-based Mode
Use when prompts contain explicit evidence/context fields:

```python
prompt = """Task: Answer based on evidence.
Question: Who won the Nobel Prize in Physics in 2019?
Evidence: James Peebles (1/2); Michel Mayor & Didier Queloz (1/2).
"""
item = OpenAIItem(prompt=prompt, skeleton_policy="evidence_erase")
```

#### Closed-book Mode
Use for queries without explicit evidence:

```python
item = OpenAIItem(
    prompt="Who won the 2019 Nobel Prize in Physics?",
    skeleton_policy="closed_book"
)
```

### Decision Heads

#### Classical Head
- Models binary event: Answer vs Refuse
- Fast and robust
- Best for simple gating decisions

#### Semantic Head (v2.4)
- Models distribution over meanings via clustering
- Detects meaning-level uncertainty
- Includes posterior-dominance latch for safety
- Better at detecting confabulations

## Configuration

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `h_star` | 0.05 | Target max hallucination rate |
| `isr_threshold` | 1.0 | Information sufficiency threshold |
| `temperature` | 1.0 | Sampling temperature |
| `M_full` | 10 | Samples for full prompt |
| `m_skeletons` | 6 | Number of skeleton variants |
| `n_samples_per_skeleton` | 6 | Samples per skeleton |
| `max_workers_skeletons` | 6 | Parallel skeleton workers |
| `use_llr_top_meaning_budget` | False | Use LLR for budget (recommended) |

### Tuning for Different Use Cases

**Strict Factual QA:**
```python
planner.run(items, h_star=0.02, isr_threshold=1.0)
```

**Creative/Code Generation:**
```python
planner.run(items, h_star=0.30, isr_threshold=0.8)
```

## API Reference

### Core Classes

**`OpenAIChatBackend`**
```python
backend = OpenAIChatBackend(
    model="gpt-4o-mini",
    api_key=None,  # Uses OPENAI_API_KEY env var
    debug=False,
    strict=False
)
```

**`SemanticPlanner`**
```python
planner = SemanticPlanner(
    backend,
    temperature=1.0,
    top_p=0.9,
    max_tokens_answer=48,
    use_llr_top_meaning_budget=True,
    max_workers_skeletons=6,
    kl_smoothing_eps=1e-2
)
```

**`OpenAIItem`**
```python
item = OpenAIItem(
    prompt="Your question here",
    M_full=10,
    m_skeletons=6,
    n_samples_per_skeleton=6,
    skeleton_policy="auto"  # "auto", "evidence_erase", or "closed_book"
)
```

### Output Metrics

`ItemMetrics` contains:
- `decision_answer`: Boolean - whether to answer
- `delta_bar`: Information budget (nats)
- `roh_bound`: EDFL hallucination risk bound
- `H_sem`: Semantic entropy
- `P_full_top`: Top meaning posterior probability
- `isr`: Information Sufficiency Ratio
- `rationale`: Human-readable explanation

## Advanced Usage

### Parallel Processing
```python
# Process multiple items concurrently
metrics = planner.run(items, h_star=0.05, max_workers_items=8)
```

### Custom Entailment Provider
```python
from hallucination_toolkit_semantic_v2_4 import OpenAILLMEntailer

entailer = OpenAILLMEntailer(backend, request_timeout=15.0)
planner = SemanticPlanner(backend, entailer=entailer)
```

## Performance Characteristics

| Metric | Typical Value | Notes |
|--------|--------------|-------|
| Latency per item | 2-5 seconds | With default settings |
| API calls | (1+m) × ⌈n/batch⌉ | Can be parallelized |
| Cost | ~$0.01-0.03 per item | Using gpt-4o-mini |

## Troubleshooting

**Low Δ̄ (frequent abstentions):**
- Increase `n_samples_per_skeleton` for stability
- Lower `temperature` to 0.3-0.5
- Check if skeletons are sufficiently weakened

**Prior collapse (q_lo → 0):**
- Apply Laplace smoothing (default: 1/(n+2))
- Reduce masking strength for closed-book mode

**Arithmetic queries abstaining:**
- Switch to correctness event instead of answer/refuse
- Provide worked examples as evidence

## Web Interface

```bash
streamlit run app/web/web_app.py
```

The web interface provides:
- Engine selection (Classic vs Semantic)
- Interactive parameter tuning
- Real-time risk visualization
- Export of SLA certificates

## License

MIT License

## Attribution

Developed by Hassana Labs (https://hassana.io).

This implementation follows the framework from the paper “Compression Failure in LLMs: Bayesian in Expectation, Not in Realization” and related EDFL/ISR/B2T methodology.

V2 contains an implementation based on the framework from "Detecting hallucinations in large language models using semantic entropy" (Nature, 2024).