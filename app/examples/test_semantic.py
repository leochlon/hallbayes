from scripts.hallucination_toolkit_semantic import (
    OpenAIChatBackend, SemanticPlanner, OpenAIItem
)

backend = OpenAIChatBackend(model="gpt-4o-mini", debug=True, strict=True)
backend.health_check()

planner = SemanticPlanner(
    backend,
    temperature=1.0, top_p=0.9,
    max_tokens_answer=48,
    max_entail_calls=120, entailment_timeout_s=15.0,
    use_llr_top_meaning_budget=True,   # strongly recommended
    max_workers_skeletons=6,           # concurrency across skeletons
    kl_smoothing_eps=1e-2,             # stable KL on multi-meaning priors
)

items = [OpenAIItem(
    prompt="Write an algorithm that generates prime numbers in python.",
    M_full=10, m_skeletons=6, n_samples_per_skeleton=6
)]
print(planner.run(items, h_star=0.05, isr_threshold=1.0)[0])

