"""
Example: Closed‑Book Hallucination Risk Check (EDFL / B2T / ISR)
----------------------------------------------------------------

This script demonstrates how to decide answer vs abstain without external
evidence, reporting Δ̄, B2T, ISR, and an EDFL RoH bound.
"""

from scripts.hallucination_toolkit import (
    OpenAIBackend,
    OpenAIItem,
    OpenAIPlanner,
    generate_answer_if_allowed,
)


def main() -> None:
    backend = OpenAIBackend(model="gpt-4o-mini")
    prompt = (
        "If James has five apples and eats three of them, "
        "how many apples does he have left?"
    )
    item = OpenAIItem(prompt=prompt, n_samples=7, m=6, skeleton_policy="closed_book")
    planner = OpenAIPlanner(backend, temperature=0.3)
    metrics = planner.run([item], h_star=0.05, isr_threshold=1.0, margin_extra_bits=0.0, B_clip=12.0, clip_mode="one-sided")
    for m in metrics:
        print(f"Answer? {m.decision_answer} | {m.rationale}")
        answer = generate_answer_if_allowed(backend, item, m, max_tokens_answer=64)
        if answer:
            print("Model answer:", answer)


if __name__ == "__main__":
    main()
# Copyright (c) 2024 Hassana Labs
# Licensed under the MIT License - see LICENSE file for details
