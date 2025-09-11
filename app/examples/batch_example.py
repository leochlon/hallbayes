"""
Batch Processing Example for HallBayes
=====================================

This script shows how to process multiple prompts efficiently and 
generate useful reports and comparisons.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from scripts.hallucination_toolkit import (
    OpenAIBackend,
    OpenAIItem,
    OpenAIPlanner,
)
from scripts.utils import (
    create_batch_from_prompts,
    print_summary,
    save_results_to_json,
    save_results_to_csv,
    create_comparison_report,
    validate_openai_setup,
)


def main():
    # Validate setup first
    if not validate_openai_setup():
        return
    
    print("Running batch evaluation example...")
    
    # Example prompts for testing
    test_prompts = [
        "What is the capital of France?",
        "If I have 10 apples and eat 3, how many do I have left?",
        "Who won the 2020 Nobel Prize in Physics?",
        "What is 2 + 2?",
        "Explain quantum entanglement in simple terms.",
        "What year was the iPhone first released?",
        "How many planets are in our solar system?",
        "What is the square root of 144?",
    ]
    
    # Create backend and items
    backend = OpenAIBackend(model="gpt-4o-mini")
    items = create_batch_from_prompts(
        test_prompts,
        n_samples=5,
        m=6,
        skeleton_policy="closed_book"
    )
    
    # Test with conservative settings
    print("\n--- Running with CONSERVATIVE settings ---")
    planner_conservative = OpenAIPlanner(backend, temperature=0.3)
    metrics_conservative = planner_conservative.run(
        items,
        h_star=0.05,  # 5% max hallucination
        isr_threshold=1.0,
        margin_extra_bits=0.2,
        B_clip=12.0,
        clip_mode="one-sided"
    )
    
    print_summary(metrics_conservative)
    
    # Test with more permissive settings
    print("\n--- Running with PERMISSIVE settings ---")
    planner_permissive = OpenAIPlanner(backend, temperature=0.3)
    metrics_permissive = planner_permissive.run(
        items,
        h_star=0.10,  # 10% max hallucination
        isr_threshold=0.8,  # Lower threshold
        margin_extra_bits=0.1,  # Less margin
        B_clip=12.0,
        clip_mode="one-sided"
    )
    
    print_summary(metrics_permissive)
    
    # Compare the two approaches
    create_comparison_report(
        metrics_conservative, 
        metrics_permissive,
        "Conservative (h*=5%)", 
        "Permissive (h*=10%)"
    )
    
    # Save results
    print("\nSaving results...")
    save_results_to_json(metrics_conservative, "results_conservative.json")
    save_results_to_json(metrics_permissive, "results_permissive.json")
    save_results_to_csv(metrics_conservative, "results_conservative.csv")
    save_results_to_csv(metrics_permissive, "results_permissive.csv")
    
    # Show detailed results for a few items
    print("\n--- DETAILED RESULTS (first 3 items) ---")
    for i, (item, metric_c, metric_p) in enumerate(zip(items[:3], metrics_conservative[:3], metrics_permissive[:3])):
        print(f"\nPrompt {i+1}: {item.prompt}")
        print(f"Conservative: {'ANSWER' if metric_c.decision_answer else 'REFUSE'} (ISR={metric_c.isr:.2f}, RoH={metric_c.roh_bound:.3f})")
        print(f"Permissive:   {'ANSWER' if metric_p.decision_answer else 'REFUSE'} (ISR={metric_p.isr:.2f}, RoH={metric_p.roh_bound:.3f})")


if __name__ == "__main__":
    main()
