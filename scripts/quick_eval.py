#!/usr/bin/env python3
"""
Quick Evaluation Tool for HallBayes
===================================

A simple command-line tool for quick hallucination risk evaluation.

Usage:
    python scripts/quick_eval.py "Your prompt here"
    python scripts/quick_eval.py --file prompts.txt
    python scripts/quick_eval.py --interactive
"""

import argparse
import sys
import os
from typing import List

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.hallucination_toolkit import (
    OpenAIBackend,
    OpenAIItem,
    OpenAIPlanner,
    generate_answer_if_allowed,
)
from scripts.utils import (
    validate_openai_setup,
    load_prompts_from_file,
    create_batch_from_prompts,
    print_summary,
    save_results_to_json,
)


def evaluate_single_prompt(prompt: str, backend: OpenAIBackend, **kwargs) -> None:
    """Evaluate a single prompt and show results."""
    print(f"\nEvaluating: {prompt}")
    print("-" * 50)
    
    item = OpenAIItem(prompt=prompt, n_samples=5, m=6, skeleton_policy="closed_book")
    planner = OpenAIPlanner(backend, temperature=0.3)
    
    metrics = planner.run([item], **kwargs)
    metric = metrics[0]
    
    print(f"Decision: {'ANSWER' if metric.decision_answer else 'REFUSE'}")
    print(f"ISR: {metric.isr:.3f}")
    print(f"Delta_bar: {metric.delta_bar:.4f} nats")
    print(f"B2T: {metric.b2t:.4f} nats")
    print(f"RoH bound: {metric.roh_bound:.3f}")
    print(f"Rationale: {metric.rationale}")
    
    if metric.decision_answer:
        print("\nGenerating answer...")
        answer = generate_answer_if_allowed(backend, item, metric)
        if answer:
            print(f"Answer: {answer}")


def interactive_mode(backend: OpenAIBackend) -> None:
    """Interactive prompt evaluation mode."""
    print("Interactive mode - enter prompts to evaluate (type 'quit' to exit)")
    print("Settings: h_star=0.05, conservative mode")
    
    while True:
        try:
            prompt = input("\nPrompt> ").strip()
            if prompt.lower() in ('quit', 'exit', 'q'):
                break
            if not prompt:
                continue
                
            evaluate_single_prompt(
                prompt, backend,
                h_star=0.05,
                isr_threshold=1.0,
                margin_extra_bits=0.2
            )
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Quick hallucination risk evaluation")
    parser.add_argument("prompt", nargs="?", help="Prompt to evaluate")
    parser.add_argument("--file", "-f", help="File containing prompts (one per line)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--model", "-m", default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--h-star", type=float, default=0.05, help="Target hallucination rate (default: 0.05)")
    parser.add_argument("--permissive", action="store_true", help="Use more permissive settings")
    parser.add_argument("--output", "-o", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Validate OpenAI setup
    if not validate_openai_setup():
        sys.exit(1)
    
    backend = OpenAIBackend(model=args.model)
    
    # Determine evaluation settings
    if args.permissive:
        eval_kwargs = {
            "h_star": args.h_star,
            "isr_threshold": 0.8,
            "margin_extra_bits": 0.1
        }
        print(f"Using PERMISSIVE settings (h*={args.h_star}, ISR=0.8)")
    else:
        eval_kwargs = {
            "h_star": args.h_star,
            "isr_threshold": 1.0,
            "margin_extra_bits": 0.2
        }
        print(f"Using CONSERVATIVE settings (h*={args.h_star}, ISR=1.0)")
    
    # Handle different input modes
    if args.interactive:
        interactive_mode(backend)
        
    elif args.file:
        print(f"Loading prompts from {args.file}...")
        prompts = load_prompts_from_file(args.file)
        print(f"Loaded {len(prompts)} prompts")
        
        items = create_batch_from_prompts(prompts, n_samples=5, m=6, skeleton_policy="closed_book")
        planner = OpenAIPlanner(backend, temperature=0.3)
        
        print("Processing batch...")
        metrics = planner.run(items, **eval_kwargs)
        
        print_summary(metrics)
        
        if args.output:
            save_results_to_json(metrics, args.output)
            
    elif args.prompt:
        evaluate_single_prompt(args.prompt, backend, **eval_kwargs)
        
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python scripts/quick_eval.py \"What is 2+2?\"")
        print("  python scripts/quick_eval.py --file my_prompts.txt")
        print("  python scripts/quick_eval.py --interactive")


if __name__ == "__main__":
    main()
