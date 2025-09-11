"""
Utility functions for the HallBayes toolkit
"""

import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import asdict
import csv
import os

from .hallucination_toolkit import ItemMetrics, AggregateReport, OpenAIItem


def save_results_to_json(metrics: List[ItemMetrics], filepath: str) -> None:
    """Save evaluation results to JSON file."""
    data = [asdict(m) for m in metrics]
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {filepath}")


def save_results_to_csv(metrics: List[ItemMetrics], filepath: str) -> None:
    """Save evaluation results to CSV file."""
    if not metrics:
        return
    
    fieldnames = list(asdict(metrics[0]).keys())
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in metrics:
            row = asdict(m)
            # Convert complex objects to strings for CSV
            for key, value in row.items():
                if isinstance(value, (dict, list)):
                    row[key] = json.dumps(value)
            writer.writerow(row)
    print(f"Results saved to {filepath}")


def print_summary(metrics: List[ItemMetrics]) -> None:
    """Print a nice summary of evaluation results."""
    if not metrics:
        print("No metrics to summarize.")
        return
    
    answered = sum(1 for m in metrics if m.decision_answer)
    total = len(metrics)
    
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total items evaluated: {total}")
    print(f"Items answered: {answered}")
    print(f"Items refused: {total - answered}")
    print(f"Answer rate: {answered/total:.1%}")
    
    if answered > 0:
        answered_metrics = [m for m in metrics if m.decision_answer]
        avg_delta = sum(m.delta_bar for m in answered_metrics) / len(answered_metrics)
        avg_roh = sum(m.roh_bound for m in answered_metrics) / len(answered_metrics)
        max_roh = max(m.roh_bound for m in answered_metrics)
        
        print(f"\nFor answered items:")
        print(f"Average Δ̄: {avg_delta:.4f} nats")
        print(f"Average RoH bound: {avg_roh:.3f}")
        print(f"Worst RoH bound: {max_roh:.3f}")
    
    print("="*60)


def create_batch_from_prompts(prompts: List[str], **item_kwargs) -> List[OpenAIItem]:
    """Create a batch of OpenAIItem objects from a list of prompts."""
    return [OpenAIItem(prompt=prompt, **item_kwargs) for prompt in prompts]


def load_prompts_from_file(filepath: str) -> List[str]:
    """Load prompts from a text file (one per line)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def create_comparison_report(metrics1: List[ItemMetrics], metrics2: List[ItemMetrics], 
                           label1: str = "Config 1", label2: str = "Config 2") -> None:
    """Compare two different evaluation runs."""
    def get_stats(metrics):
        answered = [m for m in metrics if m.decision_answer]
        return {
            'total': len(metrics),
            'answered': len(answered),
            'answer_rate': len(answered) / len(metrics) if metrics else 0,
            'avg_delta': sum(m.delta_bar for m in answered) / len(answered) if answered else 0,
            'avg_roh': sum(m.roh_bound for m in answered) / len(answered) if answered else 0,
            'max_roh': max(m.roh_bound for m in answered) if answered else 0,
        }
    
    stats1 = get_stats(metrics1)
    stats2 = get_stats(metrics2)
    
    print("\n" + "="*80)
    print("COMPARISON REPORT")
    print("="*80)
    print(f"{'Metric':<25} {label1:<20} {label2:<20} {'Difference':<15}")
    print("-"*80)
    print(f"{'Total items':<25} {stats1['total']:<20} {stats2['total']:<20} {stats2['total'] - stats1['total']:<15}")
    print(f"{'Items answered':<25} {stats1['answered']:<20} {stats2['answered']:<20} {stats2['answered'] - stats1['answered']:<15}")
    print(f"{'Answer rate':<25} {stats1['answer_rate']:<20.1%} {stats2['answer_rate']:<20.1%} {stats2['answer_rate'] - stats1['answer_rate']:<+14.1%}")
    print(f"{'Avg Δ̄ (answered)':<25} {stats1['avg_delta']:<20.4f} {stats2['avg_delta']:<20.4f} {stats2['avg_delta'] - stats1['avg_delta']:<+14.4f}")
    print(f"{'Avg RoH (answered)':<25} {stats1['avg_roh']:<20.3f} {stats2['avg_roh']:<20.3f} {stats2['avg_roh'] - stats1['avg_roh']:<+14.3f}")
    print(f"{'Max RoH (answered)':<25} {stats1['max_roh']:<20.3f} {stats2['max_roh']:<20.3f} {stats2['max_roh'] - stats1['max_roh']:<+14.3f}")
    print("="*80)


def retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0):
    """Retry a function with exponential backoff."""
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries:
                raise e
            delay = base_delay * (2 ** attempt)
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
            time.sleep(delay)


def validate_openai_setup() -> bool:
    """Check if OpenAI is properly configured."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in environment variables")
        print("Please set it with: export OPENAI_API_KEY=your_api_key_here")
        return False
    
    if not api_key.startswith("sk-"):
        print("ERROR: OPENAI_API_KEY doesn't look valid (should start with 'sk-')")
        return False
    
    print("OK: OpenAI API key found and looks valid")
    return True


def estimate_tokens(text: str) -> int:
    """Rough estimation of tokens in text (approximately 4 chars per token)."""
    return len(text) // 4
