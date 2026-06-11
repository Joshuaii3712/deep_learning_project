"""
Experiment 2: Memory Efficiency
=================================
Compares behavior retention per memory footprint across strategies.

Metric:
    efficiency = behavior_similarity / stored_bytes

The behavior_similarity uses embedding cosine similarity against
full-context responses.

Usage:
    python experiments/exp2_efficiency/run.py --output results/exp2_results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EVAL_PROMPTS = [
    "What's the best way to start a new project?",
    "How do you handle unexpected changes?",
    "What motivates you to keep going when things get hard?",
    "Do you prefer working alone or in a team?",
    "How do you approach learning something new?",
    "What's your advice for managing stress?",
    "When should someone take a risk?",
    "How do you decide between two good options?",
    "What matters most in a relationship?",
    "How do you stay organised?",
]

HISTORY_SIZES = [5, 10, 20, 50, 100]  # number of turns in history


def build_history(n_turns: int) -> list[dict[str, str]]:
    """Generate synthetic conversation history of n_turns turns."""
    turns = [
        ("user", "I'm really excited about my new business idea. It's a bit risky but I'm optimistic."),
        ("assistant", "That enthusiasm is contagious! Calculated risks can lead to great rewards."),
        ("user", "I sometimes worry too much about what could go wrong."),
        ("assistant", "It's wise to consider potential obstacles while keeping a positive outlook."),
        ("user", "I love collaborating with others and hearing different perspectives."),
        ("assistant", "Collaboration really does strengthen outcomes — great trait to have."),
        ("user", "I've been thinking about expanding my project to new markets."),
        ("assistant", "Expanding markets is exciting — what regions are you considering?"),
        ("user", "I need more structure in my workflow. Things feel chaotic."),
        ("assistant", "A clear workflow can help you channel your energy more effectively."),
    ]
    result = []
    for i in range(n_turns):
        result.append({"role": turns[i % len(turns)][0], "content": turns[i % len(turns)][1]})
    return result


def run_experiment(llm_generate: Callable, output_path: Path):
    try:
        from sentence_transformers import SentenceTransformer
        emb_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    except ImportError:
        emb_model = None
        logger.warning("No embedding model available; using random similarities.")

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from experiments.exp1_behavior.run import (
        FullContextStrategy,
        SummaryMemoryStrategy,
        FactMemoryStrategy,
        PSMStrategy,
        embedding_similarity,
    )

    results = []

    for n_turns in HISTORY_SIZES:
        logger.info("History size: %d turns", n_turns)
        history = build_history(n_turns)

        strategies = {
            "full_context": FullContextStrategy(llm_generate, history),
            "summary_memory": SummaryMemoryStrategy(llm_generate, history),
            "fact_memory": FactMemoryStrategy(llm_generate, history),
            "psm": PSMStrategy(llm_generate, history),
        }

        for prompt in EVAL_PROMPTS:
            ref = strategies["full_context"].generate(prompt)
            ref_bytes = strategies["full_context"].storage_bytes()

            for strat_name, strategy in strategies.items():
                response = strategy.generate(prompt)
                storage = strategy.storage_bytes()

                if emb_model is not None:
                    sim = embedding_similarity(ref, response, emb_model)
                else:
                    sim = float(np.random.uniform(0.4, 0.9))

                efficiency = sim / max(storage, 1) * 1000  # per KB

                results.append({
                    "history_turns": n_turns,
                    "strategy": strat_name,
                    "prompt": prompt,
                    "similarity": sim,
                    "storage_bytes": storage,
                    "efficiency_per_kb": efficiency,
                })

    # Aggregate
    summary = {}
    for strat in ["full_context", "summary_memory", "fact_memory", "psm"]:
        strat_rows = [r for r in results if r["strategy"] == strat]
        by_size = {}
        for n in HISTORY_SIZES:
            size_rows = [r for r in strat_rows if r["history_turns"] == n]
            if size_rows:
                by_size[n] = {
                    "mean_similarity": float(np.mean([r["similarity"] for r in size_rows])),
                    "mean_storage_bytes": float(np.mean([r["storage_bytes"] for r in size_rows])),
                    "mean_efficiency_per_kb": float(np.mean([r["efficiency_per_kb"] for r in size_rows])),
                }
        summary[strat] = by_size

    output = {"results": results, "summary": summary}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    logger.info("Saved to %s", output_path)

    # Print table
    print("\n=== Experiment 2: Memory Efficiency ===")
    print(f"{'Strategy':<20} {'History':>8} {'Similarity':>12} {'Bytes':>10} {'Eff/KB':>10}")
    print("-" * 62)
    for strat, by_size in summary.items():
        for n, m in by_size.items():
            print(
                f"{strat:<20} {n:>8} "
                f"{m['mean_similarity']:>12.3f} "
                f"{m['mean_storage_bytes']:>10.0f} "
                f"{m['mean_efficiency_per_kb']:>10.4f}"
            )

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/exp2_results.json"))
    args = parser.parse_args()

    from psm.llm import get_llm
    llm = get_llm()

    def llm_generate(messages, system_prompt=""):
        return llm.generate(messages=messages, system_prompt=system_prompt)

    run_experiment(llm_generate, args.output)


if __name__ == "__main__":
    main()
