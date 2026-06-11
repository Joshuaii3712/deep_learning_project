"""
Experiment 1: Behavior Preservation
====================================
Compare four memory strategies across 50 identical evaluation prompts:
  - Full Context
  - Summary Memory
  - Fact Memory
  - Psychological State Memory (PSM)

Measures:
  - LLM Judge Similarity   (0–10 score from a judge LLM)
  - Response Style Similarity  (cosine similarity of embeddings)
  - Decision Consistency    (0/1 per prompt, fraction consistent)

Usage:
    python experiments/exp1_behavior/run.py \
        --prompts-file prompts.json \
        --turns 50 \
        --output results/exp1_results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Evaluation prompts ─────────────────────────────────────────────────────────
DEFAULT_EVAL_PROMPTS = [
    "Should I take a risk on a new business idea?",
    "How do you handle disagreements with colleagues?",
    "Describe your ideal weekend.",
    "What do you think about trying new foods?",
    "How would you deal with a difficult coworker?",
] * 10  # 50 prompts total

# ── Memory strategy interfaces ─────────────────────────────────────────────────

class BaseMemoryStrategy:
    """Abstract base for memory strategies."""

    name: str

    def __init__(self, llm_generate: Callable, history: list[dict]):
        self.llm_generate = llm_generate
        self.history = list(history)

    def generate(self, prompt: str) -> str:
        raise NotImplementedError

    def storage_bytes(self) -> int:
        raise NotImplementedError


class FullContextStrategy(BaseMemoryStrategy):
    name = "full_context"

    def generate(self, prompt: str) -> str:
        messages = self.history + [{"role": "user", "content": prompt}]
        return self.llm_generate(messages=messages, system_prompt="You are a helpful assistant.")

    def storage_bytes(self) -> int:
        return sum(len(m["content"].encode()) for m in self.history)


class SummaryMemoryStrategy(BaseMemoryStrategy):
    name = "summary_memory"

    def __init__(self, llm_generate: Callable, history: list[dict]):
        super().__init__(llm_generate, history)
        self.summary = self._summarize()

    def _summarize(self) -> str:
        if not self.history:
            return ""
        concat = "\n".join(f"{m['role']}: {m['content']}" for m in self.history[-20:])
        return self.llm_generate(
            messages=[{"role": "user", "content": f"Summarise this conversation in 3 sentences:\n{concat}"}],
            system_prompt="You are a concise summariser.",
        )

    def generate(self, prompt: str) -> str:
        system = f"You are a helpful assistant.\n\nConversation summary:\n{self.summary}"
        return self.llm_generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=system,
        )

    def storage_bytes(self) -> int:
        return len(self.summary.encode())


class FactMemoryStrategy(BaseMemoryStrategy):
    name = "fact_memory"

    def __init__(self, llm_generate: Callable, history: list[dict]):
        super().__init__(llm_generate, history)
        self.facts = self._extract_facts()

    def _extract_facts(self) -> list[str]:
        if not self.history:
            return []
        concat = "\n".join(f"{m['role']}: {m['content']}" for m in self.history[-20:])
        raw = self.llm_generate(
            messages=[{"role": "user", "content": f"Extract 5 key facts from this conversation as a JSON array of strings:\n{concat}"}],
            system_prompt="You are a precise fact extractor. Respond only with valid JSON.",
        )
        try:
            return json.loads(raw)
        except Exception:
            return [raw]

    def generate(self, prompt: str) -> str:
        facts_text = "\n".join(f"- {f}" for f in self.facts)
        system = f"You are a helpful assistant.\n\nKnown facts:\n{facts_text}"
        return self.llm_generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=system,
        )

    def storage_bytes(self) -> int:
        return sum(len(f.encode()) for f in self.facts)


class PSMStrategy(BaseMemoryStrategy):
    name = "psm"

    def __init__(self, llm_generate: Callable, history: list[dict]):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from psm import PSMAgent, PSMDatabase
        import uuid

        super().__init__(llm_generate, history)
        self._db = PSMDatabase(":memory:")
        self._agent = PSMAgent(
            session_id=str(uuid.uuid4()),
            db=self._db,
            system_prompt="You are a helpful assistant.",
        )
        # Replay history
        for msg in self.history:
            if msg["role"] == "user":
                self._agent.chat(msg["content"])

    def generate(self, prompt: str) -> str:
        return self._agent.chat(prompt)

    def storage_bytes(self) -> int:
        import sys, pickle
        p = self._agent.personality
        return len(pickle.dumps(p.to_dict()))


# ── Evaluation helpers ─────────────────────────────────────────────────────────

def embedding_similarity(text_a: str, text_b: str, model) -> float:
    emb_a = model.encode([text_a])
    emb_b = model.encode([text_b])
    cos = float(np.dot(emb_a[0], emb_b[0]) / (np.linalg.norm(emb_a[0]) * np.linalg.norm(emb_b[0]) + 1e-9))
    return max(0.0, cos)


def llm_judge_similarity(
    prompt: str,
    response_a: str,
    response_b: str,
    judge_generate: Callable,
) -> float:
    judge_prompt = (
        f"Question: {prompt}\n\n"
        f"Response A: {response_a}\n\n"
        f"Response B: {response_b}\n\n"
        "On a scale of 0-10, how similar are these responses in content and style? "
        "Reply with only a single integer."
    )
    raw = judge_generate(
        messages=[{"role": "user", "content": judge_prompt}],
        system_prompt="You are an objective evaluator. Reply with only a single integer 0-10.",
    )
    try:
        return float(raw.strip().split()[0])
    except Exception:
        return 5.0


def decision_consistent(response_a: str, response_b: str) -> bool:
    """Heuristic: responses are consistent if they share a key decision word."""
    positive_words = {"yes", "recommend", "agree", "should", "good", "helpful", "benefit"}
    negative_words = {"no", "avoid", "disagree", "shouldn't", "bad", "risk", "caution"}

    def stance(text: str) -> str:
        lower = text.lower()
        pos = sum(w in lower for w in positive_words)
        neg = sum(w in lower for w in negative_words)
        if pos > neg:
            return "positive"
        if neg > pos:
            return "negative"
        return "neutral"

    return stance(response_a) == stance(response_b)


# ── Main experiment ────────────────────────────────────────────────────────────

@dataclass
class PromptResult:
    prompt: str
    strategy_a: str
    strategy_b: str
    response_a: str
    response_b: str
    llm_judge_score: float
    embedding_similarity: float
    decision_consistent: bool


def run_experiment(
    history: list[dict[str, str]],
    eval_prompts: list[str],
    llm_generate: Callable,
    output_path: Path,
):
    """Run Experiment 1 comparing all four strategies."""
    try:
        from sentence_transformers import SentenceTransformer
        emb_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    except ImportError:
        logger.warning("sentence-transformers not available; using dummy embeddings")
        emb_model = None

    logger.info("Initialising strategies…")
    strategies = {
        "full_context": FullContextStrategy(llm_generate, history),
        "summary_memory": SummaryMemoryStrategy(llm_generate, history),
        "fact_memory": FactMemoryStrategy(llm_generate, history),
        "psm": PSMStrategy(llm_generate, history),
    }

    # Use full_context as the reference baseline
    reference_strategy = strategies["full_context"]
    results: list[dict] = []

    for i, prompt in enumerate(eval_prompts):
        logger.info("Prompt %d/%d: %s", i + 1, len(eval_prompts), prompt[:60])
        ref_response = reference_strategy.generate(prompt)

        for strat_name, strategy in strategies.items():
            if strat_name == "full_context":
                continue

            response = strategy.generate(prompt)

            judge_score = llm_judge_similarity(prompt, ref_response, response, llm_generate)

            if emb_model is not None:
                emb_sim = embedding_similarity(ref_response, response, emb_model)
            else:
                emb_sim = 0.5

            decision_ok = decision_consistent(ref_response, response)

            result = {
                "prompt_idx": i,
                "prompt": prompt,
                "strategy": strat_name,
                "reference": "full_context",
                "response": response,
                "reference_response": ref_response,
                "llm_judge_score": judge_score,
                "embedding_similarity": emb_sim,
                "decision_consistent": decision_ok,
                "storage_bytes": strategy.storage_bytes(),
            }
            results.append(result)
            logger.info(
                "  %s | judge=%.1f emb=%.3f consist=%s",
                strat_name, judge_score, emb_sim, decision_ok,
            )

    # Summary
    summary: dict = {}
    for strat_name in ["summary_memory", "fact_memory", "psm"]:
        strat_results = [r for r in results if r["strategy"] == strat_name]
        summary[strat_name] = {
            "mean_llm_judge": float(np.mean([r["llm_judge_score"] for r in strat_results])),
            "mean_embedding_similarity": float(np.mean([r["embedding_similarity"] for r in strat_results])),
            "decision_consistency_rate": float(np.mean([r["decision_consistent"] for r in strat_results])),
            "mean_storage_bytes": float(np.mean([r["storage_bytes"] for r in strat_results])),
        }

    output = {"results": results, "summary": summary}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    logger.info("Results saved to %s", output_path)

    # Print summary table
    print("\n=== Experiment 1: Behavior Preservation ===")
    print(f"{'Strategy':<20} {'LLM Judge':>10} {'Emb. Sim.':>10} {'Consistency':>12} {'Storage(B)':>12}")
    print("-" * 66)
    for strat_name, metrics in summary.items():
        print(
            f"{strat_name:<20} "
            f"{metrics['mean_llm_judge']:>10.2f} "
            f"{metrics['mean_embedding_similarity']:>10.3f} "
            f"{metrics['decision_consistency_rate']:>12.2%} "
            f"{metrics['mean_storage_bytes']:>12.0f}"
        )

    return output


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    parser = argparse.ArgumentParser(description="Experiment 1: Behavior Preservation")
    parser.add_argument("--prompts-file", type=Path, default=None)
    parser.add_argument("--turns", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("results/exp1_results.json"))
    args = parser.parse_args()

    if args.prompts_file and args.prompts_file.exists():
        with open(args.prompts_file) as f:
            eval_prompts = json.load(f)
    else:
        eval_prompts = DEFAULT_EVAL_PROMPTS[: args.turns]

    # Build a dummy history
    history = [
        {"role": "user", "content": "I love exploring new ideas and creative projects."},
        {"role": "assistant", "content": "That's wonderful! Creativity and curiosity are great strengths."},
        {"role": "user", "content": "I also tend to worry a lot about outcomes."},
        {"role": "assistant", "content": "It's natural to consider risks carefully."},
    ]

    from psm.llm import get_llm
    llm = get_llm()

    def llm_generate(messages, system_prompt=""):
        return llm.generate(messages=messages, system_prompt=system_prompt)

    run_experiment(history, eval_prompts, llm_generate, args.output)


if __name__ == "__main__":
    main()
