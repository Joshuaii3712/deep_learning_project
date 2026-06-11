"""
Experiment 1: Behavior Preservation
====================================
Compare four memory strategies across identical evaluation prompts:
  - Full Context   : entire message history in context
  - Summary Memory : LLM-generated summary in system prompt
  - Fact Memory    : extracted key facts in system prompt
  - PSM            : personality profile in system prompt ONLY
                     (no conversation history at generation time)

PSM path (aligned with paper intent):
    history → personality estimation → personality state/profile
    → generation with [profile + eval_prompt] only

Measures:
  - LLM Judge Similarity   (0–10, same Llama 3.2 1.2B model)
  - Response Style Similarity  (cosine similarity via all-mpnet-base-v2)
  - Decision Consistency    (stance polarity match)

Usage:
    python experiments/exp1_behavior/run.py --turns 25
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Callable

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_EVAL_PROMPTS = [
    "Should I take a risk on a new business idea?",
    "How do you handle disagreements with colleagues?",
    "Describe your ideal weekend.",
    "What do you think about trying new foods?",
    "How would you deal with a difficult coworker?",
] * 5  # 25 prompts


# ── Memory strategies ──────────────────────────────────────────────────────────

class FullContextStrategy:
    name = "full_context"

    def __init__(self, llm_generate: Callable, history: list[dict]):
        self.llm_generate = llm_generate
        self.history = list(history)

    def generate(self, prompt: str) -> str:
        messages = self.history + [{"role": "user", "content": prompt}]
        return self.llm_generate(
            messages=messages,
            system_prompt="You are a helpful assistant.",
        )

    def storage_bytes(self) -> int:
        return sum(len(m["content"].encode()) for m in self.history)


class SummaryMemoryStrategy:
    name = "summary_memory"

    def __init__(self, llm_generate: Callable, history: list[dict]):
        self.llm_generate = llm_generate
        concat = "\n".join(f"{m['role']}: {m['content']}" for m in history)
        self.summary = llm_generate(
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


class FactMemoryStrategy:
    name = "fact_memory"

    def __init__(self, llm_generate: Callable, history: list[dict]):
        self.llm_generate = llm_generate
        concat = "\n".join(f"{m['role']}: {m['content']}" for m in history)
        raw = llm_generate(
            messages=[{"role": "user", "content":
                f"Extract 5 key facts as a JSON array of strings. No preamble:\n{concat}"}],
            system_prompt="Reply only with a valid JSON array of 5 strings.",
        )
        try:
            clean = raw.strip().strip("```json").strip("```").strip()
            self.facts = json.loads(clean)
            if not isinstance(self.facts, list):
                raise ValueError
        except Exception:
            self.facts = [raw[:120]]

    def generate(self, prompt: str) -> str:
        facts_text = "\n".join(f"- {f}" for f in self.facts)
        system = f"You are a helpful assistant.\n\nKnown facts:\n{facts_text}"
        return self.llm_generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=system,
        )

    def storage_bytes(self) -> int:
        return sum(len(f.encode()) for f in self.facts)


class PSMStrategy:
    """
    PSM path aligned with paper intent:
        history → personality estimation (user utterances only)
                → personality state (EMA, no further chat)
                → personality profile rendered
        generate(prompt):
            system = base_prompt + personality_profile
            messages = [{"role": "user", "content": prompt}]   # NO history
    """
    name = "psm"

    def __init__(self, llm_generate: Callable, history: list[dict]):
        import sys, uuid
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from psm.personality_estimator import PersonalityEstimator
        from psm.renderer import render_personality
        from psm.state import PersonalityState
        from config import BIG5_MODEL

        self.llm_generate = llm_generate

        # ── Step 1: estimate personality from user utterances only ─────────────
        user_text = "\n".join(
            m["content"] for m in history if m.get("role") == "user"
        )

        estimator = PersonalityEstimator(model_name=BIG5_MODEL)
        if user_text.strip():
            estimate = estimator.estimate(user_text)
        else:
            estimate = PersonalityState()

        # ── Step 2: use raw estimate directly (alpha=0, no EMA smoothing) ──────
        self._personality = estimate

        # ── Step 3: render profile ─────────────────────────────────────────────
        self._profile = render_personality(self._personality)

        logger.info(
            "PSMStrategy | personality=%s | profile_lines=%d",
            self._personality,
            self._profile.count("\n"),
        )

    def generate(self, prompt: str) -> str:
        # NO history in context — personality profile only
        base = "You are a helpful assistant."
        system = f"{base}\n\n{self._profile}" if self._profile else base
        return self.llm_generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=system,
        )

    def storage_bytes(self) -> int:
        # Report serialized personality payload (5 float64 values + dict overhead)
        return len(pickle.dumps(self._personality.to_dict()))


# ── Evaluation helpers ─────────────────────────────────────────────────────────

def embedding_similarity(text_a: str, text_b: str, model) -> float:
    emb_a = model.encode([text_a])[0]
    emb_b = model.encode([text_b])[0]
    cos = float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-9))
    return max(0.0, cos)


def llm_judge_similarity(prompt: str, response_a: str, response_b: str,
                          judge_generate: Callable) -> float:
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
    positive = {"yes", "recommend", "agree", "should", "good", "helpful", "benefit"}
    negative = {"no", "avoid", "disagree", "shouldn't", "bad", "risk", "caution"}

    def stance(text: str) -> str:
        lower = text.lower()
        pos = sum(w in lower for w in positive)
        neg = sum(w in lower for w in negative)
        return "positive" if pos > neg else ("negative" if neg > pos else "neutral")

    return stance(response_a) == stance(response_b)


# ── Main ───────────────────────────────────────────────────────────────────────

def run_experiment(history: list[dict], eval_prompts: list[str],
                   llm_generate: Callable, output_path: Path) -> dict:
    try:
        from sentence_transformers import SentenceTransformer
        emb_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        logger.info("Embedding model loaded.")
    except ImportError:
        emb_model = None
        logger.warning("sentence-transformers not available; EmbSim will be 0.500.")

    logger.info("Initialising strategies…")
    strategies = {
        "full_context":   FullContextStrategy(llm_generate, history),
        "summary_memory": SummaryMemoryStrategy(llm_generate, history),
        "fact_memory":    FactMemoryStrategy(llm_generate, history),
        "psm":            PSMStrategy(llm_generate, history),
    }

    ref = strategies["full_context"]
    results: list[dict] = []

    for i, prompt in enumerate(eval_prompts):
        logger.info("Prompt %d/%d: %s", i + 1, len(eval_prompts), prompt[:60])
        ref_response = ref.generate(prompt)
        ref_emb = emb_model.encode([ref_response])[0] if emb_model else None

        for name, strategy in strategies.items():
            if name == "full_context":
                continue

            response = strategy.generate(prompt)
            judge = llm_judge_similarity(prompt, ref_response, response, llm_generate)
            emb_sim = (
                float(np.dot(ref_emb, emb_model.encode([response])[0]) /
                      (np.linalg.norm(ref_emb) * np.linalg.norm(emb_model.encode([response])[0]) + 1e-9))
                if emb_model else 0.5
            )
            consist = decision_consistent(ref_response, response)

            results.append({
                "prompt_idx": i,
                "prompt": prompt,
                "strategy": name,
                "response": response,
                "reference_response": ref_response,
                "llm_judge_score": judge,
                "embedding_similarity": max(0.0, emb_sim),
                "decision_consistent": consist,
                "storage_bytes": strategy.storage_bytes(),
            })
            logger.info("  %s | judge=%.1f emb=%.3f consist=%s", name, judge, emb_sim, consist)

    summary: dict = {}
    for name in ["summary_memory", "fact_memory", "psm"]:
        rows = [r for r in results if r["strategy"] == name]
        summary[name] = {
            "mean_llm_judge":            round(float(np.mean([r["llm_judge_score"] for r in rows])), 4),
            "mean_embedding_similarity": round(float(np.mean([r["embedding_similarity"] for r in rows])), 4),
            "decision_consistency_rate": round(float(np.mean([r["decision_consistent"] for r in rows])), 4),
            "mean_storage_bytes":        round(float(np.mean([r["storage_bytes"] for r in rows])), 1),
        }

    output = {"results": results, "summary": summary}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    logger.info("Results saved → %s", output_path)

    print("\n=== Experiment 1: Behavior Preservation ===")
    print(f"{'Strategy':<20} {'LLM Judge':>10} {'Emb. Sim.':>10} {'Consistency':>12} {'Storage(B)':>12}")
    print("-" * 66)
    for name, m in summary.items():
        print(f"{name:<20} {m['mean_llm_judge']:>10.2f} {m['mean_embedding_similarity']:>10.3f} "
              f"{m['decision_consistency_rate']:>12.2%} {m['mean_storage_bytes']:>12.0f}")

    return output


def main():
    import sys
    import time
    start = time.time()
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    parser = argparse.ArgumentParser(description="Experiment 1: Behavior Preservation")
    parser.add_argument("--prompts-file", type=Path, default=None)
    parser.add_argument("--turns", type=int, default=25)
    parser.add_argument("--output", type=Path, default=Path("results/exp1_results.json"))
    args = parser.parse_args()

    if args.prompts_file and args.prompts_file.exists():
        with open(args.prompts_file) as f:
            eval_prompts = json.load(f)
    else:
        eval_prompts = DEFAULT_EVAL_PROMPTS[: args.turns]

    history = [
        {"role": "user",      "content": "I love exploring new ideas and creative projects."},
        {"role": "assistant", "content": "Creativity and curiosity are great strengths."},
        {"role": "user",      "content": "I also tend to worry a lot about outcomes."},
        {"role": "assistant", "content": "It's natural to consider risks carefully."},
        {"role": "user",      "content": "I enjoy working with others and hearing different views."},
        {"role": "assistant", "content": "Collaboration often leads to better outcomes."},
        {"role": "user",      "content": "I sometimes struggle to stay organised."},
        {"role": "assistant", "content": "Building small daily routines can help a lot."},
        {"role": "user",      "content": "I prefer cautious decisions over bold ones."},
        {"role": "assistant", "content": "A measured approach is often wise."},
    ]

    from psm.llm import get_llm
    llm = get_llm()

    def llm_generate(messages, system_prompt=""):
        return llm.generate(messages=messages, system_prompt=system_prompt)

    run_experiment(history, eval_prompts, llm_generate, args.output)
    end = time.time()
    logger.info("Experiment completed in %.2f seconds.", end - start)


if __name__ == "__main__":
    main()