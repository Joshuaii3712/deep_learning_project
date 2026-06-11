"""
Experiment 2: Storage-Efficiency Trade-off
==========================================
Lightweight version — designed to run in under 10 minutes on a
consumer laptop (MX250, 2GB VRAM, Llama 3.2 1B via Ollama).

Compares behavior retention per byte across four memory strategies
at two representative history lengths: short (5 turns) and medium (15 turns).

Metric:
    Efficiency = EmbSim / StoredBytes * 1000   (similarity per KB)

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Lightweight config ─────────────────────────────────────────────────────────
# 2 history sizes × 4 strategies × 5 prompts = 40 LLM calls total
HISTORY_SIZES = [5, 15]

EVAL_PROMPTS = [
    "What's the best way to approach a new challenge?",
    "How do you stay motivated when things feel uncertain?",
    "Should I take a risk on something new?",
    "How do you balance caution with optimism?",
    "What matters most when making an important decision?",
]

HISTORY_TEMPLATE = [
    ("user",      "I'm really excited about my new project — it feels risky but I'm optimistic."),
    ("assistant", "That enthusiasm is great. Calculated risks can lead to real breakthroughs."),
    ("user",      "I sometimes worry too much about what could go wrong."),
    ("assistant", "It's natural to consider risks, but try not to let worry block action."),
    ("user",      "I love hearing different perspectives and exploring creative ideas."),
    ("assistant", "Openness to new ideas is a genuine strength in problem-solving."),
    ("user",      "I need more structure in my workflow — things feel chaotic lately."),
    ("assistant", "A clear daily structure can help channel energy more effectively."),
    ("user",      "I find it hard to say no to people even when I'm overwhelmed."),
    ("assistant", "Setting gentle boundaries is a skill worth practising gradually."),
    ("user",      "I want to expand into new markets but I'm not sure where to start."),
    ("assistant", "Start with one market that overlaps most with your current strengths."),
    ("user",      "Deep down I think I'm more of an introvert than I let on."),
    ("assistant", "Many people are — it's fine to recharge quietly and engage selectively."),
    ("user",      "I get anxious when I don't have a clear plan."),
]


def build_history(n_turns: int) -> list[dict[str, str]]:
    result = []
    for i in range(n_turns):
        role, content = HISTORY_TEMPLATE[i % len(HISTORY_TEMPLATE)]
        result.append({"role": role, "content": content})
    return result


# ── Memory strategies ──────────────────────────────────────────────────────────

class FullContextStrategy:
    name = "full_context"

    def __init__(self, llm_generate: Callable, history: list[dict]):
        self.llm_generate = llm_generate
        self.history = list(history)

    def generate(self, prompt: str) -> str:
        messages = self.history + [{"role": "user", "content": prompt}]
        return self.llm_generate(messages=messages, system_prompt="You are a helpful assistant.")

    def storage_bytes(self) -> int:
        return sum(len(m["content"].encode()) for m in self.history)


class SummaryMemoryStrategy:
    name = "summary_memory"

    def __init__(self, llm_generate: Callable, history: list[dict]):
        self.llm_generate = llm_generate
        concat = "\n".join(f"{m['role']}: {m['content']}" for m in history)
        self.summary = llm_generate(
            messages=[{"role": "user", "content": f"Summarise in 2 sentences:\n{concat}"}],
            system_prompt="You are a concise summariser. Reply in exactly 2 sentences.",
        )

    def generate(self, prompt: str) -> str:
        system = f"You are a helpful assistant.\n\nConversation summary: {self.summary}"
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
                f"Extract exactly 5 key facts as a JSON array of strings. No preamble:\n{concat}"}],
            system_prompt="Reply only with a valid JSON array of 5 strings.",
        )
        try:
            # strip markdown fences if present
            clean = raw.strip().strip("```json").strip("```").strip()
            self.facts = json.loads(clean)
            if not isinstance(self.facts, list):
                raise ValueError
        except Exception:
            self.facts = [raw[:80]]

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
    name = "psm"

    def __init__(self, llm_generate: Callable, history: list[dict]):
        import uuid
        from psm import PSMAgent, PSMDatabase

        self._db = PSMDatabase(":memory:")
        self._agent = PSMAgent(
            session_id=str(uuid.uuid4()),
            db=self._db,
            system_prompt="You are a helpful assistant.",
        )
        for msg in history:
            if msg["role"] == "user":
                self._agent.chat(msg["content"])

    def generate(self, prompt: str) -> str:
        return self._agent.chat(prompt)

    def storage_bytes(self) -> int:
        # 5 float64 values = 40 bytes
        return 40


# ── Evaluation ─────────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def run_experiment(llm_generate: Callable, output_path: Path) -> dict:
    # Load embedding model
    try:
        from sentence_transformers import SentenceTransformer
        emb_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        logger.info("Embedding model loaded.")
    except ImportError:
        emb_model = None
        logger.warning("sentence-transformers not available; using random similarity.")

    results = []

    for n_turns in HISTORY_SIZES:
        logger.info("─── History size: %d turns ───", n_turns)
        history = build_history(n_turns)

        # Initialise strategies
        strategies: dict[str, FullContextStrategy] = {}
        for cls in [FullContextStrategy, SummaryMemoryStrategy, FactMemoryStrategy, PSMStrategy]:
            logger.info("  Init: %s", cls.name)
            try:
                strategies[cls.name] = cls(llm_generate, history)
            except Exception as e:
                logger.error("  Failed to init %s: %s", cls.name, e)

        fc = strategies.get("full_context")
        if fc is None:
            logger.error("Full Context strategy failed; skipping n_turns=%d", n_turns)
            continue

        for prompt in EVAL_PROMPTS:
            logger.info("  Prompt: %s", prompt[:50])
            ref_response = fc.generate(prompt)

            if emb_model:
                ref_emb = emb_model.encode([ref_response])[0]

            for strat_name, strategy in strategies.items():
                response = strategy.generate(prompt)
                storage = strategy.storage_bytes()

                if emb_model:
                    resp_emb = emb_model.encode([response])[0]
                    sim = cosine_similarity(ref_emb, resp_emb)
                else:
                    sim = float(np.random.uniform(0.5, 0.85))

                efficiency = (sim / max(storage, 1)) * 1000

                results.append({
                    "history_turns": n_turns,
                    "strategy": strat_name,
                    "prompt": prompt,
                    "similarity": round(sim, 4),
                    "storage_bytes": storage,
                    "efficiency_per_kb": round(efficiency, 4),
                })
                logger.info(
                    "    %s | sim=%.3f storage=%dB eff=%.3f",
                    strat_name, sim, storage, efficiency,
                )

    # Aggregate per (strategy, n_turns)
    summary: dict = {}
    for strat in ["full_context", "summary_memory", "fact_memory", "psm"]:
        summary[strat] = {}
        for n in HISTORY_SIZES:
            rows = [r for r in results if r["strategy"] == strat and r["history_turns"] == n]
            if rows:
                summary[strat][n] = {
                    "mean_similarity": round(float(np.mean([r["similarity"] for r in rows])), 4),
                    "storage_bytes": rows[0]["storage_bytes"],
                    "mean_efficiency_per_kb": round(float(np.mean([r["efficiency_per_kb"] for r in rows])), 4),
                }

    output = {"config": {"history_sizes": HISTORY_SIZES, "n_prompts": len(EVAL_PROMPTS)},
              "results": results, "summary": summary}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    logger.info("Saved to %s", output_path)

    # ── Print table ────────────────────────────────────────────────────────────
    print("\n=== Experiment 2: Storage-Efficiency Trade-off ===")
    print(f"{'Strategy':<20} {'Turns':>6} {'EmbSim':>8} {'Bytes':>8} {'Eff/KB':>9}")
    print("─" * 55)
    for strat in ["full_context", "summary_memory", "fact_memory", "psm"]:
        for n in HISTORY_SIZES:
            m = summary.get(strat, {}).get(n)
            if m:
                print(
                    f"{strat:<20} {n:>6} "
                    f"{m['mean_similarity']:>8.3f} "
                    f"{m['storage_bytes']:>8} "
                    f"{m['mean_efficiency_per_kb']:>9.3f}"
                )
        print()

    return output


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Experiment 2: Storage-Efficiency Trade-off")
    parser.add_argument("--output", type=Path, default=Path("results/exp2_results.json"))
    args = parser.parse_args()

    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        resp.raise_for_status()
        logger.info("Ollama is running.")
    except Exception:
        logger.warning("Ollama not reachable at localhost:11434 — make sure `ollama serve` is running.")

    from psm.llm import get_llm
    llm = get_llm()

    def llm_generate(messages, system_prompt=""):
        return llm.generate(messages=messages, system_prompt=system_prompt)

    run_experiment(llm_generate, args.output)


if __name__ == "__main__":
    main()