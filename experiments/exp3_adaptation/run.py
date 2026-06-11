"""
Experiment 3: Interaction Adaptation
======================================
Simulates 500 turns with four user archetypes and tracks Big Five
personality trajectory over time.

User archetypes:
  - Optimistic User
  - Skeptical User
  - Analytical User
  - Emotional User

Tracks ΔO  ΔC  ΔE  ΔA  ΔN per archetype.

Usage:
    python experiments/exp3_adaptation/run.py \
        --turns 500 \
        --output results/exp3_results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── User archetypes ────────────────────────────────────────────────────────────

@dataclass
class UserArchetype:
    name: str
    message_templates: list[str]

    def next_message(self, turn: int) -> str:
        return self.message_templates[turn % len(self.message_templates)]


ARCHETYPES = [
    UserArchetype(
        name="optimistic",
        message_templates=[
            "I'm really excited about this new opportunity!",
            "Things are going great, I can feel it working out.",
            "Every challenge is a chance to grow — I love that.",
            "Let's try something bold and creative!",
            "I believe the best outcome is just around the corner.",
            "What new possibilities can we explore today?",
            "I'm feeling inspired and energetic. Let's push forward!",
            "This is going to be amazing, I just know it.",
        ],
    ),
    UserArchetype(
        name="skeptical",
        message_templates=[
            "I'm not sure this will work. What's the evidence?",
            "This sounds too good to be true. What are the downsides?",
            "I've heard this before and it didn't pan out.",
            "Can you prove that claim with data?",
            "I doubt the mainstream view here. Let me challenge it.",
            "What if everything you said is wrong?",
            "I need more proof before I can agree.",
            "This seems overly optimistic. Be realistic.",
        ],
    ),
    UserArchetype(
        name="analytical",
        message_templates=[
            "Let's break this down into components and analyse each one.",
            "What are the root causes of this problem?",
            "I want to understand the mechanism, not just the surface.",
            "Walk me through the logic step by step.",
            "What metrics would you use to evaluate success?",
            "Let's compare Option A and Option B systematically.",
            "What assumptions are we making here?",
            "Give me a structured framework for thinking about this.",
        ],
    ),
    UserArchetype(
        name="emotional",
        message_templates=[
            "I've been really anxious about this situation lately.",
            "I feel misunderstood and it's upsetting.",
            "Sometimes I just need someone to listen to me.",
            "This whole thing makes me feel overwhelmed.",
            "I'm scared about what might happen next.",
            "I just need some reassurance right now.",
            "Why do I always feel like things go wrong for me?",
            "I feel a deep sadness I can't shake.",
        ],
    ),
]


# ── Simulation ─────────────────────────────────────────────────────────────────

def simulate_archetype(
    archetype: UserArchetype,
    n_turns: int,
    llm_generate,
    snapshot_interval: int = 10,
) -> dict:
    """
    Simulate n_turns of conversation with a given user archetype.
    Returns trajectory of personality states.
    """
    from psm import PSMAgent, PSMDatabase
    from psm.state import PersonalityState

    db = PSMDatabase(":memory:")
    session_id = str(uuid.uuid4())
    agent = PSMAgent(
        session_id=session_id,
        db=db,
        system_prompt="You are a helpful, adaptive assistant.",
    )

    trajectory = []
    initial = agent.personality.to_dict()
    trajectory.append({"turn": 0, **initial})

    for turn in range(1, n_turns + 1):
        msg = archetype.next_message(turn)

        try:
            agent.chat(msg)
        except Exception as exc:
            logger.warning("Turn %d failed: %s", turn, exc)
            continue

        if turn % snapshot_interval == 0:
            p = agent.personality.to_dict()
            trajectory.append({"turn": turn, **p})
            logger.info(
                "Archetype=%s turn=%d O=%.3f C=%.3f E=%.3f A=%.3f N=%.3f",
                archetype.name, turn,
                p["openness"], p["conscientiousness"], p["extraversion"],
                p["agreeableness"], p["neuroticism"],
            )

    # Compute deltas
    final = agent.personality.to_dict()
    deltas = {
        f"delta_{trait}": round(final[trait] - initial[trait], 4)
        for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    }

    return {
        "archetype": archetype.name,
        "trajectory": trajectory,
        "initial": initial,
        "final": final,
        "deltas": deltas,
    }


def run_experiment(n_turns: int, llm_generate, output_path: Path) -> dict:
    results = {}

    for archetype in ARCHETYPES:
        logger.info("Simulating archetype: %s (%d turns)…", archetype.name, n_turns)
        results[archetype.name] = simulate_archetype(
            archetype=archetype,
            n_turns=n_turns,
            llm_generate=llm_generate,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    logger.info("Saved to %s", output_path)

    # Print summary
    traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    short = ["ΔO", "ΔC", "ΔE", "ΔA", "ΔN"]
    print(f"\n=== Experiment 3: Interaction Adaptation ({n_turns} turns) ===")
    header = f"{'Archetype':<15} " + " ".join(f"{s:>7}" for s in short)
    print(header)
    print("-" * len(header))
    for arch_name, data in results.items():
        deltas_str = " ".join(
            f"{data['deltas'][f'delta_{t}']:>+7.4f}" for t in traits
        )
        print(f"{arch_name:<15} {deltas_str}")

    return results


def plot_trajectories(results: dict, output_dir: Path):
    """Visualise personality trajectories using matplotlib."""
    try:
        import matplotlib.pyplot as plt

        traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
        colors = ["#7c6af5", "#4fc3f7", "#81c784", "#ffb74d", "#ef5350"]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for ax_idx, (arch_name, data) in enumerate(results.items()):
            ax = axes[ax_idx]
            traj = data["trajectory"]
            turns = [t["turn"] for t in traj]

            for trait, color in zip(traits, colors):
                values = [t[trait] for t in traj]
                ax.plot(turns, values, label=trait.capitalize(), color=color, linewidth=1.5)

            ax.set_title(f"Archetype: {arch_name.capitalize()}", fontsize=12)
            ax.set_xlabel("Turn")
            ax.set_ylabel("Score")
            ax.set_ylim(0, 1)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.2)

        plt.suptitle("Personality Trajectories by User Archetype", fontsize=14)
        plt.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "exp3_trajectories.png", dpi=150)
        logger.info("Trajectory plot saved.")
        plt.close()
    except ImportError:
        logger.warning("matplotlib not available; skipping plot.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path("results/exp3_results.json"))
    parser.add_argument("--plot-dir", type=Path, default=Path("results/plots"))
    args = parser.parse_args()

    from psm.llm import get_llm
    llm = get_llm()

    def llm_generate(messages, system_prompt=""):
        return llm.generate(messages=messages, system_prompt=system_prompt)

    results = run_experiment(args.turns, llm_generate, args.output)
    plot_trajectories(results, args.plot_dir)


if __name__ == "__main__":
    main()
