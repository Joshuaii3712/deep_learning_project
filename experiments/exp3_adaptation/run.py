"""
Experiment 3: Interaction Adaptation
======================================
Lightweight version for MX250 / Llama 3.2 1B / Ollama.

Simulates 40 turns per archetype (4 archetypes = 160 LLM calls total).
Tracks Big Five trajectory and ΔO ΔC ΔE ΔA ΔN.
Estimated runtime: ~8–12 minutes on target hardware.

Usage:
    python experiments/exp3_adaptation/run.py --output results/exp3_results.json
    python experiments/exp3_adaptation/run.py --turns 40 --plot
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
DEFAULT_TURNS = 40          # 40 turns × 4 archetypes = 160 LLM calls
SNAPSHOT_INTERVAL = 5       # record personality every 5 turns


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
            "Every challenge is a chance to grow — I love that.",
            "Let's try something bold and creative!",
            "I believe the best outcome is just around the corner.",
            "What new possibilities can we explore today?",
            "I'm feeling inspired and energetic. Let's push forward!",
            "Things are going great, I can feel it working out.",
            "This is going to be amazing, I just know it.",
        ],
    ),
    UserArchetype(
        name="skeptical",
        message_templates=[
            "I'm not sure this will work. What's the evidence?",
            "This sounds too good to be true. What are the downsides?",
            "Can you prove that claim with data?",
            "I doubt the mainstream view here. Let me challenge it.",
            "I need more proof before I can agree.",
            "This seems overly optimistic. Be realistic.",
            "I've heard this before and it didn't pan out.",
            "What if everything you said is wrong?",
        ],
    ),
    UserArchetype(
        name="analytical",
        message_templates=[
            "Let's break this down into components and analyse each one.",
            "Walk me through the logic step by step.",
            "What metrics would you use to evaluate success?",
            "What assumptions are we making here?",
            "Give me a structured framework for thinking about this.",
            "What are the root causes of this problem?",
            "Let's compare Option A and Option B systematically.",
            "I want to understand the mechanism, not just the surface.",
        ],
    ),
    UserArchetype(
        name="emotional",
        message_templates=[
            "I've been really anxious about this situation lately.",
            "I feel misunderstood and it's upsetting.",
            "This whole thing makes me feel overwhelmed.",
            "I just need some reassurance right now.",
            "I'm scared about what might happen next.",
            "Sometimes I just need someone to listen to me.",
            "Why do I always feel like things go wrong for me?",
            "I feel a deep sadness I can't shake.",
        ],
    ),
]

# ── Simulation ─────────────────────────────────────────────────────────────────

def simulate_archetype(
    archetype: UserArchetype,
    n_turns: int,
) -> dict:
    from psm import PSMAgent, PSMDatabase

    db = PSMDatabase(":memory:")
    agent = PSMAgent(
        session_id=str(uuid.uuid4()),
        db=db,
        system_prompt="You are a helpful, adaptive assistant.",
    )

    initial = agent.personality.to_dict()
    trajectory = [{"turn": 0, **initial}]

    for turn in range(1, n_turns + 1):
        msg = archetype.next_message(turn)
        try:
            agent.chat(msg)
        except Exception as exc:
            logger.warning("Turn %d failed: %s", turn, exc)
            continue

        if turn % SNAPSHOT_INTERVAL == 0 or turn == n_turns:
            p = agent.personality.to_dict()
            trajectory.append({"turn": turn, **p})
            logger.info(
                "  [%s] turn=%d  O=%.3f C=%.3f E=%.3f A=%.3f N=%.3f",
                archetype.name, turn,
                p["openness"], p["conscientiousness"], p["extraversion"],
                p["agreeableness"], p["neuroticism"],
            )

    final = agent.personality.to_dict()
    deltas = {
        f"delta_{t}": round(final[t] - initial[t], 4)
        for t in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    }

    return {
        "archetype": archetype.name,
        "n_turns": n_turns,
        "trajectory": trajectory,
        "initial": initial,
        "final": final,
        "deltas": deltas,
    }


def run_experiment(n_turns: int, output_path: Path) -> dict:
    results = {}
    for archetype in ARCHETYPES:
        logger.info("── Archetype: %s (%d turns) ──", archetype.name, n_turns)
        results[archetype.name] = simulate_archetype(archetype, n_turns)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    logger.info("Saved → %s", output_path)

    _print_summary(results)
    return results


def _print_summary(results: dict):
    traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    short  = ["ΔO", "ΔC", "ΔE", "ΔA", "ΔN"]
    n_turns = next(iter(results.values()))["n_turns"]

    print(f"\n=== Experiment 3: Interaction Adaptation ({n_turns} turns/archetype) ===")
    header = f"{'Archetype':<14} " + "  ".join(f"{s:>8}" for s in short)
    print(header)
    print("─" * len(header))
    for arch_name, data in results.items():
        row = "  ".join(f"{data['deltas'][f'delta_{t}']:>+8.4f}" for t in traits)
        print(f"{arch_name:<14}  {row}")

    # Directional hypothesis check
    print("\nDirectional hypothesis check (✓ = expected direction):")
    checks = {
        "optimistic": {"openness": ">", "neuroticism": "<"},
        "skeptical":  {"neuroticism": ">", "agreeableness": "<"},
        "analytical": {"conscientiousness": ">", "extraversion": "<"},
        "emotional":  {"agreeableness": ">", "neuroticism": ">"},
    }
    for arch, hyps in checks.items():
        if arch not in results:
            continue
        d = results[arch]["deltas"]
        for trait, direction in hyps.items():
            val = d[f"delta_{trait}"]
            ok = (direction == ">" and val > 0) or (direction == "<" and val < 0)
            mark = "✓" if ok else "✗"
            print(f"  {mark} [{arch}] Δ{trait[:1].upper()} {direction} 0  (got {val:+.4f})")


def plot_trajectories(results: dict, output_dir: Path):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
        colors = ["#7c6af5", "#4fc3f7", "#81c784", "#ffb74d", "#ef5350"]

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        axes = axes.flatten()

        for ax_idx, (arch_name, data) in enumerate(results.items()):
            ax = axes[ax_idx]
            traj = data["trajectory"]
            turns = [t["turn"] for t in traj]

            for trait, color in zip(traits, colors):
                values = [t[trait] for t in traj]
                ax.plot(turns, values, label=trait.capitalize(), color=color, linewidth=2)

            ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.set_title(f"{arch_name.capitalize()} Archetype", fontsize=11, fontweight="bold")
            ax.set_xlabel("Turn")
            ax.set_ylabel("Score")
            ax.set_ylim(0.3, 0.7)
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(alpha=0.15)

        plt.suptitle(
            f"PSM Personality Trajectories by User Archetype\n"
            f"({data['n_turns']} turns, α=0.99, Llama 3.2 1B via Ollama)",
            fontsize=12,
        )
        plt.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / "exp3_trajectories.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        logger.info("Plot saved → %s", out)
        plt.close()
    except ImportError:
        logger.warning("matplotlib not available; skipping plot.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Experiment 3: Interaction Adaptation")
    parser.add_argument("--turns", type=int, default=DEFAULT_TURNS,
                        help=f"Turns per archetype (default {DEFAULT_TURNS})")
    parser.add_argument("--output", type=Path, default=Path("results/exp3_results.json"))
    parser.add_argument("--plot", action="store_true", help="Save trajectory plot")
    parser.add_argument("--plot-dir", type=Path, default=Path("results/plots"))
    args = parser.parse_args()

    logger.info("Running Exp3 — %d turns × %d archetypes = %d LLM calls",
                args.turns, len(ARCHETYPES), args.turns * len(ARCHETYPES))

    results = run_experiment(args.turns, args.output)

    if args.plot:
        plot_trajectories(results, args.plot_dir)


if __name__ == "__main__":
    main()