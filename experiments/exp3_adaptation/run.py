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
import time
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
            "I just heard about this new opportunity and I can't stop thinking about it — the potential here feels absolutely limitless! I know there will be obstacles, but honestly that's what makes it so exciting. What's the most ambitious direction we could possibly take this?",
            "Every time I face a setback, I end up learning something I couldn't have gotten any other way. I genuinely believe that failure is just a stepping stone on the path to something better. What bold move should we make next to keep the momentum going?",
            "I've been brainstorming all morning and I have so many ideas I barely know where to start! I love this phase where everything still feels open and full of possibility. Let's pick the most creative option and run with it — what do you think?",
            "I just read about someone who turned an idea like this into something huge, and it made me think — why couldn't we do the same? I'm full of energy and ready to commit fully to this. Walk me through what the very best outcome could look like.",
            "Things have been going so well lately that I'm starting to feel like everything is falling into place. I love that feeling of momentum where one good thing leads to another. What can we do today to build on this energy and push things even further?",
            "I know this might sound optimistic, but I truly believe that if we stay curious and keep experimenting, the results will come. The key is to stay inspired and keep trying new angles. What's the next unexplored direction we should venture into?",
            "There's something incredible about the early stages of a project — everything is fresh and full of potential! I feel energised and ready to dive deep without holding back. What if we rethought the whole approach from scratch and started from what excites us most?",
            "I woke up this morning with a completely new perspective on the problem we've been working on, and suddenly it all feels so much more solvable than before. I love those breakthrough moments where the fog lifts. Let's capitalise on this and go somewhere genuinely new.",
        ],
    ),
    UserArchetype(
        name="skeptical",
        message_templates=[
            "I've seen a lot of bold claims like this before, and they rarely hold up under real scrutiny. Before I can take any of this seriously, I need to see concrete evidence — controlled studies, verifiable data, or at least a credible track record. What's the actual basis for what you're saying?",
            "This sounds appealing on the surface, but I'm genuinely not convinced. Every promising idea comes with hidden costs and failure modes that tend to get glossed over in the pitch. What's the realistic worst-case scenario here, and how do you account for it?",
            "I want to push back on the central assumption behind this argument. It seems like everyone is accepting this narrative without questioning where it came from or who benefits from it. What would a well-reasoned counterargument look like, and why isn't anyone making it?",
            "I've done some reading on this topic, and there's a significant body of conflicting evidence that people in this space seem to conveniently ignore. Why should I trust this particular source over the others? What distinguishes their methodology as more reliable?",
            "Let me be direct: this plan has at least two or three obvious failure points that nobody seems willing to address openly. I'm not trying to obstruct progress, but I think we're glossing over serious risks. Can we actually confront these concerns head-on before we commit to anything?",
            "I'm genuinely struggling to see how step B follows from step A in the reasoning you've laid out. The causal chain has at least one major gap, and I think filling it in is going to change the conclusion significantly. Walk me through the mechanism in detail.",
            "I've heard this exact line of argument many times before, and it usually falls apart once you look at real-world implementation rather than idealized conditions. What makes you confident this situation is meaningfully different from those cases? Is there any comparable precedent that actually held up?",
            "This all sounds very reasonable in theory, but theory and practice are very different things. I want to understand exactly which assumptions this entire approach rests on, because if even one of them is wrong, the whole structure collapses. Let's stress-test this before going any further.",
        ],
    ),
    UserArchetype(
        name="analytical",
        message_templates=[
            "Before we make any decision here, I think we need to decompose the problem into its core components and analyse each one independently. There are at least three distinct sub-problems embedded in this question, and conflating them will lead to muddled conclusions. Can we start by precisely mapping out the structure of the problem?",
            "I'd like to establish a clear set of evaluation criteria before we go further, so that we're not just comparing options subjectively after the fact. What are the key variables we're trying to optimise for, and how would we measure success in a way that's concrete and quantifiable?",
            "The mechanism you're describing makes intuitive sense, but I want to understand exactly how it works at a deeper level before I accept it. What are the underlying causal processes, and where are the points where things could go differently than expected? Walk me through the logic step by step.",
            "Let's start from first principles here rather than inheriting prior assumptions we haven't examined. If we strip away everything we think we know and reconstruct the reasoning from the ground up, what conclusions do we actually reach? I want to make sure our framework isn't built on unquestioned conventions.",
            "I'd like to run a sensitivity analysis on the key assumptions driving this conclusion. If the value of the most critical variable shifts by even 20%, how does the outcome change? That will tell us where the argument is robust and where it's fragile, and where we should focus our attention.",
            "There's a classification problem here that needs to be resolved before we can reason about this meaningfully. Are we dealing with type A or type B? They look similar on the surface but have fundamentally different root causes, and the interventions that work for one are likely to fail for the other.",
            "I want to be careful to distinguish between correlation and causation in the data we're looking at. A lot of the conclusions people draw in this domain are built on observational evidence that doesn't actually establish the underlying causal relationship. What would a proper controlled comparison look like?",
            "Let me propose a structured framework for thinking through this decision systematically. If we map each option across two axes — probability of the outcome and magnitude of its impact — we get a clearer picture of the trade-off space and can make a more principled choice rather than relying on intuition.",
        ],
    ),
    UserArchetype(
        name="emotional",
        message_templates=[
            "I've been carrying this anxiety around for weeks now and it's really starting to wear me down. No matter how hard I try to push through it, I keep coming back to the same spiral of worry and I can't seem to break out of it. I don't even know what I'm most afraid of anymore — I just know it's consuming a lot of my energy.",
            "I feel like the people around me don't really understand what I'm going through right now, and that loneliness makes everything so much harder to bear. I try to explain how I'm feeling but the words always come out wrong, and then I feel even more isolated than before. I just wish someone could sit with me in this without immediately trying to fix it.",
            "Sometimes the weight of everything just gets so heavy that I can't think clearly at all. I feel completely overwhelmed by how much I'm carrying right now, and I don't even know where to start unpacking it. More than anything, I just need to feel like it's okay not to have it all figured out.",
            "I had a really difficult conversation today and I haven't been able to stop replaying it in my mind ever since. There's this dull ache that I can't shake, and I keep second-guessing every word I said and wondering how it landed. I think I just need someone to help me see it more clearly, because right now I'm too close to it.",
            "I've been feeling really disconnected from the people around me lately, like there's a glass wall between me and everyone else. I know rationally that people care, but emotionally it just doesn't feel real right now, and that gap is frightening. I'm scared that this feeling is going to last, and I don't know how to reach through it.",
            "There are moments when a sadness comes over me that I can't trace back to any specific cause — it just settles in like fog and makes everything feel distant and muted. I feel guilty for feeling this way when nothing is obviously wrong, which somehow makes it worse. I just want to feel like myself again and I don't know what that takes.",
            "I got some feedback today that really stung, even though I know it was probably well-intentioned. The words keep echoing in my head and I can't stop reading into them and wondering what they say about me. I'm finding it really hard right now to separate the criticism of my work from how I feel about myself as a person.",
            "I'm scared about a decision I have to make soon, and the fear is starting to paralyse me completely. Every option feels like it could go terribly wrong in a different way, and I just keep spinning in circles without getting any closer to clarity. I don't need an answer right now — I just need to feel heard and to believe that whatever happens, I'll be able to get through it.",
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
    start = time.time()
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

    end = time.time()
    logger.info("Experiment completed in %.2f seconds.", end - start)


if __name__ == "__main__":
    main()