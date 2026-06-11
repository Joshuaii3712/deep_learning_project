"""
Case Studies
=============
Analyzes personality evolution for three user archetypes:
  1. Entrepreneurial User
  2. Highly Risk-Averse User
  3. Support-Seeking User

Analyzes:
  - personality evolution (trajectory)
  - prompt profiles (rendered personality profiles per snapshot)
  - generation changes (response style shift over time)

Usage:
    python case_studies/run.py --turns 100 --output results/case_studies.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── User scripts ───────────────────────────────────────────────────────────────

CASE_SCRIPTS: dict[str, list[str]] = {
    "entrepreneurial": [
        "I have a wild idea to disrupt the logistics industry with drones.",
        "I know the risk is high but the payoff is massive. Should I go all in?",
        "I want to hire fast and build an aggressive roadmap.",
        "Competition doesn't scare me — it excites me.",
        "What's the fastest path to product-market fit?",
        "I'm thinking about pitching investors next month.",
        "I need bold, unconventional growth strategies.",
        "How do I keep momentum going after setbacks?",
        "I want to pivot quickly if the first approach fails.",
        "What mindset separates successful founders from the rest?",
    ],
    "risk_averse": [
        "I'm terrified of losing my savings. Is investing safe?",
        "What if the market crashes right after I invest?",
        "I'd rather keep money in a bank than risk it.",
        "Even a small loss would devastate me emotionally.",
        "Can you list every possible way this could go wrong?",
        "I've never taken a financial risk and I'm proud of it.",
        "Security and certainty are my top priorities.",
        "I need a guaranteed return, not a probable one.",
        "I don't trust optimistic projections. Show me worst-case.",
        "What's the safest thing I could possibly do with $10,000?",
    ],
    "support_seeking": [
        "I've been feeling really alone lately and don't know who to talk to.",
        "My friends don't understand what I'm going through.",
        "Sometimes I cry without even knowing why.",
        "I just need someone to listen without judging me.",
        "I feel like a burden to everyone around me.",
        "Can you just be here with me for a while?",
        "I struggle to find meaning in what I do.",
        "I'm really sensitive and I take things personally.",
        "Do you think I'm being too emotional?",
        "I just want someone to tell me it's going to be okay.",
    ],
}


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_case_study(
    case_name: str,
    script: list[str],
    llm_generate: Callable,
    n_turns: int,
    snapshot_every: int = 5,
) -> dict:
    from psm import PSMAgent, PSMDatabase
    from psm.renderer import render_personality_verbose

    db = PSMDatabase(":memory:")
    session_id = str(uuid.uuid4())
    agent = PSMAgent(
        session_id=session_id,
        db=db,
        system_prompt="You are a supportive and thoughtful assistant.",
    )

    trajectory = []
    profiles = []
    responses_sample = []

    for turn in range(n_turns):
        msg = script[turn % len(script)]

        try:
            response = agent.chat(msg)
        except Exception as exc:
            logger.warning("Turn %d error: %s", turn, exc)
            continue

        if turn % snapshot_every == 0 or turn == n_turns - 1:
            p = agent.personality
            verbose = render_personality_verbose(p)
            snapshot = {
                "turn": turn,
                "personality": p.to_dict(),
                "profile": agent.personality_profile,
                "verbose": verbose,
            }
            trajectory.append({"turn": turn, **p.to_dict()})
            profiles.append(snapshot)

            if turn % 20 == 0:
                responses_sample.append({
                    "turn": turn,
                    "user_message": msg,
                    "response_excerpt": response[:300],
                    "profile": agent.personality_profile,
                })
                logger.info(
                    "Case=%s turn=%d | O=%.3f C=%.3f E=%.3f A=%.3f N=%.3f",
                    case_name, turn,
                    p.openness, p.conscientiousness, p.extraversion,
                    p.agreeableness, p.neuroticism,
                )

    initial = trajectory[0] if trajectory else {}
    final = trajectory[-1] if trajectory else {}
    deltas = {
        f"delta_{trait}": round(final.get(trait, 0.5) - initial.get(trait, 0.5), 4)
        for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    }

    return {
        "case_name": case_name,
        "n_turns": n_turns,
        "trajectory": trajectory,
        "profiles": profiles,
        "responses_sample": responses_sample,
        "initial_personality": initial,
        "final_personality": final,
        "deltas": deltas,
    }


def run_all(n_turns: int, llm_generate: Callable, output_path: Path) -> dict:
    results = {}
    for case_name, script in CASE_SCRIPTS.items():
        logger.info("Running case study: %s (%d turns)…", case_name, n_turns)
        results[case_name] = run_case_study(case_name, script, llm_generate, n_turns)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    logger.info("Saved to %s", output_path)

    # Print summary
    traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    print("\n=== Case Studies: Personality Evolution ===")
    for case_name, data in results.items():
        print(f"\n[{case_name.upper()}]")
        d = data["deltas"]
        for t in traits:
            delta = d.get(f"delta_{t}", 0)
            bar = "▲" if delta > 0.001 else ("▼" if delta < -0.001 else "─")
            print(f"  {t:<20} {bar} {delta:+.4f}")
        # Last active profile
        if data["profiles"]:
            last_profile = data["profiles"][-1]["profile"]
            if last_profile:
                print(f"\n  Final Profile:\n  {last_profile.replace(chr(10), chr(10)+'  ')}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("results/case_studies.json"))
    args = parser.parse_args()

    from psm.llm import get_llm
    llm = get_llm()

    def llm_generate(messages, system_prompt=""):
        return llm.generate(messages=messages, system_prompt=system_prompt)

    run_all(args.turns, llm_generate, args.output)


if __name__ == "__main__":
    main()
