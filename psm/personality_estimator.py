"""
Big Five personality estimator.

Wraps a HuggingFace text-classification pipeline that outputs scores
for the five OCEAN traits.  Falls back to a heuristic keyword estimator
when the model is unavailable.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache

from psm.state import PersonalityState

logger = logging.getLogger(__name__)

# Trait label aliases across common HuggingFace Big Five models
_TRAIT_ALIASES: dict[str, list[str]] = {
    "openness": ["openness", "OPN", "O", "open"],
    "conscientiousness": ["conscientiousness", "CSN", "C", "conscientious"],
    "extraversion": ["extraversion", "EXT", "E", "extrovert", "extravert"],
    "agreeableness": ["agreeableness", "AGR", "A", "agreeable"],
    "neuroticism": ["neuroticism", "NEU", "N", "neurotic", "emotional stability"],
}


def _match_trait(label: str) -> str | None:
    label_lower = label.lower()
    for trait, aliases in _TRAIT_ALIASES.items():
        if any(a.lower() in label_lower for a in aliases):
            return trait
    return None


class PersonalityEstimator:
    """
    Thin wrapper around a HuggingFace pipeline for Big Five prediction.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID.  Must output labels mappable to OCEAN traits.
    use_fallback : bool
        If True, use a keyword heuristic when the model can't be loaded.
    """

    def __init__(self, model_name: str, use_fallback: bool = True):
        self.model_name = model_name
        self.use_fallback = use_fallback
        self._pipeline = None
        self._load_pipeline()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_pipeline(self) -> None:
        try:
            from transformers import pipeline  # type: ignore
            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                top_k=None,
                truncation=True,
                max_length=512,
            )
            logger.info("Big Five pipeline loaded: %s", self.model_name)
        except Exception as exc:
            logger.warning(
                "Could not load Big Five model %s: %s. "
                "Using keyword fallback.",
                self.model_name,
                exc,
            )
            self._pipeline = None

    # ── Public API ────────────────────────────────────────────────────────────

    def estimate(self, text: str) -> PersonalityState:
        """
        Estimate Big Five traits from *text* (typically a conversation window).
        Returns a PersonalityState with values in [0, 1].
        """
        if self._pipeline is not None:
            return self._estimate_hf(text)
        if self.use_fallback:
            return self._estimate_keyword(text)
        raise RuntimeError("No personality estimator available.")

    # ── HuggingFace path ──────────────────────────────────────────────────────

    def _estimate_hf(self, text: str) -> PersonalityState:
        # Truncate to avoid token overflow
        text_chunk = text[-2000:]
        try:
            results = self._pipeline(text_chunk)
            # pipeline returns list[list[dict]] when top_k=None
            if results and isinstance(results[0], list):
                results = results[0]
            scores: dict[str, float] = {}
            for item in results:
                label = item.get("label", "")
                score = float(item.get("score", 0.5))
                trait = _match_trait(label)
                if trait:
                    scores[trait] = score

            # Fill any missing traits with neutral
            return PersonalityState(
                openness=scores.get("openness", 0.5),
                conscientiousness=scores.get("conscientiousness", 0.5),
                extraversion=scores.get("extraversion", 0.5),
                agreeableness=scores.get("agreeableness", 0.5),
                neuroticism=scores.get("neuroticism", 0.5),
            ).clamp()
        except Exception as exc:
            logger.error("HF estimation failed: %s. Falling back.", exc)
            return self._estimate_keyword(text)

    # ── Keyword fallback ──────────────────────────────────────────────────────

    def _estimate_keyword(self, text: str) -> PersonalityState:
        """
        Lightweight keyword heuristic.  Counts trait-associated words and
        normalises to [0, 1].  Suitable as a graceful fallback only.
        """
        text_lower = text.lower()
        word_count = max(1, len(text_lower.split()))

        def density(keywords: list[str]) -> float:
            count = sum(text_lower.count(kw) for kw in keywords)
            return min(1.0, (count / word_count) * 50)  # scale factor

        o = density([
            "curious", "explore", "creative", "imagine", "novel",
            "art", "ideas", "philosophy", "wonder", "discover",
        ])
        c = density([
            "plan", "organise", "organize", "schedule", "efficient",
            "deadline", "careful", "detail", "goal", "disciplined",
        ])
        e = density([
            "social", "friend", "party", "talk", "people",
            "outgoing", "excited", "laugh", "fun", "energetic",
        ])
        a = density([
            "help", "kind", "support", "understand", "care",
            "cooperate", "agree", "trust", "warm", "empathy",
        ])
        n = density([
            "worry", "anxious", "stress", "fear", "nervous",
            "upset", "sad", "overwhelm", "panic", "risk",
        ])

        # Shift each density to be centred around 0.5
        def shift(d: float) -> float:
            return max(0.0, min(1.0, 0.5 + (d - 0.1) * 2))

        return PersonalityState(
            openness=shift(o),
            conscientiousness=shift(c),
            extraversion=shift(e),
            agreeableness=shift(a),
            neuroticism=shift(n),
        )
