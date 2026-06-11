"""
Big Five personality estimator using Minej/bert-base-personality.

The model's config.json does not store the id2label mapping, so the
transformers pipeline() returns LABEL_0~4 by default.  We load the model
directly with BertForSequenceClassification and inject the correct mapping
at load time, which is exactly how the model card instructs you to use it.

Reference:
    https://huggingface.co/Minej/bert-base-personality
    id2label = {"0": "Extroversion", "1": "Neuroticism",
                "2": "Agreeableness", "3": "Conscientiousness", "4": "Openness"}
"""
from __future__ import annotations

import logging
from functools import lru_cache

from psm.state import PersonalityState

logger = logging.getLogger(__name__)

# Official id2label from the Minej/bert-base-personality model card.
# Index order matches the model's 5-class output head.
_ID2LABEL: dict[int, str] = {
    0: "Extroversion",
    1: "Neuroticism",
    2: "Agreeableness",
    3: "Conscientiousness",
    4: "Openness",
}

# Canonical internal trait names (lowercase)
_LABEL2TRAIT: dict[str, str] = {
    "extroversion":      "extraversion",
    "neuroticism":       "neuroticism",
    "agreeableness":     "agreeableness",
    "conscientiousness": "conscientiousness",
    "openness":          "openness",
}


class PersonalityEstimator:
    """
    Big Five estimator backed by Minej/bert-base-personality.

    Loads BertForSequenceClassification directly and injects the correct
    id2label so outputs are always named traits, never LABEL_0~4.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._load()

    def _load(self) -> None:
        from transformers import BertForSequenceClassification, BertTokenizer  # type: ignore

        logger.info("Loading personality model: %s", self.model_name)
        self._tokenizer = BertTokenizer.from_pretrained(
            self.model_name, do_lower_case=True
        )
        self._model = BertForSequenceClassification.from_pretrained(
            self.model_name, num_labels=5
        )
        # Inject the correct mapping — the saved config.json lacks it
        self._model.config.id2label = {str(k): v for k, v in _ID2LABEL.items()}
        self._model.config.label2id = {v: k for k, v in _ID2LABEL.items()}
        self._model.eval()
        logger.info("Personality model loaded: %s", self.model_name)

    def estimate(self, text: str) -> PersonalityState:
        """
        Estimate Big Five traits from text.
        Returns PersonalityState with values in [0, 1].
        Raises RuntimeError if the model is not loaded.
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError(
                f"Personality model '{self.model_name}' is not loaded."
            )

        import torch  # type: ignore

        # Tokenize — truncate to 512 tokens
        inputs = self._tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = self._model(**inputs).logits  # shape: (1, 5)

        # Sigmoid — model was trained with BCE, not softmax
        probs = torch.sigmoid(logits).squeeze().tolist()  # list of 5 floats

        scores: dict[str, float] = {}
        for idx, prob in enumerate(probs):
            label = _ID2LABEL[idx]           # e.g. "Extroversion"
            trait = _LABEL2TRAIT[label.lower()]  # e.g. "extraversion"
            scores[trait] = float(prob)

        result = PersonalityState(
            openness=scores["openness"],
            conscientiousness=scores["conscientiousness"],
            extraversion=scores["extraversion"],
            agreeableness=scores["agreeableness"],
            neuroticism=scores["neuroticism"],
        ).clamp()

        logger.info(
            "Estimate: O=%.3f C=%.3f E=%.3f A=%.3f N=%.3f",
            result.openness, result.conscientiousness, result.extraversion,
            result.agreeableness, result.neuroticism,
        )
        return result