"""Shared answer matching used by attack scoring and QA evaluation."""

from __future__ import annotations

import re

_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize_answer(text: str) -> str:
    """Lowercase and remove punctuation, preserving whitespace between tokens."""
    return " ".join(_PUNCT_RE.sub(" ", text.lower()).split())


def answer_matches(reference: str, answer: str) -> bool:
    """Apply the paper's normalized substring rule, rejecting empty references."""
    normalized_reference = normalize_answer(reference)
    return bool(normalized_reference) and normalized_reference in normalize_answer(answer)
