from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass(slots=True)
class RouteDecision:
    mode: str
    retrieval_k: int
    use_consensus: bool
    entropy: float


class EntropyRouter:
    """
    Lightweight lexical-entropy router adapted from the prior modular Colab engine.

    Higher entropy queries get broader retrieval and optional consensus checking.
    """

    def decide(self, text: str, default_k: int = 6) -> RouteDecision:
        entropy = self.lexical_entropy(text)
        tokens = _tokenize(text)

        if len(tokens) < 6 and entropy < 1.1:
            return RouteDecision(mode="direct", retrieval_k=0, use_consensus=False, entropy=entropy)

        if entropy < 1.8:
            return RouteDecision(
                mode="focused_rag",
                retrieval_k=max(2, int(default_k) - 1),
                use_consensus=False,
                entropy=entropy,
            )

        return RouteDecision(
            mode="broad_rag",
            retrieval_k=max(4, int(default_k) + 2),
            use_consensus=True,
            entropy=entropy,
        )

    @staticmethod
    def lexical_entropy(text: str) -> float:
        toks = _tokenize(text)
        if not toks:
            return 0.0

        counts: dict[str, int] = {}
        for tok in toks:
            counts[tok] = counts.get(tok, 0) + 1

        total = len(toks)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log2(p)
        return float(entropy)


class ConsensusGuard:
    def __init__(self, min_similarity: float = 0.35) -> None:
        self.min_similarity = float(min_similarity)

    def harmonize(self, primary: str, secondary: str) -> tuple[str, float]:
        score = self.jaccard_similarity(primary, secondary)
        if score >= self.min_similarity:
            return primary, score

        merged = (
            f"Primary answer:\n{primary}\n\n"
            f"Alternate perspective (low-agreement {score:.2f}):\n{secondary}"
        )
        return merged, score

    @staticmethod
    def jaccard_similarity(a: str, b: str) -> float:
        left = set(_tokenize(a))
        right = set(_tokenize(b))
        if not left and not right:
            return 1.0
        if not left or not right:
            return 0.0
        return len(left.intersection(right)) / max(1, len(left.union(right)))


def _tokenize(text: str) -> list[str]:
    return [x.lower() for x in re.findall(r"[A-Za-z0-9_'-]+", text or "") if x.strip()]
