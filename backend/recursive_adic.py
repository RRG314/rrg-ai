from __future__ import annotations

import math
import threading


_CACHE_LOCK = threading.Lock()
_DEPTH_CACHE: dict[float, list[float]] = {}


def recursive_depth(n: int, alpha: float = 1.5) -> float:
    """
    Recursive-Adic depth from the paper's RDT recurrence:
    R(1)=0, R(n)=1+min_{1<=k<n} (R(k)+R(n-k))/alpha
    """
    if n <= 1:
        return 0.0
    if alpha <= 0:
        raise ValueError("alpha must be > 0")

    key = round(float(alpha), 8)
    with _CACHE_LOCK:
        series = _DEPTH_CACHE.setdefault(key, [0.0, 0.0])  # index 0 unused, R(1)=0
        current = len(series) - 1
        if n <= current:
            return series[n]

        for m in range(current + 1, n + 1):
            best = min((series[k] + series[m - k]) / key for k in range(1, m))
            series.append(1.0 + best)
        return series[n]


def depth_laplace_weight(
    n: int,
    beta: float = 0.35,
    alpha: float = 1.5,
    min_weight: float = 0.15,
) -> float:
    if n <= 0:
        n = 1
    d = recursive_depth(n, alpha=alpha)
    w = math.exp(-beta * float(d))
    return max(min_weight, min(1.0, w))


def recursive_adic_score(
    base_score: float,
    rank_index: int,
    beta: float = 0.35,
    alpha: float = 1.5,
) -> float:
    return float(base_score) * depth_laplace_weight(rank_index, beta=beta, alpha=alpha)
