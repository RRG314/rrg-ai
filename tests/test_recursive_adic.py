from backend.recursive_adic import depth_laplace_weight, recursive_adic_score, recursive_depth


def test_recursive_depth_matches_known_values_alpha_15() -> None:
    alpha = 1.5
    assert recursive_depth(1, alpha=alpha) == 0.0
    assert abs(recursive_depth(2, alpha=alpha) - 1.0) < 1e-6
    assert abs(recursive_depth(3, alpha=alpha) - 1.6666667) < 1e-5
    assert abs(recursive_depth(5, alpha=alpha) - 2.4074074) < 1e-5


def test_depth_laplace_weight_decreases_over_depth() -> None:
    alpha = 1.5
    assert depth_laplace_weight(1, alpha=alpha) >= depth_laplace_weight(2, alpha=alpha)
    assert depth_laplace_weight(2, alpha=alpha) >= depth_laplace_weight(6, alpha=alpha)


def test_recursive_adic_score_scales_base_score() -> None:
    high = recursive_adic_score(base_score=5.0, rank_index=4, beta=0.35, alpha=1.5)
    low = recursive_adic_score(base_score=2.0, rank_index=4, beta=0.35, alpha=1.5)
    assert high > low
