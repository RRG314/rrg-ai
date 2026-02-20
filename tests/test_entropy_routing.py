from backend.entropy_routing import ConsensusGuard, EntropyRouter


def test_entropy_router_direct_vs_broad() -> None:
    router = EntropyRouter()

    direct = router.decide("define entropy", default_k=6)
    assert direct.mode in {"direct", "focused_rag"}

    broad = router.decide(
        "compare recursive adic retrieval with entropy-balanced modular planners and cross-domain benchmark drift",
        default_k=6,
    )
    assert broad.mode == "broad_rag"
    assert broad.retrieval_k >= 6
    assert broad.use_consensus is True


def test_consensus_guard_harmonize() -> None:
    guard = ConsensusGuard(min_similarity=0.8)
    merged, score = guard.harmonize("alpha beta", "gamma delta")
    assert score < 0.8
    assert "Alternate perspective" in merged
