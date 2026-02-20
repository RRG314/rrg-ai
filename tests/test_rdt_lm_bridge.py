from backend.rdt_lm_bridge import RDTNgramGenerator, rdt_log_depth, shell_alignment_score


def test_rdt_log_depth_monotonicity() -> None:
    d2 = rdt_log_depth(2, alpha=1.5)
    d100 = rdt_log_depth(100, alpha=1.5)
    d10000 = rdt_log_depth(10000, alpha=1.5)
    assert d2 <= d100 <= d10000


def test_rdt_generator_produces_text() -> None:
    gen = RDTNgramGenerator(alpha=1.5, seed=7)
    gen.fit([])
    out = gen.generate("the man walked", max_length=8)
    assert isinstance(out, str)
    assert len(out.split()) >= 2


def test_shell_alignment_report() -> None:
    report = shell_alignment_score(
        "recursive adic shell depth",
        ["recursive depth transforms improve shell structure"],
        alpha=1.5,
    )
    assert report["query_token_count"] > 0
    assert report["snippet_token_count"] > 0
    assert 0.0 <= float(report["shell_overlap"]) <= 1.0
    assert 0.0 <= float(report["token_overlap"]) <= 1.0
