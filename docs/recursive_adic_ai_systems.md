# Recursive-Adic AI Systems Blueprint

## Goal
Integrate Recursive-Adic math as first-class behavior in local AI, not as a passive document reference.

## Implemented In This Repo
1. Recursive-Adic retrieval ranking in `SQLiteStore.search_chunks`:
- Depth recurrence: `R(1)=0`, `R(n)=1+min((R(k)+R(n-k))/alpha)`
- Weight: `w(n)=exp(-beta*R(n))` (clamped)
- Chunk score: `lexical_score * w(rank_index)`
2. Runtime controls:
- `AI_RECURSIVE_ADIC_RANKING=1|0`
- `AI_RADF_ALPHA` (default `1.5`)
- `AI_RADF_BETA` (default `0.35`)
3. Diagnostics:
- `/api/health` returns RADF enable/alpha/beta values
- chat context includes per-hit RADF score/depth/weight

## High-Impact Next Modules
1. RADF Memory Scheduler
- Use `R(delta_t)` to decay memory facts and prioritize “structurally persistent” facts over short bursts.
- Replace fixed-size recency windows with depth-indexed memory retention.

2. Recursive-Adic Attention Bias (Training-Time)
- Modify attention logits: `logit_ij = q_i k_j / sqrt(d) - lambda * |R(i)-R(j)|`
- Add toggle for depth-aware attention in custom transformer blocks.

3. Recursive-Adic Document Segmentation
- Build chunk boundaries by minimizing recursive split cost instead of fixed token windows.
- Expected effect: cleaner hierarchical chunk semantics for retrieval.

4. Topological Adam-R (Proposed)
- Add curvature/depth-aware scaling term to Adam moments:
  - `m_t, v_t` as usual
  - `eta_t = eta / (1 + gamma * R(layer_or_feature_index))`
- Use for stability experiments on deep hierarchical models.

## Evaluation Plan
1. Retrieval benchmarks:
- Baselines: BM25-style lexical + cosine embedding retriever
- Compare with RADF-enabled retriever on same corpus
- Metrics: MRR@k, nDCG@k, answer faithfulness

2. Conversational quality:
- Multi-turn tasks requiring persistent memory and document grounding
- Score factuality under strict mode vs non-strict mode

3. Efficiency:
- Measure query latency with and without RADF cache warmup
- Verify complexity growth and cache behavior as document count scales

## Practical Interpretation
This architecture will not create AGI by itself. It does provide:
- a novel mathematical inductive bias,
- a configurable systems implementation,
- and a reproducible path to test whether the bias improves real tasks.
