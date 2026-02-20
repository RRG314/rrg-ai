# RRG AI v0.1.0

Release date: 2026-02-20

## Summary

v0.1.0 is the first full local-first release of RRG AI as a modular agent platform.
It provides grounded conversational workflows, tool routing, structured memory, adaptive planning updates, and reproducible benchmark/eval harnesses.

## What Is Distinct In This Release

- Recursive-Adic retrieval integration used directly in chunk ranking and grounding flow.
- Evidence Mode for claim-level grounding (`claim -> sources -> snippets -> confidence`).
- Agent trace output with plans, tool calls, provenance, and adaptive-update metadata.
- Self-improvement style heuristic updates persisted across tasks (`planning_heuristics`, `improvement_rules`).
- Benchmark isolation by default to avoid score inflation from persistent state leakage.

## What Is Standard/Expected Infrastructure

- FastAPI API surface and static frontend hosting.
- SQLite persistence model.
- Optional Ollama for local model inference.
- Local tools for web/files/docs/code execution workflows.

## Key Capabilities

- Chat and multi-step agent execution (`/api/chat`, `/api/agent`).
- Strict Fact Mode and Evidence Mode for grounded responses.
- Web search/fetch/download with local safety checks.
- Document upload/indexing and OCR-assisted image analysis.
- File read/search/list under rooted local sandbox.
- Local code run/test tooling.
- System quality gates and eval suites (`system_check`, `eval`, `industry_bench`).

## Reproducibility and Quality

- Eval/benchmark runs are isolated by default:
  - fresh run directory
  - fresh SQLite DB
  - run metadata emitted in reports (`run_id`, `db_path`, `data_dir`, `isolated`)
- Local test suite passing at release cut (`54 passed`).

## Security and Operational Defaults

- Localhost-first operation.
- Token auth enabled by default.
- Localhost-only default CORS regex.
- Pairing-gated bootstrap for non-local origins.
- File-root confinement and private/loopback URL blocking.

## Related Repositories

- https://github.com/RRG314/Recursive-Adic-Number-Field
- https://github.com/RRG314/Recursive-Division-Tree-Algorithm--Preprint
- https://github.com/RRG314/recursive-depth-integration-
- https://github.com/RRG314/rdt-collatz-orthogonality
- https://github.com/RRG314/RDT-entropy
- https://github.com/RRG314/Entorpy-RAG
- https://github.com/RRG314/topological-adam
- https://github.com/RRG314/topological-neural-net

## Related Zenodo References

- https://zenodo.org/records/17555644
- https://zenodo.org/records/18012166
- https://zenodo.org/records/17753502
- https://zenodo.org/records/17766570
- https://zenodo.org/records/17682287

## License

MIT (`LICENSE`).
