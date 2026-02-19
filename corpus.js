window.RRG_CORPUS = [
  {
    title: "System Blueprint",
    tags: ["architecture", "engine", "chat"],
    text: "Core modules include engine orchestration, provider plugins, retrieval, memory, routing, and interfaces. Engine coordinates planning, retrieval, tool observations, and response generation."
  },
  {
    title: "Data Indexing",
    tags: ["data", "index", "retrieval"],
    text: "Document indexing uses chunking, deduplication, source metadata, and cache signatures to speed startup and keep retrieval consistent."
  },
  {
    title: "Entropy Router",
    tags: ["router", "entropy", "reasoning"],
    text: "Entropy router estimates query complexity and chooses direct mode, focused retrieval, or broad retrieval with optional consensus behavior."
  },
  {
    title: "Entropy Retrieval",
    tags: ["rag", "retrieval", "entropy"],
    text: "Entropy-balanced retrieval avoids semantic collapse by balancing relevance and diversity. Fallback retrieval combines TF-IDF style similarity, diversity weighting, recency weighting, and result caching."
  },
  {
    title: "Consensus Guard",
    tags: ["consistency", "safety"],
    text: "Consensus guard compares alternate responses and can merge perspectives when agreement is low to reduce unstable output."
  },
  {
    title: "Autonomy Loop",
    tags: ["autonomy", "planning", "agent"],
    text: "Autonomous loop runs plan-execute-verify steps. It retrieves evidence, runs relevant tools, scores coverage, and reports step status with source references."
  },
  {
    title: "Persistent Memory",
    tags: ["memory", "facts", "session"],
    text: "Persistent memory stores conversation messages and extracted user facts like name, goals, preferences, and needs. Facts are retrieved by relevance and recency."
  },
  {
    title: "Tooling",
    tags: ["tools", "math", "search", "stats"],
    text: "Built-in capabilities include safe math evaluation, repository text search, and corpus statistics. Tools can be triggered by explicit commands or inferred from plain language."
  },
  {
    title: "Eval Harness",
    tags: ["evaluation", "metrics", "quality"],
    text: "Evaluation suite reports latency, grounding score, hallucination risk, and task success. These metrics make optimization measurable and repeatable."
  },
  {
    title: "Quantized Model Support",
    tags: ["model", "gguf", "quantized"],
    text: "Quantized local model support is designed for llama.cpp GGUF models. If unavailable or misconfigured, system falls back to deterministic mode."
  },
  {
    title: "Repo Reuse",
    tags: ["rrg314", "repos", "integration"],
    text: "Integration targets include Entorpy-RAG, topological-adam, topological-neural-net, rdt-kernel, and torchrge256."
  },
  {
    title: "Optimization Focus",
    tags: ["optimization", "performance"],
    text: "High-impact optimization priorities: profile latency bottlenecks, cache expensive operations, improve retrieval quality, and continuously benchmark with fixed eval cases."
  },
  {
    title: "GitHub Pages Constraint",
    tags: ["pages", "deployment", "html"],
    text: "GitHub Pages hosts static files. Python backends are not executed on Pages. A Pages deployment must run fully in browser JavaScript or call an external API."
  },
  {
    title: "Advanced Roadmap",
    tags: ["roadmap", "agi", "capabilities"],
    text: "Progress toward advanced AI systems should be iterative: stronger models, more reliable memory, tool execution, multimodal inputs, and rigorous eval loops."
  },
  {
    title: "Safety and Reliability",
    tags: ["safety", "reliability"],
    text: "Reliability requires explicit uncertainty handling, stable planning traces, provenance-aware retrieval, and regression tests for every added capability."
  }
];
