(function () {
  const STORAGE_KEY = "rrg_ai_store_v2";

  const ui = {
    chatLog: document.getElementById("chat-log"),
    chatForm: document.getElementById("chat-form"),
    chatInput: document.getElementById("chat-input"),
    send: document.getElementById("send"),
    status: document.getElementById("status"),
    sessionList: document.getElementById("session-list"),
    newSession: document.getElementById("new-session"),
    clearMemory: document.getElementById("clear-memory"),
    autoConnect: document.getElementById("auto-connect"),
    connectionHint: document.getElementById("connection-hint"),
    quickAction: document.getElementById("quick-action"),
    applyQuickAction: document.getElementById("apply-quick-action"),
    backendUrl: document.getElementById("backend-url"),
    connectBackend: document.getElementById("connect-backend"),
    uploadFile: document.getElementById("upload-file"),
    uploadButton: document.getElementById("upload-button"),
    imageFile: document.getElementById("image-file"),
    imagePrompt: document.getElementById("image-prompt"),
    imageButton: document.getElementById("image-button"),
    ingestUrl: document.getElementById("ingest-url"),
    ingestButton: document.getElementById("ingest-button"),
    strictFacts: document.getElementById("strict-facts"),
    runAgent: document.getElementById("run-agent"),
    evidenceMode: document.getElementById("evidence-mode"),
    agentTracePanel: document.getElementById("agent-trace-panel"),
    agentPlanView: document.getElementById("agent-plan-view"),
    agentToolsView: document.getElementById("agent-tools-view"),
    agentProvenanceView: document.getElementById("agent-provenance-view"),
    agentEvidenceView: document.getElementById("agent-evidence-view"),
  };

  const runtime = {
    backendOnline: false,
    backendMode: "static-browser",
    backendBase: "",
    apiToken: "",
    authRequired: true,
    modelAvailable: false,
    model: "",
    modelReason: "",
    recursiveAdicRanking: false,
    radfAlpha: 1.5,
    radfBeta: 0.35,
    imageOCRAvailable: false,
    imageOCRReason: "",
  };

  const state = loadState();
  if (!state.currentSessionId) {
    state.currentSessionId = createSession("New chat");
    saveState();
  }

  ui.backendUrl.value = state.backendUrl || "";
  ui.strictFacts.checked = state.strictFacts !== false;
  ui.runAgent.checked = state.runAgent !== false;
  ui.evidenceMode.checked = state.evidenceMode !== false;

  renderAll();
  updateConnectionHint();
  checkBackend(false);
  window.setInterval(() => {
    checkBackend(false);
  }, 15000);

  ui.chatForm.addEventListener("submit", onSubmit);
  ui.newSession.addEventListener("click", onNewSession);
  ui.clearMemory.addEventListener("click", onClearMemory);
  ui.autoConnect.addEventListener("click", onAutoConnect);
  ui.applyQuickAction.addEventListener("click", onQuickAction);
  ui.connectBackend.addEventListener("click", onConnectBackend);
  ui.uploadButton.addEventListener("click", onUpload);
  ui.imageButton.addEventListener("click", onAnalyzeImage);
  ui.ingestButton.addEventListener("click", onIngest);
  ui.strictFacts.addEventListener("change", onStrictFactsChanged);
  ui.runAgent.addEventListener("change", onRunAgentChanged);
  ui.evidenceMode.addEventListener("change", onEvidenceModeChanged);

  async function onSubmit(event) {
    event.preventDefault();
    const text = ui.chatInput.value.trim();
    if (!text) return;

    ui.chatInput.value = "";
    appendMessage(state.currentSessionId, "user", text);

    ui.send.disabled = true;
    try {
      if (!runtime.backendOnline) {
        await checkBackend(false);
      }

      let answer;
      if (runtime.backendOnline) {
        if (state.runAgent !== false) {
          showAgentRunning(text);
          answer = await backendAgentRespond(text, state.currentSessionId);
        } else {
          answer = await backendRespond(text, state.currentSessionId);
        }
      } else {
        answer = staticRespond(text, state.currentSessionId);
      }
      appendMessage(state.currentSessionId, "ai", answer);
    } catch (err) {
      appendMessage(state.currentSessionId, "ai", `Request failed: ${String(err)}`);
    } finally {
      ui.send.disabled = false;
      ui.chatInput.focus();
      renderAll();
    }
  }

  function onNewSession() {
    const id = createSession("New chat");
    state.currentSessionId = id;
    saveState();
    renderAll();
    addBubble("ai", "Started a new chat session.");
  }

  function onClearMemory() {
    const ok = window.confirm("Clear all local sessions and memory facts?");
    if (!ok) return;

    state.sessions = {};
    state.facts = {};
    state.currentSessionId = createSession("New chat");
    saveState();
    renderAll();
    addBubble("ai", "Local browser memory cleared.");
  }

  async function onConnectBackend() {
    state.backendUrl = (ui.backendUrl.value || "").trim();
    saveState();
    await checkBackend(true);
  }

  async function onAutoConnect() {
    state.backendUrl = "";
    ui.backendUrl.value = "";
    saveState();
    await checkBackend(true);
  }

  function onQuickAction() {
    const action = (ui.quickAction.value || "").trim();
    if (!action) return;

    const templates = {
      define: "define entropy",
      search_web: "search the web for latest local llm benchmarks",
      read_site: "read website https://en.wikipedia.org/wiki/Entropy_(information_theory)",
      download_url: "download https://arxiv.org/pdf/1706.03762.pdf",
      read_file: "read file /Users/stevenreid/Documents/your-file.txt",
      find_files: "search files for optimizer in /Users/stevenreid/Documents",
      analyze_image: "analyze the text in this uploaded image",
      run_tests: "run tests in .",
      run_command: "run command python -m pytest -q in .",
    };

    ui.chatInput.value = templates[action] || "";
    ui.chatInput.focus();
  }

  function onStrictFactsChanged() {
    state.strictFacts = Boolean(ui.strictFacts.checked);
    saveState();
  }

  function onRunAgentChanged() {
    state.runAgent = Boolean(ui.runAgent.checked);
    saveState();
  }

  function onEvidenceModeChanged() {
    state.evidenceMode = Boolean(ui.evidenceMode.checked);
    saveState();
  }

  async function onUpload() {
    if (!runtime.backendOnline) {
      appendMessage(
        state.currentSessionId,
        "ai",
        "Not connected yet. Click Auto Connect first. If needed, run ./start_local_ai.sh."
      );
      return;
    }

    const file = ui.uploadFile.files && ui.uploadFile.files[0];
    if (!file) {
      appendMessage(state.currentSessionId, "ai", "Choose a file first.");
      return;
    }

    ui.uploadButton.disabled = true;
    try {
      const form = new FormData();
      form.append("file", file);

      const response = await apiFetch("/api/upload", {
        method: "POST",
        body: form,
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || `upload failed (${response.status})`);
      }

      appendMessage(
        state.currentSessionId,
        "ai",
        `Uploaded and indexed: ${payload.name}\nkind: ${payload.kind}\nchars: ${payload.char_count}\ndoc_id: ${payload.doc_id}`
      );
      ui.uploadFile.value = "";
    } catch (err) {
      appendMessage(state.currentSessionId, "ai", `Upload failed: ${String(err)}`);
    } finally {
      ui.uploadButton.disabled = false;
      renderAll();
    }
  }

  async function onIngest() {
    if (!runtime.backendOnline) {
      appendMessage(
        state.currentSessionId,
        "ai",
        "Not connected yet. Click Auto Connect first. If needed, run ./start_local_ai.sh."
      );
      return;
    }

    const url = (ui.ingestUrl.value || "").trim();
    if (!url) {
      appendMessage(state.currentSessionId, "ai", "Enter a URL first.");
      return;
    }

    ui.ingestButton.disabled = true;
    try {
      const response = await apiFetch("/api/ingest-url", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || `ingest failed (${response.status})`);
      }

      appendMessage(
        state.currentSessionId,
        "ai",
        `Ingested URL: ${payload.url}\nkind: ${payload.kind}\nchars: ${payload.char_count}\ndoc_id: ${payload.doc_id}`
      );
      ui.ingestUrl.value = "";
    } catch (err) {
      appendMessage(state.currentSessionId, "ai", `URL ingest failed: ${String(err)}`);
    } finally {
      ui.ingestButton.disabled = false;
      renderAll();
    }
  }

  async function onAnalyzeImage() {
    if (!runtime.backendOnline) {
      appendMessage(
        state.currentSessionId,
        "ai",
        "Not connected yet. Click Auto Connect first. If needed, run ./start_local_ai.sh."
      );
      return;
    }

    if (!runtime.imageOCRAvailable) {
      appendMessage(
        state.currentSessionId,
        "ai",
        `Image OCR is unavailable: ${runtime.imageOCRReason || "unknown reason"}`
      );
      return;
    }

    const file = ui.imageFile.files && ui.imageFile.files[0];
    if (!file) {
      appendMessage(state.currentSessionId, "ai", "Choose an image first.");
      return;
    }

    ui.imageButton.disabled = true;
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("prompt", (ui.imagePrompt.value || "").trim());
      form.append("session_id", state.currentSessionId);

      const response = await apiFetch("/api/image/analyze", {
        method: "POST",
        body: form,
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || `image analyze failed (${response.status})`);
      }

      const lines = [
        `Image analyzed: ${payload.filename}`,
        `format: ${payload.image_format} | size: ${payload.width}x${payload.height}`,
        `ocr chars: ${payload.ocr_char_count}`,
      ];
      if (payload.doc_id) {
        lines.push(`doc_id: ${payload.doc_id}`);
      }
      if (payload.answer) {
        lines.push("", `Model analysis:\n${payload.answer}`);
      }
      if (payload.ocr_text) {
        lines.push("", `OCR preview:\n${String(payload.ocr_text).slice(0, 1200)}`);
      }

      appendMessage(state.currentSessionId, "ai", lines.join("\n"));
      ui.imageFile.value = "";
      ui.imagePrompt.value = "";
    } catch (err) {
      appendMessage(state.currentSessionId, "ai", `Image analysis failed: ${String(err)}`);
    } finally {
      ui.imageButton.disabled = false;
      renderAll();
    }
  }

  async function checkBackend(showMessage) {
    const candidates = discoverBackendCandidates();
    let connected = false;

    for (const base of candidates) {
      if (!isLocalBackendUrl(base)) {
        continue;
      }
      try {
        const payload = await healthCheck(base);

        runtime.backendOnline = Boolean(payload.ok);
        runtime.backendMode = payload.backend || "local-python";
        runtime.backendBase = base;
        runtime.modelAvailable = Boolean(payload.model_available);
        runtime.model = payload.model || "";
        runtime.modelReason = payload.model_reason || "";
        runtime.recursiveAdicRanking = Boolean(payload.recursive_adic_ranking);
        runtime.radfAlpha = Number(payload.radf_alpha || 1.5);
        runtime.radfBeta = Number(payload.radf_beta || 0.35);
        runtime.imageOCRAvailable = Boolean(payload.image_ocr_available);
        runtime.imageOCRReason = payload.image_ocr_reason || "";
        runtime.authRequired = Boolean(payload.auth_required);

        const bootstrap = await bootstrapBackend(base);
        runtime.apiToken = bootstrap.api_token || "";
        runtime.authRequired = Boolean(bootstrap.auth_required);
        if (runtime.authRequired && !runtime.apiToken) {
          throw new Error("Backend auth token unavailable");
        }

        state.backendUrl = base;
        ui.backendUrl.value = base;
        saveState();

        if (showMessage) {
          addBubble(
            "ai",
            `Backend connected: ${runtime.backendMode}\nurl: ${base}\nmodel: ${runtime.model || "none"}\nmodel_available: ${runtime.modelAvailable}\nauth_required: ${runtime.authRequired}\nrecursive_adic_ranking: ${runtime.recursiveAdicRanking}\nradf_alpha: ${runtime.radfAlpha}\nradf_beta: ${runtime.radfBeta}\nimage_ocr_available: ${runtime.imageOCRAvailable}`
          );
        }

        connected = true;
        break;
      } catch {
        // Try next candidate.
      }
    }

    if (!connected) {
      runtime.backendOnline = false;
      runtime.backendMode = "static-browser";
      runtime.backendBase = "";
      runtime.apiToken = "";
      runtime.authRequired = true;
      runtime.modelAvailable = false;
      runtime.model = "";
      runtime.modelReason = "No backend candidate responded";
      runtime.recursiveAdicRanking = false;
      runtime.radfAlpha = 1.5;
      runtime.radfBeta = 0.35;
      runtime.imageOCRAvailable = false;
      runtime.imageOCRReason = "Backend offline";

      if (showMessage) {
        addBubble(
          "ai",
          "Still not connected to a local backend.\nRun this once in Terminal:\ncd /Users/stevenreid/Documents/New\\ project/repo_audit/rrg314/ai\n./start_local_ai.sh"
        );
      }
    }

    updateConnectionHint();
    renderStatus();
  }

  async function backendRespond(text, sessionId) {
    extractFacts(text, sessionId);

    const response = await apiFetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        session_id: sessionId,
        strict_facts: Boolean(state.strictFacts),
      }),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || `chat failed (${response.status})`);
    }

    if (payload.session_id && !state.sessions[payload.session_id]) {
      state.sessions[payload.session_id] = {
        id: payload.session_id,
        label: "Chat",
        createdAt: Date.now(),
        updatedAt: Date.now(),
        messages: [],
      };
    }

    const lines = [payload.answer || "(empty response)"];

    if (Array.isArray(payload.tool_events) && payload.tool_events.length) {
      lines.push("Tools:");
      payload.tool_events.forEach((event) => {
        lines.push(`- [${event.status}] ${event.tool}: ${event.detail}`);
      });
    }

    if (!payload.model_available && payload.model_reason) {
      lines.push(`Model status: ${payload.model_reason}`);
    }

    return lines.join("\n");
  }

  async function backendAgentRespond(text, sessionId) {
    extractFacts(text, sessionId);

    const payloadRequest = {
      message: text,
      session_id: sessionId,
      strict_fact_mode: Boolean(state.strictFacts),
      strict_facts: Boolean(state.strictFacts),
      evidence_mode: Boolean(state.evidenceMode),
      prefer_local_core: true,
      allow_web: true,
      allow_files: true,
      allow_docs: true,
      allow_code: true,
      allow_downloads: true,
      max_steps: 8,
    };

    const response = await apiFetch("/api/agent", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payloadRequest),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || `agent failed (${response.status})`);
    }

    renderAgentTrace(payload);

    if (payload.session_id && !state.sessions[payload.session_id]) {
      state.sessions[payload.session_id] = {
        id: payload.session_id,
        label: "Chat",
        createdAt: Date.now(),
        updatedAt: Date.now(),
        messages: [],
      };
    }

    const lines = [payload.answer || "(empty response)"];
    lines.push("");
    lines.push(`Agent mode: ${payload.mode || "unknown"} | task_id: ${payload.task_id || "none"}`);
    lines.push(`Local core used: ${payload.original_work_used ? "yes" : "unknown"} | llm_used: ${Boolean(payload.llm_used)}`);

    const plan = Array.isArray(payload.plan) ? payload.plan : [];
    if (plan.length) {
      lines.push(`Plan steps: ${plan.length}`);
      const doneCount = plan.filter((s) => s.status === "done").length;
      lines.push(`Completed: ${doneCount}/${plan.length}`);
    }

    const toolCalls = Array.isArray(payload.tool_calls) ? payload.tool_calls : [];
    if (toolCalls.length) {
      lines.push(`Tool calls: ${toolCalls.length}`);
    }

    const prov = Array.isArray(payload.provenance) ? payload.provenance : [];
    if (prov.length) {
      lines.push(`Provenance records: ${prov.length}`);
    }

    const evidence = Array.isArray(payload.evidence) ? payload.evidence : [];
    if (Boolean(state.evidenceMode) && evidence.length) {
      lines.push(`Evidence objects: ${evidence.length}`);
    }

    return lines.join("\n");
  }

  function showAgentRunning(text) {
    if (ui.agentTracePanel) {
      ui.agentTracePanel.open = true;
    }
    if (ui.agentPlanView) {
      ui.agentPlanView.textContent = `Running agent for:\n${text}\n\nStatus: planning/executing...`;
    }
    if (ui.agentToolsView) {
      ui.agentToolsView.textContent = "Waiting for tool trace...";
    }
    if (ui.agentProvenanceView) {
      ui.agentProvenanceView.textContent = "Waiting for provenance...";
    }
    if (ui.agentEvidenceView) {
      ui.agentEvidenceView.textContent = Boolean(state.evidenceMode)
        ? "Evidence mode active. Waiting for evidence objects..."
        : "Evidence mode off.";
    }
  }

  function renderAgentTrace(payload) {
    if (ui.agentTracePanel) {
      ui.agentTracePanel.open = true;
    }

    const plan = Array.isArray(payload.plan) ? payload.plan : [];
    if (ui.agentPlanView) {
      ui.agentPlanView.textContent = plan.length
        ? plan
            .map((step) => {
              const id = step.step_id ?? "?";
              const title = step.title || "step";
              const status = step.status || "unknown";
              const detail = step.detail ? `\n   detail: ${step.detail}` : "";
              return `${id}. [${status}] ${title}${detail}`;
            })
            .join("\n")
        : "No plan data.";
    }

    const calls = Array.isArray(payload.tool_calls) ? payload.tool_calls : [];
    if (ui.agentToolsView) {
      ui.agentToolsView.textContent = calls.length
        ? calls
            .map((call) => {
              const name = call.name || "tool";
              const status = call.status || "unknown";
              const attempt = call.attempt || 1;
              const summary = call.result_summary || "";
              return `[${status}] ${name} (attempt ${attempt})\nargs: ${JSON.stringify(call.args || {})}\nresult: ${summary}`;
            })
            .join("\n\n")
        : "No tool calls.";
    }

    const provenance = Array.isArray(payload.provenance) ? payload.provenance : [];
    if (ui.agentProvenanceView) {
      ui.agentProvenanceView.textContent = provenance.length
        ? provenance
            .map((p, idx) => {
              const source = p.source || p.path || p.url || "unknown";
              const doc = p.doc_id ? ` | doc_id=${p.doc_id}` : "";
              const snippet = p.snippet ? `\nsnippet: ${String(p.snippet).slice(0, 220)}` : "";
              return `${idx + 1}. ${p.source_type || "source"}: ${source}${doc}${snippet}`;
            })
            .join("\n\n")
        : "No provenance entries.";
    }

    const evidence = Array.isArray(payload.evidence) ? payload.evidence : [];
    if (ui.agentEvidenceView) {
      ui.agentEvidenceView.textContent = evidence.length
        ? evidence
            .map((ev, idx) => {
              const claim = ev.claim || "";
              const conf = ev.confidence ?? 0;
              const sources = Array.isArray(ev.sources) ? ev.sources.join("; ") : "";
              const snippets = Array.isArray(ev.snippets)
                ? ev.snippets.map((s) => String(s).slice(0, 120)).join(" | ")
                : "";
              return `${idx + 1}. claim: ${claim}\nconfidence: ${conf}\nsources: ${sources}\nsnippets: ${snippets}`;
            })
            .join("\n\n")
        : "No evidence objects.";
    }
  }

  function apiUrl(path) {
    const raw = (runtime.backendBase || state.backendUrl || "").trim();
    if (!raw) return path;
    const base = raw.endsWith("/") ? raw.slice(0, -1) : raw;
    return `${base}${path}`;
  }

  function discoverBackendCandidates() {
    const values = [];
    const add = (v) => {
      const cleaned = (v || "").trim().replace(/\/+$/, "");
      if (!cleaned) return;
      if (!values.includes(cleaned)) values.push(cleaned);
    };

    add(state.backendUrl || "");
    add(ui.backendUrl.value || "");

    if (window.location.protocol !== "file:") {
      add(window.location.origin);
    }

    for (let port = 8000; port <= 8020; port += 1) {
      add(`http://127.0.0.1:${port}`);
      add(`http://localhost:${port}`);
    }
    return values;
  }

  function isLocalBackendUrl(base) {
    try {
      const url = new URL(base);
      return url.hostname === "127.0.0.1" || url.hostname === "localhost";
    } catch {
      return false;
    }
  }

  async function healthCheck(base) {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 2500);
    try {
      const response = await fetch(`${base}/api/health`, {
        method: "GET",
        cache: "no-store",
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`health ${response.status}`);
      return await response.json();
    } finally {
      window.clearTimeout(timeout);
    }
  }

  async function bootstrapBackend(base) {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 2500);
    try {
      const response = await fetch(`${base}/api/bootstrap`, {
        method: "GET",
        cache: "no-store",
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`bootstrap ${response.status}`);
      return await response.json();
    } finally {
      window.clearTimeout(timeout);
    }
  }

  async function apiFetch(path, options = {}) {
    const opts = { ...options };
    const headers = new Headers(options.headers || {});
    if (runtime.apiToken) {
      headers.set("X-AI-Token", runtime.apiToken);
    }
    opts.headers = headers;
    return fetch(apiUrl(path), opts);
  }

  function updateConnectionHint() {
    if (runtime.backendOnline) {
      ui.connectionHint.textContent = `Connected to local backend ${runtime.backendBase}`;
      return;
    }
    ui.connectionHint.textContent = "Not connected to local backend. Click Auto Connect.";
  }

  function staticRespond(text, sessionId) {
    extractFacts(text, sessionId);

    const lower = text.toLowerCase();
    const directMemory = answerMemoryQuestion(lower, sessionId);
    if (directMemory) return directMemory;

    const tool = inferStaticTool(text);
    if (tool) return tool;

    const hits = retrieve(text, 4);
    const facts = relevantFacts(text, sessionId, 2);

    const lines = [
      "Running browser-only mode right now.",
      "For full tools, click Auto Connect (or run ./start_local_ai.sh).",
      "",
      "Response:",
      "I can still provide planning and memory-based help from browser state.",
    ];

    if (state.strictFacts !== false) {
      lines.push("Strict facts mode is ON, but static mode has limited source verification.");
    }

    if (hits.length) {
      lines.push("", "Grounding:");
      hits.forEach((h) => lines.push(`- ${h.title}: ${h.text.slice(0, 170)}...`));
    }

    if (facts.length) {
      lines.push("", "Memory:");
      facts.forEach((f) => lines.push(`- ${f.key}: ${f.value}`));
    }

    return lines.join("\n");
  }

  function inferStaticTool(text) {
    const low = text.toLowerCase().trim();

    const mathMatch = low.match(/(?:calculate|compute|solve|evaluate|what is)\s+([-+*/().0-9\s]{3,})$/);
    if (mathMatch) {
      const expr = mathMatch[1].trim();
      const value = safeEvalMath(expr);
      if (value !== null) return `[math_eval]\n${expr} = ${value}`;
    }

    const searchMatch = low.match(/(?:find|search|look up|where is)\s+([a-z0-9_./-]{2,80})(?:\s+(?:in|across)\s+(?:repo|repos|code|codebase))?/);
    if (searchMatch) {
      const term = searchMatch[1].trim();
      const results = searchCorpus(term, 6);
      if (results.length) {
        const rows = results.map((r) => `- ${r.title}: ${snippet(r.text, term)}`);
        return `[repo_search]\n${rows.join("\n")}`;
      }
      return `[repo_search]\nno matches for: ${term}`;
    }

    return "";
  }

  function retrieve(query, topK) {
    const q = tokenize(query);
    if (!q.length) return [];

    return corpus()
      .map((doc) => {
        const text = `${doc.title} ${doc.text} ${(doc.tags || []).join(" ")}`.toLowerCase();
        let score = 0;
        q.forEach((tok) => {
          if (text.includes(tok)) score += 1;
        });
        return { doc, score };
      })
      .filter((x) => x.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, topK)
      .map((x) => x.doc);
  }

  function searchCorpus(term, topK) {
    const t = term.toLowerCase();
    return corpus()
      .map((doc) => {
        const text = `${doc.title} ${doc.text}`.toLowerCase();
        const hit = text.includes(t);
        const score = hit ? 1 + (text.split(t).length - 1) : 0;
        return { doc, score };
      })
      .filter((x) => x.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, topK)
      .map((x) => x.doc);
  }

  function safeEvalMath(expr) {
    if (!/^[0-9+\-*/().\s]+$/.test(expr)) return null;
    try {
      const value = Function(`"use strict"; return (${expr});`)();
      if (typeof value !== "number" || !Number.isFinite(value)) return null;
      return Number(value.toFixed(10));
    } catch {
      return null;
    }
  }

  function extractFacts(text, sessionId) {
    const raw = text.trim();
    const low = raw.toLowerCase();

    const pushes = [];

    const name = low.match(/\bmy name is\s+([a-z][a-z\s'-]{0,40}?)(?:\s+and\b|[,.!?]|$)/i);
    if (name && name[1]) pushes.push({ key: "name", value: cleanFact(raw, name[1]) });

    const goal = low.match(/\bmy goal is\s+([^.!?\n]{3,200})/i);
    if (goal && goal[1]) pushes.push({ key: "goal", value: cleanFact(raw, goal[1]) });

    const need = low.match(/\bi need\s+([^.!?\n]{3,200})/i);
    if (need && need[1]) pushes.push({ key: "need", value: cleanFact(raw, need[1]) });

    const pref = low.match(/\bi (?:prefer|like)\s+([^.!?\n]{3,200})/i);
    if (pref && pref[1]) pushes.push({ key: "preference", value: cleanFact(raw, pref[1]) });

    if (!state.facts[sessionId]) state.facts[sessionId] = [];

    pushes.forEach((item) => {
      const arr = state.facts[sessionId];
      const existing = arr.find((x) => x.key === item.key);
      if (existing) {
        existing.value = item.value;
        existing.updatedAt = Date.now();
      } else {
        arr.push({ key: item.key, value: item.value, updatedAt: Date.now() });
      }
    });

    if (pushes.length) saveState();
  }

  function answerMemoryQuestion(low, sessionId) {
    const facts = relevantFacts(low, sessionId, 6);
    if (!facts.length) return "";

    if (low.includes("my name")) {
      const name = facts.find((f) => f.key === "name");
      if (name) return `From memory: name: ${name.value}`;
    }

    if (low.includes("my goal")) {
      const goal = facts.find((f) => f.key === "goal");
      if (goal) return `From memory: goal: ${goal.value}`;
    }

    if (low.includes("remember") || low.includes("what do you know about me")) {
      return `From memory:\n${facts.slice(0, 4).map((f) => `- ${f.key}: ${f.value}`).join("\n")}`;
    }

    return "";
  }

  function relevantFacts(query, sessionId, limit) {
    const tokens = tokenize(query);
    const own = state.facts[sessionId] || [];
    const global = Object.values(state.facts).flat();

    const byKey = new Map();
    [...own, ...global]
      .sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0))
      .forEach((f) => {
        if (!byKey.has(f.key)) byKey.set(f.key, f);
      });

    return [...byKey.values()]
      .map((f) => {
        const text = `${f.key} ${f.value}`.toLowerCase();
        const overlap = tokens.reduce((n, t) => n + (text.includes(t) ? 1 : 0), 0);
        return { ...f, score: overlap + ((f.updatedAt || 0) % 1000) / 10000 };
      })
      .filter((f) => f.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);
  }

  function tokenize(text) {
    return (text.toLowerCase().match(/[a-z][a-z0-9_-]+/g) || []).filter(
      (t) => t.length >= 3 && !["the", "and", "for", "with", "that", "this", "from"].includes(t)
    );
  }

  function cleanFact(original, lowerSlice) {
    const idx = original.toLowerCase().indexOf(lowerSlice.toLowerCase());
    if (idx < 0) return lowerSlice.trim();
    return original.slice(idx, idx + lowerSlice.length).trim().replace(/[\s.]+$/, "");
  }

  function createSession(label) {
    const id = crypto.randomUUID();
    state.sessions[id] = {
      id,
      label,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      messages: [],
    };
    return id;
  }

  function appendMessage(sessionId, role, content) {
    const session =
      state.sessions[sessionId] ||
      (state.sessions[sessionId] = {
        id: sessionId,
        label: "Chat",
        createdAt: Date.now(),
        updatedAt: Date.now(),
        messages: [],
      });

    session.messages.push({ role, content, at: Date.now() });
    session.updatedAt = Date.now();

    if (role === "user" && session.label === "New chat") {
      session.label = content.slice(0, 42);
    }

    saveState();
    renderChat();
    renderSessions();
  }

  function renderAll() {
    renderStatus();
    renderSessions();
    renderChat();
  }

  function renderStatus() {
    const docs = corpus().length;
    const sessions = Object.keys(state.sessions).length;
    const facts = Object.values(state.facts).reduce((n, arr) => n + arr.length, 0);

    const backend = runtime.backendOnline ? "connected" : "offline";
    const model = runtime.modelAvailable ? runtime.model : "none";
    const imageOCR = runtime.imageOCRAvailable ? "on" : "off";
    const radf = runtime.recursiveAdicRanking
      ? `on(alpha=${runtime.radfAlpha.toFixed(2)},beta=${runtime.radfBeta.toFixed(2)})`
      : "off";
    const strictFacts = state.strictFacts !== false ? "on" : "off";
    const runAgent = state.runAgent !== false ? "on" : "off";
    const evidenceMode = state.evidenceMode !== false ? "on" : "off";
    const auth = runtime.authRequired ? "token" : "off";
    const backendBase = runtime.backendBase || state.backendUrl || "none";

    ui.status.textContent =
      `backend: ${backend} | auth: ${auth} | strict facts: ${strictFacts} | evidence mode: ${evidenceMode} | run-agent: ${runAgent} | model: ${model} | image-ocr: ${imageOCR} | radf: ${radf} | url: ${backendBase} | ` +
      `docs(static): ${docs} | sessions: ${sessions} | facts: ${facts}`;
  }

  function renderSessions() {
    const items = Object.values(state.sessions)
      .sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0))
      .slice(0, 80);

    ui.sessionList.innerHTML = "";

    items.forEach((session) => {
      const btn = document.createElement("button");
      btn.className = `session-item ${session.id === state.currentSessionId ? "active" : ""}`;
      btn.textContent = `${session.label || "Chat"}`;
      btn.title = new Date(session.updatedAt || Date.now()).toLocaleString();
      btn.addEventListener("click", () => {
        state.currentSessionId = session.id;
        saveState();
        renderAll();
      });
      ui.sessionList.appendChild(btn);
    });
  }

  function renderChat() {
    ui.chatLog.innerHTML = "";
    const session = state.sessions[state.currentSessionId];
    if (!session || !session.messages.length) {
      addBubble("ai", "Ready. Chat naturally. Connect backend for full local AI tools.");
      return;
    }

    session.messages.forEach((m) => addBubble(m.role === "user" ? "user" : "ai", m.content));
  }

  function addBubble(role, text) {
    const div = document.createElement("div");
    div.className = `msg ${role}`;
    div.textContent = text;
    ui.chatLog.appendChild(div);
    ui.chatLog.scrollTop = ui.chatLog.scrollHeight;
  }

  function snippet(text, term) {
    const low = text.toLowerCase();
    const idx = low.indexOf(term.toLowerCase());
    if (idx < 0) return text.slice(0, 140) + (text.length > 140 ? "..." : "");
    const start = Math.max(0, idx - 36);
    const end = Math.min(text.length, idx + term.length + 90);
    return text.slice(start, end).trim() + (end < text.length ? "..." : "");
  }

  function corpus() {
    return Array.isArray(window.RRG_CORPUS) ? window.RRG_CORPUS : [];
  }

  function saveState() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    renderStatus();
  }

  function loadState() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) {
        return {
          sessions: {},
          facts: {},
          currentSessionId: "",
          backendUrl: "",
          strictFacts: true,
          runAgent: true,
          evidenceMode: true,
        };
      }
      const parsed = JSON.parse(raw);
      return {
        sessions: parsed.sessions || {},
        facts: parsed.facts || {},
        currentSessionId: parsed.currentSessionId || "",
        backendUrl: parsed.backendUrl || "",
        strictFacts: parsed.strictFacts !== false,
        runAgent: parsed.runAgent !== false,
        evidenceMode: parsed.evidenceMode !== false,
      };
    } catch {
      return {
        sessions: {},
        facts: {},
        currentSessionId: "",
        backendUrl: "",
        strictFacts: true,
        runAgent: true,
        evidenceMode: true,
      };
    }
  }
})();
