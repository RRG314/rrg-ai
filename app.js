(function () {
  const STORAGE_KEY = "rrg_ai_store_v1";

  const ui = {
    chatLog: document.getElementById("chat-log"),
    chatForm: document.getElementById("chat-form"),
    chatInput: document.getElementById("chat-input"),
    send: document.getElementById("send"),
    status: document.getElementById("status"),
    sessionList: document.getElementById("session-list"),
    newSession: document.getElementById("new-session"),
    clearMemory: document.getElementById("clear-memory"),
  };

  const state = loadState();
  if (!state.currentSessionId) {
    state.currentSessionId = createSession("New chat");
    saveState();
  }

  renderAll();

  ui.chatForm.addEventListener("submit", onSubmit);
  ui.newSession.addEventListener("click", onNewSession);
  ui.clearMemory.addEventListener("click", onClearMemory);

  function onSubmit(event) {
    event.preventDefault();
    const text = ui.chatInput.value.trim();
    if (!text) return;

    ui.chatInput.value = "";
    appendMessage(state.currentSessionId, "user", text);

    ui.send.disabled = true;
    setTimeout(() => {
      const answer = respond(text, state.currentSessionId);
      appendMessage(state.currentSessionId, "ai", answer);
      ui.send.disabled = false;
      ui.chatInput.focus();
      renderAll();
    }, 40);
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
    addBubble("ai", "Local memory cleared.");
  }

  function respond(text, sessionId) {
    extractFacts(text, sessionId);

    const lower = text.toLowerCase();

    const directMemory = answerMemoryQuestion(lower, sessionId);
    if (directMemory) return directMemory;

    const tool = inferTool(text);
    if (tool) return tool;

    const planMode = shouldPlan(lower);
    const hits = retrieve(text, 4);

    const lines = [];
    lines.push("RRG AI response");

    if (planMode) {
      lines.push("");
      lines.push("Plan:");
      planFor(text).forEach((step, i) => lines.push(`${i + 1}. ${step}`));
    }

    if (hits.length) {
      lines.push("");
      lines.push("Grounding:");
      hits.forEach((h) => lines.push(`- ${h.title}: ${h.text.slice(0, 170)}...`));
    }

    const facts = relevantFacts(text, sessionId, 2);
    if (facts.length) {
      lines.push("");
      lines.push("Memory:");
      facts.forEach((f) => lines.push(`- ${f.key}: ${f.value}`));
    }

    if (!planMode && !hits.length) {
      lines.push("");
      lines.push("I can help with architecture, optimization, data handling, and implementation planning.");
    }

    return lines.join("\n");
  }

  function inferTool(text) {
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

    if (
      low.includes("index stats") ||
      low.includes("memory stats") ||
      low.includes("doc count") ||
      low.includes("how many docs")
    ) {
      const sessions = Object.keys(state.sessions).length;
      const factCount = Object.values(state.facts).reduce((n, arr) => n + arr.length, 0);
      const docs = corpus().length;
      return `[stats]\ndocs=${docs}\nsessions=${sessions}\nfacts=${factCount}`;
    }

    return "";
  }

  function shouldPlan(low) {
    return (
      low.includes("step by step") ||
      low.includes("roadmap") ||
      low.includes("full plan") ||
      (low.includes("optimiz") && (low.includes("data") || low.includes("system") || low.includes("architecture")))
    );
  }

  function planFor(text) {
    const low = text.toLowerCase();
    const steps = [];

    if (low.includes("optimiz") || low.includes("latency") || low.includes("efficiency")) {
      steps.push("Profile runtime and identify top latency bottlenecks.");
      steps.push("Add caching for expensive retrieval and tool operations.");
    }

    if (low.includes("data") || low.includes("pipeline") || low.includes("processing")) {
      steps.push("Improve document chunking and metadata quality for retrieval.");
      steps.push("Add validation and regression tests for data ingestion.");
    }

    if (low.includes("model") || low.includes("agi") || low.includes("advanced")) {
      steps.push("Swap to a stronger open-weight model and re-run eval suite.");
      steps.push("Track grounding and hallucination metrics per release.");
    }

    if (!steps.length) steps.push("Clarify the target capability and success metrics.");

    return steps.slice(0, 6);
  }

  function answerMemoryQuestion(low, sessionId) {
    const facts = relevantFacts(low, sessionId, 6);
    if (!facts.length) return "";

    if (low.includes("my name") && low.includes("my goal")) {
      const name = facts.find((f) => f.key === "name");
      const goal = facts.find((f) => f.key === "goal");
      if (name || goal) {
        return `From memory:\n- name: ${name ? name.value : "unknown"}\n- goal: ${goal ? goal.value : "unknown"}`;
      }
    }

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

  function retrieve(query, topK) {
    const q = tokenize(query);
    if (!q.length) return [];

    const scored = corpus()
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

    return scored;
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

    const ranked = [...byKey.values()]
      .map((f) => {
        const text = `${f.key} ${f.value}`.toLowerCase();
        const overlap = tokens.reduce((n, t) => n + (text.includes(t) ? 1 : 0), 0);
        return { ...f, score: overlap + ((f.updatedAt || 0) % 1000) / 10000 };
      })
      .filter((f) => f.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);

    return ranked;
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
    const session = state.sessions[sessionId] || (state.sessions[sessionId] = {
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
    ui.status.textContent = `mode: static-browser | docs: ${docs} | sessions: ${sessions} | facts: ${facts}`;
  }

  function renderSessions() {
    const items = Object.values(state.sessions)
      .sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0))
      .slice(0, 50);

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
      addBubble("ai", "Ready. Chat naturally. I keep local memory in your browser.");
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
      if (!raw) return { sessions: {}, facts: {}, currentSessionId: "" };
      const parsed = JSON.parse(raw);
      return {
        sessions: parsed.sessions || {},
        facts: parsed.facts || {},
        currentSessionId: parsed.currentSessionId || "",
      };
    } catch {
      return { sessions: {}, facts: {}, currentSessionId: "" };
    }
  }
})();
