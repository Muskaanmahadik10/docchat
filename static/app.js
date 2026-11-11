// DocChat frontend: document list + upload, streaming chat over SSE, sources & token ledger.
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

const state = { docs: [], activeId: null, extra: new Set(), history: {}, busy: false, ctxLimit: 10000 };

// ------------------------------------------------------------------ health
async function loadHealth() {
  try {
    const h = await (await fetch("/api/health")).json();
    state.ctxLimit = h.max_context_tokens;
    $("ctx-limit").textContent = h.max_context_tokens.toLocaleString();
    $("health").classList.toggle("ok", !!h.llm.ok);
    $("health").classList.toggle("bad", !h.llm.ok);
    $("health-text").textContent = `${h.llm.model} · ${h.embeddings.provider === "none" ? "BM25" : h.embeddings.model}${h.rerank && h.rerank.enabled ? " + rerank" : ""} · ${(h.max_context_tokens / 1000).toFixed(0)}K ctx` +
      (h.llm.ok ? "" : ` — ${h.llm.detail}`);
  } catch {
    $("health").classList.add("bad");
    $("health-text").textContent = "server unreachable";
  }
}

// ------------------------------------------------------------------ documents
async function loadDocs() {
  state.docs = await (await fetch("/api/documents")).json();
  renderDocs();
  const pending = state.docs.some((d) => d.status === "processing" || d.summary_status === "running");
  clearTimeout(loadDocs.t);
  if (pending) loadDocs.t = setTimeout(loadDocs, 2500);
}

function docStatus(d) {
  if (d.status === "error") return `<span class="badge err">error</span>`;
  if (d.status === "processing") return `<span class="badge run">${esc(d.stage)}</span>`;
  const s = d.summary_status === "ready" ? `<span class="badge ok">summary ready</span>`
    : d.summary_status === "running" ? `<span class="badge run" title="${esc(d.summary_stage)}">summary: ${esc(d.summary_stage)}</span>`
    : d.summary_status === "error" ? `<span class="badge err" title="${esc(d.summary_stage)}">summary failed</span>` : "";
  return `<span class="badge ok">ready</span>${s}`;
}

function renderDocs() {
  const ul = $("doc-list");
  ul.innerHTML = state.docs.length ? "" : `<li class="muted">No documents yet.</li>`;
  for (const d of state.docs) {
    const li = document.createElement("li");
    li.className = "doc" + (d.id === state.activeId ? " active" : "");
    li.innerHTML = `<div class="doc-name">${esc(d.title || d.filename)}</div>
      <div class="doc-sub">${esc(d.filename)}${d.num_pages ? ` · ${d.num_pages} pages` : ""}${d.num_chunks ? ` · ${d.num_chunks} chunks` : ""}</div>
      <div class="doc-badges">${docStatus(d)}</div>
      ${d.error ? `<div class="doc-err">${esc(d.error)}</div>` : ""}
      <button class="del" title="Delete document">×</button>
      ${d.id !== state.activeId && d.status === "ready" && state.activeId
        ? `<button class="add ${state.extra.has(d.id) ? "on" : ""}" title="Also chat with this document">${state.extra.has(d.id) ? "✓ in chat" : "+ chat"}</button>` : ""}`;
    li.onclick = () => selectDoc(d.id);
    const add = li.querySelector(".add");
    if (add) add.onclick = (e) => {
      e.stopPropagation();
      state.extra.has(d.id) ? state.extra.delete(d.id) : state.extra.add(d.id);
      renderDocs();
    };
    li.querySelector(".del").onclick = async (e) => {
      e.stopPropagation();
      if (!li.dataset.confirm) { li.dataset.confirm = "1"; e.target.textContent = "delete?"; setTimeout(() => { delete li.dataset.confirm; e.target.textContent = "×"; }, 2500); return; }
      await fetch(`/api/documents/${d.id}`, { method: "DELETE" });
      if (state.activeId === d.id) { state.activeId = null; renderChat(); }
      loadDocs();
    };
    ul.appendChild(li);
  }
  renderHeader();
}

function activeDoc() { return state.docs.find((d) => d.id === state.activeId); }

function chatDocs() {
  const d = activeDoc();
  return d ? [d, ...state.docs.filter((x) => state.extra.has(x.id) && x.id !== d.id && x.status === "ready")] : [];
}

function selectDoc(id) {
  state.extra.delete(id);
  state.activeId = id;
  renderDocs();
  renderChat();
}

function renderHeader() {
  const d = activeDoc();
  $("doc-header").hidden = !d;
  const ready = d && d.status === "ready";
  $("input").disabled = !ready || state.busy;
  $("send").disabled = !ready || state.busy;
  $("input").placeholder = !d ? "Upload or select a document to start…" : ready ? (chatDocs().length > 1 ? `Ask across ${chatDocs().length} documents…` : `Ask about “${(d.title || d.filename).slice(0, 70)}”…`) : "Processing document…";
  $("suggestions").hidden = !ready || (state.history[state.activeId] || []).length > 0;
  if (!d) return;
  $("doc-title").textContent = chatDocs().map((x) => x.title || x.filename).join("  +  ");
  $("doc-meta").textContent = `${d.num_pages} pages · ${d.num_chunks} chunks (${d.num_tables} tables) · ~${d.total_tokens.toLocaleString()} tokens of text · retrieval: ${d.retrieval}` +
    (d.sha256 ? ` · sha256 ${d.sha256.slice(0, 12)}…` : "") +
    (d.total_tokens > state.ctxLimit ? ` · larger than the ${state.ctxLimit.toLocaleString()}-token context → retrieval + map-reduce` : "");
}

async function upload(file) {
  const fd = new FormData();
  fd.append("file", file);
  $("dropzone").classList.add("busy");
  try {
    const r = await fetch("/api/documents", { method: "POST", body: fd });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    state.activeId = data.id;
    await loadDocs();
    renderChat();
  } catch (e) {
    alertBanner(`Upload failed: ${e.message}`);
  } finally {
    $("dropzone").classList.remove("busy");
  }
}

function alertBanner(text) {
  const el = document.createElement("div");
  el.className = "banner";
  el.textContent = text;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 6000);
}

// ------------------------------------------------------------------ chat rendering
function renderChat() {
  const box = $("messages");
  box.innerHTML = "";
  const turns = state.history[state.activeId] || [];
  if (!turns.length) {
    box.innerHTML = `<div class="empty"><h1>Chat with your document</h1><p>${activeDoc()
      ? "Ask anything — or try one of the suggested questions below. Answers cite the pages they come from."
      : "Upload a PDF on the left, then ask anything. Answers are grounded in the document with page citations."}</p></div>`;
  }
  for (const t of turns) box.appendChild(t.role === "user" ? userBubble(t.content) : assistantBubble(t));
  renderHeader();
  box.scrollTop = box.scrollHeight;
}

function userBubble(text) {
  const el = document.createElement("div");
  el.className = "msg user";
  el.innerHTML = `<div class="bubble">${esc(text)}</div>`;
  return el;
}

function assistantBubble(t) {
  const el = document.createElement("div");
  el.className = "msg assistant";
  el.innerHTML = `<div class="bubble"><div class="status" hidden></div><div class="answer md"></div><div class="extras"></div></div>`;
  t.el = el;
  updateAssistant(t);
  return el;
}

function updateAssistant(t) {
  const el = t.el;
  if (!el) return;
  const status = el.querySelector(".status");
  status.hidden = !t.status;
  status.innerHTML = t.status ? `<span class="spinner"></span>${esc(t.status)}` : "";
  el.querySelector(".answer").innerHTML = renderMarkdown(t.content || "") + (t.error ? `<div class="error">⚠ ${esc(t.error)}</div>` : "");
  el.querySelector(".extras").innerHTML = extrasHtml(t);
  el.querySelectorAll("a.cite").forEach((a) => {
    a.onclick = () => openSource(el, a.dataset.src);
    const src = (t.sources || []).find((s) => s.id === a.dataset.src);
    if (src) a.title = `${src.pages} · ${src.section}`;
  });
}

function extrasHtml(t) {
  let html = "";
  if (t.plan) {
    const q = t.retrievalInfo?.queries?.length ? t.retrievalInfo.queries : t.plan.queries;
    html += `<details><summary>Route: <b>${t.plan.intent === "summary" ? "full-document summary (map-reduce)" : "retrieval Q&A"}</b>${t.retrievalInfo ? ` · ${esc(t.retrievalInfo.retrieval)}` : ""}</summary>
      <div class="kv">Planner: ${esc(t.plan.source)}</div>
      ${q && q.length ? `<div class="kv">Search queries:</div><ul class="plain">${q.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : ""}
      ${t.retrievalInfo ? `<div class="kv">Evidence: ${t.retrievalInfo.evidence_tokens.toLocaleString()} tokens (budget ${t.retrievalInfo.evidence_budget.toLocaleString()})</div>` : ""}
    </details>`;
  }
  if (t.sources && t.sources.length) {
    html += `<details class="sources"><summary>Sources (${t.sources.length})</summary>` +
      t.sources.map((s) => `<details class="src" data-id="${s.id}"><summary><b>[${s.id}]</b> ${esc(s.pages)} · ${esc(s.section)}${s.kind === "table" ? ' <span class="badge">table</span>' : ""}</summary><pre class="src-text">${esc(s.text)}</pre></details>`).join("") +
      `</details>`;
  }
  if (t.citations) {
    const bad = t.citations.invalid || [];
    html += bad.length
      ? `<div class="cite-check err">⚠ Cited ${bad.map(esc).join(", ")}, which ${bad.length > 1 ? "were" : "was"} not among the provided sources — treat those claims with caution.</div>`
      : t.citations.cited.length ? `<div class="cite-check">✓ ${t.citations.cited.length === 1 ? "The cited source exists" : `All ${t.citations.cited.length} cited sources exist`} in the evidence given to the model.</div>` : "";
  }
  if (t.ungrounded && t.ungrounded.length) {
    html += `<div class="cite-check err">⚠ Number check: ${t.ungrounded.map(esc).join(", ")} ${t.ungrounded.length > 1 ? "do" : "does"} not appear in any source shown to the model.</div>`;
  } else if (t.ungrounded) {
    html += `<div class="cite-check">✓ Number check: every number in the answer appears in the sources.</div>`;
  }
  if (t.misattributed && t.misattributed.length) {
    html += t.misattributed.map((m) => `<div class="cite-check err" title="${esc(m.context)}">⚠ Table check: ${esc(m.value)} is in the sources, but under a different row/column (${m.cells.map(esc).join("; ")}).</div>`).join("");
  } else if (t.misattributed && (t.sources || []).some((s) => s.kind === "table")) {
    html += `<div class="cite-check">✓ Table check: no number is placed under a row or column that contradicts its source table.</div>`;
  }
  if (t.instructionLike && t.instructionLike.length) {
    html += `<div class="cite-check err">⚠ Source ${t.instructionLike.map(esc).join(", ")} contains text that tries to instruct the AI. It was passed to the model as quoted document content, not as an instruction; check that the answer still addresses your question.</div>`;
  }
  const rows = [...(t.summaryLedger || []).map((e) => ({ ...e, pre: true })), ...(t.ledger || [])];
  if (rows.length) {
    const allOk = rows.every((e) => e.within_limit);
    html += `<details class="ledger"><summary>Token ledger: ${rows.length} LLM call${rows.length > 1 ? "s" : ""} ${allOk ? `<span class="badge ok">all ≤ ${state.ctxLimit.toLocaleString()}</span>` : `<span class="badge err">limit exceeded</span>`}${t.seconds ? ` · ${t.seconds}s` : ""}</summary>
      <div class="table-wrap"><table><thead><tr><th>call</th><th>prompt (est.)</th><th>prompt (actual)</th><th>output</th><th>prompt+output</th><th>limit</th><th>sec</th></tr></thead><tbody>
      ${rows.map((e) => {
        const p = e.actual_prompt_tokens ?? e.est_prompt_tokens;
        const o = e.actual_output_tokens ?? 0;
        return `<tr${e.pre ? ' class="pre"' : ""}><td>${esc(e.call)}${e.pre ? " (cached)" : ""}</td><td>${e.est_prompt_tokens}</td><td>${e.actual_prompt_tokens ?? "–"}</td><td>${e.actual_output_tokens ?? "–"}</td><td>${p + o}</td><td>${e.context_limit} ${e.within_limit ? "✓" : "✗"}</td><td>${e.seconds ?? ""}</td></tr>`;
      }).join("")}
      </tbody></table></div></details>`;
    const answerCalls = (t.ledger || []).filter((e) => /answer(-continue)?$/.test(e.call));
    const lastCall = answerCalls[answerCalls.length - 1];  // a capped answer is continued automatically
    if (lastCall && lastCall.actual_output_tokens && lastCall.actual_output_tokens >= lastCall.max_output_tokens)
      html += `<div class="cite-check err">⚠ The answer reached its ${lastCall.max_output_tokens.toLocaleString()}-token output cap and may be cut off at the end.</div>`;
    else if (answerCalls.length > 1)
      html += `<div class="cite-check">✓ The answer hit the provider's output limit and was continued in ${answerCalls.length} calls, each within the 10,000-token limit.</div>`;
  }
  return html;
}

function openSource(el, id) {
  const wrap = el.querySelector("details.sources");
  if (!wrap) return;
  wrap.open = true;
  const src = wrap.querySelector(`details.src[data-id="${id}"]`);
  if (!src) return;
  src.open = true;
  src.classList.add("flash");
  setTimeout(() => src.classList.remove("flash"), 1200);
  src.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ------------------------------------------------------------------ send
async function send(text) {
  const d = activeDoc();
  if (!d || state.busy || !text.trim()) return;
  const turns = (state.history[d.id] ||= []);
  const prior = turns.filter((t) => !t.error).map((t) => ({ role: t.role, content: t.content }));
  turns.push({ role: "user", content: text });
  const a = { role: "assistant", content: "", status: "Starting…" };
  turns.push(a);
  state.busy = true;
  renderChat();
  const box = $("messages");
  try {
    const r = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ doc_id: d.id, doc_ids: chatDocs().slice(1).map((x) => x.id), message: text, history: prior }),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      throw new Error(err.detail || r.statusText);
    }
    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const raw = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        const ev = raw.match(/^event: (.*)$/m)?.[1];
        const data = JSON.parse(raw.match(/^data: (.*)$/m)?.[1] || "{}");
        handleEvent(a, ev, data);
      }
      const nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 120;
      updateAssistant(a);
      if (nearBottom) box.scrollTop = box.scrollHeight;
    }
  } catch (e) {
    a.error = e.message;
  } finally {
    a.status = "";
    state.busy = false;
    updateAssistant(a);
    renderHeader();
    loadDocs();
  }
}

function handleEvent(a, ev, data) {
  switch (ev) {
    case "status": a.status = data.text; break;
    case "plan": a.plan = data; break;
    case "sources": a.sources = data.sources; a.retrievalInfo = data; break;
    case "token": a.status = ""; a.content += data.text; break;
    case "done": a.citations = data.citations; a.ungrounded = data.ungrounded_numbers; a.misattributed = data.misattributed_numbers; a.instructionLike = data.instruction_like_sources; a.ledger = data.ledger; a.summaryLedger = data.summary_ledger; a.seconds = data.seconds; break;
    case "error": a.error = data.text; break;
  }
}

// ------------------------------------------------------------------ chunks inspector
async function showChunks() {
  const d = activeDoc();
  if (!d) return;
  const chunks = await (await fetch(`/api/documents/${d.id}/chunks`)).json();
  $("chunks-body").innerHTML = `<p class="muted">${chunks.length} chunks · section-aware, ~350 tokens each with sentence overlap; tables are rebuilt from word positions and linearised as "row: column = value".</p>` +
    chunks.map((c) => `<div class="chunk ${c.kind}"><div class="chunk-head">#${c.id} · ${c.page_start === c.page_end ? "p. " + c.page_start : `pp. ${c.page_start}-${c.page_end}`} · ${esc(c.section)} · ${c.kind} · ${c.tokens} tokens</div><pre>${esc(c.text)}</pre></div>`).join("");
  $("chunks-dialog").showModal();
}

// ------------------------------------------------------------------ wiring
function init() {
  $("file").onchange = (e) => { if (e.target.files[0]) upload(e.target.files[0]); e.target.value = ""; };
  const dz = $("dropzone");
  dz.ondragover = (e) => { e.preventDefault(); dz.classList.add("over"); };
  dz.ondragleave = () => dz.classList.remove("over");
  dz.ondrop = (e) => { e.preventDefault(); dz.classList.remove("over"); if (e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]); };

  const input = $("input");
  input.oninput = () => { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 160) + "px"; };
  input.onkeydown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("composer").requestSubmit(); } };
  $("composer").onsubmit = (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    input.style.height = "auto";
    send(text);
  };
  document.querySelectorAll("#suggestions .chip").forEach((b) => (b.onclick = () => send(b.textContent)));
  $("btn-chunks").onclick = showChunks;
  $("btn-clear").onclick = () => { state.history[state.activeId] = []; renderChat(); };

  loadHealth();
  loadDocs().then(() => { if (!state.activeId && state.docs[0]) selectDoc(state.docs[0].id); });
}

init();
