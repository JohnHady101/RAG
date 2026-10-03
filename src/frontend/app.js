/* Endpoints used (see src/backend/app.py):
 *   GET  /health -> { status: "ok" }
 *   POST /ask    { question: string, top_k: number }
 *            -> { answer: string, sources: [{ id, content, score }] }
 */

const API_BASE = ""; // same-origin; served via FastAPI StaticFiles. Set e.g. "http://localhost:8000" when opened as file://
const HEALTH_URL = `${API_BASE}/health`;
const ASK_URL = `${API_BASE}/ask`;

const form = document.getElementById("ask-form");
const questionEl = document.getElementById("question");
const topKEl = document.getElementById("top-k");
const askBtn = document.getElementById("ask-btn");
const clearBtn = document.getElementById("clear-btn");
const errorEl = document.getElementById("error");
const resultEl = document.getElementById("result");
const answerEl = document.getElementById("answer");
const sourcesEl = document.getElementById("sources");
const sourceCountEl = document.getElementById("source-count");
const healthEl = document.getElementById("health");
const historyEl = document.getElementById("history");

const history = [];

function showError(msg) {
  errorEl.textContent = msg;
  errorEl.hidden = !msg;
}

function setLoading(loading) {
  askBtn.disabled = loading;
  askBtn.textContent = loading ? "Asking…" : "Ask";
}

function setHealth(state, text) {
  healthEl.dataset.state = state;
  healthEl.textContent = text;
}

async function checkHealth() {
  setHealth("checking", "checking backend…");
  try {
    const res = await fetch(HEALTH_URL);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (data.status === "ok") {
      setHealth("ok", "● backend online");
    } else {
      setHealth("down", `● unexpected health: ${JSON.stringify(data)}`);
    }
  } catch (err) {
    setHealth("down", `● backend offline (${err.message}) — run: uvicorn src.backend.app:app --reload`);
  }
}

function renderSources(sources) {
  sourcesEl.innerHTML = "";
  sourceCountEl.textContent = sources.length ? `(${sources.length})` : "";
  if (!sources.length) {
    const li = document.createElement("li");
    li.textContent = "No sources returned — ingest a PDF first: python src/main.py ingest";
    sourcesEl.appendChild(li);
    return;
  }
  for (const s of sources) {
    const li = document.createElement("li");
    li.className = "source";

    const head = document.createElement("div");
    head.className = "source-head";
    const pct = typeof s.score === "number" ? `${(s.score * 100).toFixed(2)}%` : "n/a";
    head.textContent = `#${s.id} · ${pct}`;

    const body = document.createElement("div");
    body.className = "source-body";
    body.textContent = s.content;

    li.appendChild(head);
    li.appendChild(body);
    sourcesEl.appendChild(li);
  }
}

function pushHistory(question, answer) {
  history.unshift({ question, answer });
  if (history.length > 10) history.pop();
  historyEl.innerHTML = "";
  for (const h of history) {
    const li = document.createElement("li");
    const q = document.createElement("button");
    q.type = "button";
    q.className = "history-q";
    q.textContent = h.question;
    q.addEventListener("click", () => {
      questionEl.value = h.question;
      answerEl.textContent = h.answer;
      resultEl.hidden = false;
    });
    li.appendChild(q);
    historyEl.appendChild(li);
  }
}

async function ask(question, top_k) {
  const res = await fetch(ASK_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, top_k }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    // FastAPI validation errors come as { detail: [...] }
    const detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail ?? data);
    throw new Error(detail || `Request failed with HTTP ${res.status}`);
  }
  return data; // { answer, sources }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  showError("");
  const question = questionEl.value.trim();
  const top_k = Math.min(50, Math.max(1, parseInt(topKEl.value, 10) || 5));
  topKEl.value = top_k;
  if (!question) {
    showError("Please enter a question.");
    return;
  }
  setLoading(true);
  try {
    const data = await ask(question, top_k);
    answerEl.textContent = data.answer ?? "(empty answer)";
    renderSources(data.sources ?? []);
    resultEl.hidden = false;
    pushHistory(question, data.answer ?? "");
  } catch (err) {
    showError(err.message);
  } finally {
    setLoading(false);
  }
});

clearBtn.addEventListener("click", () => {
  questionEl.value = "";
  resultEl.hidden = true;
  answerEl.textContent = "";
  sourcesEl.innerHTML = "";
  showError("");
  questionEl.focus();
});

// Enter (without Shift) submits, Shift+Enter adds newline.
questionEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

checkHealth();
