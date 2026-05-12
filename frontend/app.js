const params = new URLSearchParams(window.location.search);
const configuredApiBase = params.get("api");
const API_BASE = (
  configuredApiBase ||
  (window.location.protocol === "file:" || window.location.port === "5500"
    ? "http://127.0.0.1:8000"
    : window.location.origin)
).replace(/\/$/, "");

const state = {
  conversationId: sessionStorage.getItem("conversationId") || null,
  busy: false,
};

const messageList = document.querySelector("#messageList");
const chatForm = document.querySelector("#chatForm");
const messageInput = document.querySelector("#messageInput");
const topKSelect = document.querySelector("#topKSelect");
const sendButton = document.querySelector("#sendButton");
const resetButton = document.querySelector("#resetButton");
const healthBadge = document.querySelector("#healthBadge");
const messageTemplate = document.querySelector("#messageTemplate");
const recommendationTemplate = document.querySelector("#recommendationTemplate");

function escapeText(value) {
  return value == null ? "" : String(value);
}

function setBusy(nextBusy) {
  state.busy = nextBusy;
  sendButton.disabled = nextBusy;
  messageInput.disabled = nextBusy;
  topKSelect.disabled = nextBusy;
}

function scrollToBottom() {
  messageList.scrollTop = messageList.scrollHeight;
}

function addMessage(role, text, recommendations = []) {
  const node = messageTemplate.content.firstElementChild.cloneNode(true);
  node.classList.add(role);
  node.querySelector(".bubble").textContent = text;

  if (recommendations.length > 0) {
    const list = document.createElement("div");
    list.className = "recommendations";
    recommendations.forEach((item) => list.appendChild(createRecommendationCard(item)));
    node.querySelector(".bubble").appendChild(list);
  }

  messageList.appendChild(node);
  scrollToBottom();
}

function createRecommendationCard(item) {
  const card = recommendationTemplate.content.firstElementChild.cloneNode(true);
  const categories = escapeText(item.categories);
  const address = escapeText(item.address);

  card.querySelector("h3").textContent = escapeText(item.shop_name);
  card.querySelector(".meta").textContent = [categories, address].filter(Boolean).join(" · ");
  card.querySelector(".score").textContent = formatScore(item.score);
  card.querySelector(".reason").textContent = escapeText(item.reason);

  const details = card.querySelector(".details");
  addDetail(details, "메뉴", item.menus);
  addDetail(details, "수상", item.awards);
  addDetail(details, "ID", item.shop_id);

  const bars = card.querySelector(".score-bars");
  addScoreBar(bars, "의미", item.semantic_score);
  addScoreBar(bars, "행동", item.behavior_score);

  return card;
}

function addDetail(parent, label, value) {
  if (!value) return;

  const term = document.createElement("dt");
  term.textContent = label;
  const desc = document.createElement("dd");
  desc.textContent = escapeText(value);
  parent.append(term, desc);
}

function addScoreBar(parent, label, value) {
  if (value == null || Number.isNaN(Number(value))) return;

  const normalized = Math.max(0, Math.min(1, Number(value)));
  const row = document.createElement("div");
  row.className = "bar-row";
  row.innerHTML = `
    <span>${label}</span>
    <span class="bar-track"><span class="bar-fill" style="width: ${normalized * 100}%"></span></span>
    <span>${normalized.toFixed(2)}</span>
  `;
  parent.appendChild(row);
}

function formatScore(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(2) : "-";
}

async function apiFetch(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  if (!response.ok) {
    let detail = `요청 실패 (${response.status})`;
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch {
      detail = response.statusText || detail;
    }
    throw new Error(detail);
  }

  if (response.status === 204) return null;
  return response.json();
}

async function checkHealth() {
  try {
    const health = await apiFetch("/health");
    if (health.recommender_loaded) {
      healthBadge.textContent = "추천 준비됨";
      healthBadge.className = "health-badge ready";
    } else {
      healthBadge.textContent = "인덱스 필요";
      healthBadge.className = "health-badge warn";
    }
  } catch {
    healthBadge.textContent = "서버 꺼짐";
    healthBadge.className = "health-badge error";
  }
}

async function ensureSession() {
  if (state.conversationId) return state.conversationId;

  const session = await apiFetch("/chat/session", { method: "POST" });
  state.conversationId = session.conversation_id;
  sessionStorage.setItem("conversationId", state.conversationId);
  return state.conversationId;
}

async function sendMessage(message) {
  setBusy(true);
  addMessage("user", message);

  try {
    const conversationId = await ensureSession();
    const payload = await apiFetch("/chat/recommend", {
      method: "POST",
      body: JSON.stringify({
        conversation_id: conversationId,
        message,
        top_k: Number(topKSelect.value),
      }),
    });

    state.conversationId = payload.conversation_id;
    sessionStorage.setItem("conversationId", state.conversationId);
    addMessage("assistant", payload.answer, payload.recommendations || []);
  } catch (error) {
    addMessage("system", error.message || "서버 요청 중 오류가 발생했습니다.");
    checkHealth();
  } finally {
    setBusy(false);
    messageInput.focus();
  }
}

async function resetConversation() {
  const previousId = state.conversationId;
  state.conversationId = null;
  sessionStorage.removeItem("conversationId");
  messageList.innerHTML = "";
  addMessage("system", "새 대화를 시작했습니다.");

  if (previousId) {
    try {
      await fetch(`${API_BASE}/chat/session/${previousId}`, { method: "DELETE" });
    } catch {
      // Resetting the local conversation is still useful even if the server is offline.
    }
  }
}

function resizeInput() {
  messageInput.style.height = "auto";
  messageInput.style.height = `${messageInput.scrollHeight}px`;
}

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  if (state.busy) return;

  const message = messageInput.value.trim();
  if (!message) return;

  messageInput.value = "";
  resizeInput();
  sendMessage(message);
});

messageInput.addEventListener("input", resizeInput);
messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    messageInput.value = button.dataset.prompt;
    resizeInput();
    messageInput.focus();
  });
});

resetButton.addEventListener("click", resetConversation);

addMessage("system", "원하는 식당 조건을 입력해 주세요.");
checkHealth();
