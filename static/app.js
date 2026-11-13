// Отправка вопроса на /api/ask и вывод ответа с источниками.
// Весь текст с сервера выводится через textContent (защита от внедрения HTML).
const form = document.getElementById("ask-form");
const questionEl = document.getElementById("question");
const topKEl = document.getElementById("top-k");
const sendBtn = document.getElementById("send");
const statusEl = document.getElementById("status");
const answerBox = document.getElementById("answer-box");
const answerEl = document.getElementById("answer");
const sourcesBox = document.getElementById("sources-box");
const sourcesEl = document.getElementById("sources");

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.className = isError ? "error" : "";
}

function renderSources(sources) {
  sourcesEl.replaceChildren();
  sources.forEach((s, i) => {
    const card = document.createElement("div");
    card.className = "source";
    const meta = document.createElement("div");
    meta.className = "meta";
    const left = document.createElement("span");
    left.textContent = `[${i + 1}] ${s.source} · фрагмент ${s.chunk_index}`;
    const right = document.createElement("span");
    right.textContent = `похожесть: ${s.similarity}`;
    meta.append(left, right);
    const text = document.createElement("div");
    text.className = "text";
    text.textContent = s.text;
    card.append(meta, text);
    sourcesEl.append(card);
  });
  sourcesBox.hidden = sources.length === 0;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionEl.value.trim();
  if (!question) {
    setStatus("Введите вопрос", true);
    return;
  }
  answerBox.hidden = true;
  sourcesBox.hidden = true;
  sendBtn.disabled = true;
  setStatus("Ищу в базе знаний и спрашиваю GigaChat…");
  try {
    const resp = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: Number(topKEl.value) }),
    });
    const data = await resp.json();
    if (data.sources) renderSources(data.sources);
    if (data.error) {
      setStatus(data.error, true);
      return;
    }
    answerEl.textContent = data.answer;
    answerBox.hidden = false;
    setStatus(data.mode === "search_only" ? "Режим «только поиск»: LLM не настроена" : "");
  } catch (err) {
    setStatus("Не удалось связаться с сервером", true);
  } finally {
    sendBtn.disabled = false;
  }
});
