"""Flask-сервер: страница сайта и API."""
from flask import Flask, jsonify, render_template, request

import config
import llm
import rag

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html", default_top_k=config.TOP_K)


@app.get("/api/health")
def health():
    return jsonify({"llm_configured": config.llm_configured(), "chunks_in_db": rag.count_chunks()})


def _bad_request(message):
    return jsonify({"error": message}), 400


@app.post("/api/ask")
def ask():
    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()
    if not question:
        return _bad_request("Введите вопрос")
    if len(question) > config.MAX_QUESTION_LEN:
        return _bad_request(f"Вопрос слишком длинный (максимум {config.MAX_QUESTION_LEN} символов)")
    try:
        top_k = int(data.get("top_k", config.TOP_K))
    except (TypeError, ValueError):
        return _bad_request("top_k должен быть целым числом")
    if not config.MIN_TOP_K <= top_k <= config.MAX_TOP_K:
        return _bad_request(f"top_k должен быть от {config.MIN_TOP_K} до {config.MAX_TOP_K}")

    try:
        sources = rag.search(question, top_k)
    except rag.RAGError as err:
        return jsonify({"error": f"Ошибка поиска: {err}"}), 500

    if not config.llm_configured():
        return jsonify({
            "mode": "search_only",
            "answer": "LLM (GigaChat) не настроена, поэтому показан только поиск по базе знаний.",
            "sources": sources,
        })
    try:
        answer = llm.ask_llm(question, sources)
    except llm.LLMError as err:
        return jsonify({"mode": "error", "error": str(err), "sources": sources}), 502
    return jsonify({"mode": "llm", "answer": answer, "sources": sources})


if __name__ == "__main__":
    try:
        rag.ensure_index()
    except rag.RAGError as err:
        print(f"Предупреждение: база не построена ({err})")
    # use_reloader=False: иначе код (и построение базы) запускался бы дважды
    app.run(host=config.HOST, port=config.PORT, debug=False, use_reloader=False)
