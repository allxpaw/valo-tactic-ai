"""Автотесты. ChromaDB, модель эмбеддингов и GigaChat заменены заглушками."""
import math
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as app_module  # noqa: E402
import config  # noqa: E402
import llm  # noqa: E402
import rag  # noqa: E402

KEYWORDS = ["эко", "кредит", "дым", "спайк", "снайпер", "ретейк"]


def fake_embed(texts):
    out = []
    for t in texts:
        t = t.lower()
        v = [float(t.count(k)) for k in KEYWORDS] + [0.01]
        n = math.sqrt(sum(x * x for x in v))
        out.append([x / n for x in v])
    return out


class FakeCollection:
    def __init__(self, items=()):
        self.items = list(items)  # (id, text, vec, meta)

    def count(self):
        return len(self.items)

    def add(self, ids, documents, embeddings, metadatas):
        self.items += list(zip(ids, documents, embeddings, metadatas))

    def query(self, query_embeddings, n_results, include):
        q = query_embeddings[0]
        scored = sorted(
            ((sum((a - b) ** 2 for a, b in zip(q, v)), t, m) for _, t, v, m in self.items),
            key=lambda x: x[0],
        )[:n_results]
        return {
            "documents": [[t for _, t, _ in scored]],
            "metadatas": [[m for _, _, m in scored]],
            "distances": [[d for d, _, _ in scored]],
        }


def make_collection():
    texts = [
        ("economy.md", 0, "Эко-раунд: сохраняйте кредиты, эко и кредиты важны."),
        ("attack.md", 0, "Закладка спайка: поставьте дым и закладывайте спайк."),
        ("weapons.md", 0, "Снайпер Operator — сильная снайперская винтовка."),
    ]
    col = FakeCollection()
    for name, idx, text in texts:
        col.add([f"{name}-{idx}"], [text], fake_embed([text]), [{"source": name, "chunk_index": idx}])
    return col


class FakeResponse:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


TOKEN_OK = FakeResponse(200, {"access_token": "tok-1", "expires_at": 9999999999999})
CHAT_OK = FakeResponse(200, {"choices": [{"message": {"content": " Ответ [1] "}}]})
CHUNKS = [{"source": "economy.md", "chunk_index": 0, "text": "Текст про эко.", "similarity": 0.9}]


class ChunkingTests(unittest.TestCase):
    def test_chunks_not_longer_than_size(self):
        text = ("Предложение номер один. " * 100).strip()
        for c in rag.split_into_chunks(text, 200, 20):
            self.assertLessEqual(len(c), 200)

    def test_overlap_between_neighbours(self):
        text = "".join(f"слово{i} " for i in range(300))
        chunks = rag.split_into_chunks(text, 200, 50)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(any(chunks[i][-10:] in chunks[i + 1] for i in range(len(chunks) - 1)))

    def test_short_text_single_chunk(self):
        self.assertEqual(rag.split_into_chunks("Коротко", 800, 100), ["Коротко"])

    def test_empty_text(self):
        self.assertEqual(rag.split_into_chunks("   ", 800, 100), [])

    def test_bad_overlap(self):
        with self.assertRaises(ValueError):
            rag.split_into_chunks("текст", 100, 100)


class LoadDocumentsTests(unittest.TestCase):
    def test_skips_other_extensions(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.md").write_text("текст", encoding="utf-8")
            (Path(d) / "b.png").write_bytes(b"\x89PNG")
            self.assertEqual([n for n, _ in rag.load_documents(Path(d))], ["a.md"])

    def test_missing_folder(self):
        with self.assertRaises(rag.RAGError):
            rag.load_documents(Path("/no/such/folder"))

    def test_empty_folder(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(rag.RAGError):
                rag.load_documents(Path(d))

    def test_real_knowledge_base(self):
        docs = rag.load_documents()
        self.assertEqual(len(docs), 9)
        for _, text in docs:
            self.assertTrue(rag.split_into_chunks(text))


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.p1 = mock.patch.object(rag, "embed", fake_embed)
        self.p2 = mock.patch.object(rag, "_get_collection", lambda create=False: self.col)
        self.col = make_collection()
        self.p1.start()
        self.p2.start()

    def tearDown(self):
        mock.patch.stopall()

    def test_best_chunk_first(self):
        res = rag.search("Когда делать эко и копить кредиты?", 3)
        self.assertEqual(res[0]["source"], "economy.md")
        self.assertGreaterEqual(res[0]["similarity"], res[1]["similarity"])

    def test_respects_top_k(self):
        self.assertEqual(len(rag.search("дым", 2)), 2)

    def test_top_k_larger_than_base(self):
        self.assertEqual(len(rag.search("дым", 10)), 3)

    def test_empty_base(self):
        self.col = FakeCollection()
        with self.assertRaises(rag.RAGError):
            rag.search("вопрос", 3)


class LLMTests(unittest.TestCase):
    def setUp(self):
        llm._token.update(value=None, expires_at=0.0)
        self.cfg = mock.patch.multiple(config, GIGACHAT_AUTH_KEY="key", LLM_BASE_URL="https://x/api/v1", LLM_MODEL="GigaChat")
        self.cfg.start()

    def tearDown(self):
        mock.patch.stopall()

    def test_build_prompt(self):
        p = llm.build_prompt("Вопрос?", CHUNKS)
        self.assertIn("[1] (файл economy.md)", p)
        self.assertIn("Вопрос: Вопрос?", p)

    def test_not_configured(self):
        with mock.patch.object(config, "GIGACHAT_AUTH_KEY", ""):
            with self.assertRaises(llm.LLMError):
                llm.ask_llm("в", CHUNKS)

    def test_request_format_and_token(self):
        with mock.patch("requests.post", side_effect=[TOKEN_OK, CHAT_OK]) as post:
            self.assertEqual(llm.ask_llm("в", CHUNKS), "Ответ [1]")
        oauth, chat = post.call_args_list
        self.assertEqual(oauth.kwargs["headers"]["Authorization"], "Basic key")
        self.assertIn("RqUID", oauth.kwargs["headers"])
        self.assertEqual(oauth.kwargs["data"], {"scope": config.GIGACHAT_SCOPE})
        self.assertEqual(chat.args[0], "https://x/api/v1/chat/completions")
        self.assertEqual(chat.kwargs["headers"]["Authorization"], "Bearer tok-1")
        body = chat.kwargs["json"]
        self.assertEqual(body["model"], "GigaChat")
        self.assertEqual(body["messages"][0]["role"], "system")

    def test_token_is_cached(self):
        with mock.patch("requests.post", side_effect=[TOKEN_OK, CHAT_OK, CHAT_OK]) as post:
            llm.ask_llm("в", CHUNKS)
            llm.ask_llm("в", CHUNKS)
        self.assertEqual(post.call_count, 3)  # один OAuth + два запроса

    def test_token_refreshed_on_401(self):
        token2 = FakeResponse(200, {"access_token": "tok-2", "expires_at": 9999999999999})
        with mock.patch("requests.post", side_effect=[TOKEN_OK, FakeResponse(401, text="expired"), token2, CHAT_OK]):
            self.assertEqual(llm.ask_llm("в", CHUNKS), "Ответ [1]")

    def test_oauth_error(self):
        with mock.patch("requests.post", return_value=FakeResponse(401, text="bad key")):
            with self.assertRaises(llm.LLMError) as ctx:
                llm.ask_llm("в", CHUNKS)
        self.assertIn("401", str(ctx.exception))

    def test_timeout_and_network(self):
        for exc in (requests.exceptions.Timeout(), requests.exceptions.ConnectionError()):
            llm._token.update(value=None, expires_at=0.0)
            with mock.patch("requests.post", side_effect=exc):
                with self.assertRaises(llm.LLMError):
                    llm.ask_llm("в", CHUNKS)

    def test_http_error(self):
        with mock.patch("requests.post", side_effect=[TOKEN_OK, FakeResponse(500, text="boom")]):
            with self.assertRaises(llm.LLMError) as ctx:
                llm.ask_llm("в", CHUNKS)
        self.assertIn("500", str(ctx.exception))

    def test_unexpected_format(self):
        with mock.patch("requests.post", side_effect=[TOKEN_OK, FakeResponse(200, {"oops": 1})]):
            with self.assertRaises(llm.LLMError):
                llm.ask_llm("в", CHUNKS)


class FlaskTests(unittest.TestCase):
    def setUp(self):
        app_module.app.testing = True
        self.client = app_module.app.test_client()
        self.col = make_collection()
        mock.patch.object(rag, "embed", fake_embed).start()
        mock.patch.object(rag, "_get_collection", lambda create=False: self.col).start()
        self.cfg = mock.patch.multiple(config, GIGACHAT_AUTH_KEY="key", LLM_BASE_URL="https://x/api/v1", LLM_MODEL="GigaChat")
        self.cfg.start()
        llm._token.update(value=None, expires_at=0.0)

    def tearDown(self):
        mock.patch.stopall()

    def ask(self, **payload):
        return self.client.post("/api/ask", json=payload)

    def test_index_opens(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("Valo", r.get_data(as_text=True))

    def test_validation_400(self):
        self.assertEqual(self.ask(question="  ", top_k=3).status_code, 400)
        self.assertEqual(self.ask(question="а" * 501, top_k=3).status_code, 400)
        self.assertEqual(self.ask(question="в", top_k=0).status_code, 400)
        self.assertEqual(self.ask(question="в", top_k=11).status_code, 400)
        self.assertEqual(self.ask(question="в", top_k="abc").status_code, 400)

    def test_search_only_mode(self):
        with mock.patch.object(config, "GIGACHAT_AUTH_KEY", ""):
            r = self.ask(question="эко", top_k=2)
        data = r.get_json()
        self.assertEqual(r.status_code, 200)
        self.assertEqual(data["mode"], "search_only")
        self.assertEqual(len(data["sources"]), 2)

    def test_full_chain(self):
        with mock.patch("requests.post", side_effect=[TOKEN_OK, CHAT_OK]):
            r = self.ask(question="Когда делать эко?", top_k=2)
        data = r.get_json()
        self.assertEqual(r.status_code, 200)
        self.assertEqual(data["mode"], "llm")
        self.assertEqual(data["answer"], "Ответ [1]")
        self.assertEqual(len(data["sources"]), 2)

    def test_llm_error_502_keeps_sources(self):
        with mock.patch("requests.post", side_effect=[TOKEN_OK, FakeResponse(500, text="x")]):
            r = self.ask(question="эко", top_k=2)
        data = r.get_json()
        self.assertEqual(r.status_code, 502)
        self.assertIn("error", data)
        self.assertEqual(len(data["sources"]), 2)

    def test_rag_error_500(self):
        self.col = FakeCollection()
        r = self.ask(question="эко", top_k=2)
        self.assertEqual(r.status_code, 500)

    def test_default_top_k(self):
        with mock.patch.object(config, "GIGACHAT_AUTH_KEY", ""):
            r = self.client.post("/api/ask", json={"question": "эко"})
        self.assertEqual(len(r.get_json()["sources"]), 3)

    def test_health(self):
        data = self.client.get("/api/health").get_json()
        self.assertTrue(data["llm_configured"])
        self.assertEqual(data["chunks_in_db"], 3)


class SecurityTests(unittest.TestCase):
    def test_no_keys_in_sources(self):
        root = Path(__file__).resolve().parent.parent
        pattern = re.compile(r"(sk-[A-Za-z0-9]{20,}|[A-Za-z0-9+/]{60,}={0,2})")
        for path in list(root.glob("*.py")) + list(root.glob("static/*")) + list(root.glob("templates/*")) + [root / ".env.example"]:
            self.assertIsNone(pattern.search(path.read_text(encoding="utf-8")), path.name)


if __name__ == "__main__":
    unittest.main()
