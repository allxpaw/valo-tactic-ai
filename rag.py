"""RAG-модуль: чтение базы знаний, chunking, эмбеддинги, ChromaDB, поиск top_k.

Запуск `python rag.py` пересоздаёт векторную базу с нуля.
Тяжёлые библиотеки импортируются лениво — так тесты идут без интернета.
"""
import sys

import config

_model = None
_collection = None


class RAGError(Exception):
    """Ошибка RAG-модуля (пустая база, недоступна модель или ChromaDB)."""


# ---------- Чтение файлов ----------
def load_documents(folder=None):
    """Возвращает список (имя_файла, текст) для всех .md/.txt в папке."""
    folder = folder or config.KNOWLEDGE_DIR
    if not folder.exists():
        raise RAGError(f"Папка базы знаний не найдена: {folder}")
    docs = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in (".md", ".txt"):
            continue
        try:
            text = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError) as err:
            print(f"Файл пропущен ({path.name}): {err}")
            continue
        if text:
            docs.append((path.name, text))
    if not docs:
        raise RAGError("В папке knowledge_base/ нет файлов с текстом")
    return docs


# ---------- Chunking ----------
def split_into_chunks(text, chunk_size=None, overlap=None):
    """Режет текст на фрагменты до chunk_size символов с перекрытием overlap.

    Конец фрагмента ищется на границе абзаца или предложения в правой половине окна.
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap должен быть от 0 до chunk_size - 1")
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            half = start + chunk_size // 2
            cut = max(text.rfind("\n\n", half, end), -1)
            if cut == -1:
                cut = max(text.rfind(". ", half, end), text.rfind("\n", half, end))
                if cut != -1:
                    cut += 1
            if cut > start:
                end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


# ---------- Эмбеддинги ----------
def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(config.EMBEDDING_MODEL)
        except Exception as err:
            raise RAGError(f"Не удалось загрузить модель эмбеддингов: {err}") from err
    return _model


def embed(texts):
    """Список текстов -> список нормализованных векторов (списки float)."""
    vectors = _get_model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [list(map(float, v)) for v in vectors]


# ---------- ChromaDB ----------
def _get_client():
    try:
        import chromadb
        return chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    except Exception as err:
        raise RAGError(f"ChromaDB недоступна: {err}") from err


def _get_collection(create=False):
    global _collection
    if _collection is None or create:
        client = _get_client()
        if create:
            try:
                client.delete_collection(config.COLLECTION_NAME)
            except Exception:
                pass
        _collection = client.get_or_create_collection(
            name=config.COLLECTION_NAME, metadata={"hnsw:space": "l2"}
        )
    return _collection


def build_index():
    """Читает базу знаний, режет на фрагменты, считает эмбеддинги, пишет в ChromaDB."""
    docs = load_documents()
    ids, texts, metas = [], [], []
    for name, text in docs:
        for i, chunk in enumerate(split_into_chunks(text)):
            ids.append(f"{name}-{i}")
            texts.append(chunk)
            metas.append({"source": name, "chunk_index": i})
    collection = _get_collection(create=True)
    collection.add(ids=ids, documents=texts, embeddings=embed(texts), metadatas=metas)
    print(f"База построена: {len(docs)} файлов, {len(texts)} фрагментов")
    return len(texts)


def count_chunks():
    try:
        return _get_collection().count()
    except RAGError:
        return 0


def ensure_index():
    """Строит базу, только если она пуста."""
    if count_chunks() == 0:
        build_index()


# ---------- Поиск ----------
def search(question, top_k=None):
    """Возвращает top_k ближайших фрагментов: [{source, chunk_index, text, similarity}]."""
    top_k = top_k or config.TOP_K
    collection = _get_collection()
    total = collection.count()
    if total == 0:
        raise RAGError("База знаний пуста. Выполните: python rag.py")
    result = collection.query(
        query_embeddings=embed([question]),
        n_results=min(top_k, total),
        include=["documents", "metadatas", "distances"],
    )
    found = []
    for text, meta, dist in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        # Для векторов длины 1: L2² = 2 - 2·cos  =>  cos = 1 - L2²/2
        found.append({
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "text": text,
            "similarity": round(1 - dist / 2, 3),
        })
    return found


if __name__ == "__main__":
    try:
        build_index()
    except RAGError as err:
        print(f"Ошибка: {err}")
        sys.exit(1)
