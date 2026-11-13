# Valo Tactic AI

Веб-приложение с RAG-системой: помощник по тактике Valorant. Отвечает на русском языке по собственной базе знаний (`knowledge_base/`) с помощью **GigaChat** и показывает фрагменты-источники.

## Как это работает
Вопрос → эмбеддинг (`paraphrase-multilingual-MiniLM-L12-v2`) → поиск top_k фрагментов в ChromaDB → промпт (инструкция + контекст + вопрос) → GigaChat → ответ и источники.

## Быстрый старт
```bash
git clone <АДРЕС_РЕПОЗИТОРИЯ>
cd valo-tactic-ai
python -m venv .venv
.venv\Scripts\activate            # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env            # macOS/Linux: cp .env.example .env
python app.py                     # откройте http://127.0.0.1:5000
```
Первый запуск скачивает модель эмбеддингов и строит базу `chroma_db/`.

## Подключение GigaChat
1. Зарегистрируйтесь на https://developers.sber.ru/gigachat, создайте проект **GigaChat API** и получите **Authorization key**.
2. Впишите его в `.env`: `GIGACHAT_AUTH_KEY=...` (scope по умолчанию `GIGACHAT_API_PERS`).
3. Сертификаты Минцифры: скачайте корневой сертификат (https://gu-st.ru/content/lending/russian_trusted_root_ca_pem.crt) и укажите путь в `GIGACHAT_CA_BUNDLE`. Для учебного запуска можно поставить `GIGACHAT_VERIFY_SSL=false`.

Приложение само получает access-токен (живёт ~30 минут), кэширует его и обновляет при ответе 401.
Без ключа работает режим «только поиск».

## Пополнение базы знаний
Добавьте `.md`/`.txt` файлы в `knowledge_base/` и выполните `python rag.py` (база пересоздаётся без дублей).

## API
- `POST /api/ask` — `{question, top_k}` → `{mode, answer, sources}`; ошибки: 400 (данные), 500 (RAG), 502 (LLM, источники сохраняются)
- `GET /api/health` — `{llm_configured, chunks_in_db}`

## Тесты
```bash
python -m unittest discover -s tests -v
```
Тесты идут без интернета: ChromaDB, эмбеддинги и GigaChat заменены заглушками. Качество реального поиска проверяется вручную.

## Ограничения
База знаний учебная: цены и список агентов зависят от патча. Ответ LLM не гарантирует истинность, поэтому показываются источники.
