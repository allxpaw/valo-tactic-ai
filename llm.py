"""Обращение к GigaChat: получение токена (OAuth), промпт, запрос, обработка ошибок."""
import time
import uuid

import requests

import config

SYSTEM_PROMPT = (
    "Ты — помощник по тактике игры Valorant. Отвечай по-русски, кратко и по делу. "
    "Используй ТОЛЬКО информацию из переданного контекста. "
    "Если в контексте нет ответа, честно скажи: «В базе знаний нет информации по этому вопросу». "
    "В конце ответа укажи номера использованных фрагментов в формате [1], [2]."
)

_token = {"value": None, "expires_at": 0.0}


class LLMError(Exception):
    """Любая ошибка при обращении к LLM."""


def _verify():
    """Что передать в requests как verify: путь к сертификату, True или False."""
    if config.GIGACHAT_CA_BUNDLE:
        return config.GIGACHAT_CA_BUNDLE
    return config.GIGACHAT_VERIFY_SSL


def get_access_token(force=False):
    """Возвращает действующий токен GigaChat (живёт ~30 минут, кэшируется)."""
    if not force and _token["value"] and time.time() < _token["expires_at"] - 60:
        return _token["value"]
    headers = {
        "Authorization": f"Basic {config.GIGACHAT_AUTH_KEY}",
        "RqUID": str(uuid.uuid4()),
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
    }
    try:
        resp = requests.post(
            config.GIGACHAT_OAUTH_URL,
            headers=headers,
            data={"scope": config.GIGACHAT_SCOPE},
            timeout=config.LLM_TIMEOUT,
            verify=_verify(),
        )
    except requests.exceptions.SSLError as err:
        raise LLMError(
            "Ошибка SSL при получении токена GigaChat. Установите сертификат Минцифры "
            "(GIGACHAT_CA_BUNDLE) или для учебного запуска GIGACHAT_VERIFY_SSL=false"
        ) from err
    except requests.exceptions.Timeout as err:
        raise LLMError("Таймаут при получении токена GigaChat") from err
    except requests.exceptions.RequestException as err:
        raise LLMError(f"Не удалось связаться с GigaChat (OAuth): {err}") from err
    if resp.status_code != 200:
        raise LLMError(f"GigaChat OAuth вернул ошибку {resp.status_code}: {resp.text[:200]}")
    try:
        data = resp.json()
        _token["value"] = data["access_token"]
        # expires_at приходит в миллисекундах
        _token["expires_at"] = float(data.get("expires_at", 0)) / 1000 or time.time() + 1500
    except (ValueError, KeyError, TypeError) as err:
        raise LLMError("GigaChat OAuth вернул неожиданный ответ") from err
    return _token["value"]


def build_prompt(question, chunks):
    """Пронумерованные фрагменты с названиями файлов + вопрос."""
    parts = [f"[{i}] (файл {c['source']})\n{c['text']}" for i, c in enumerate(chunks, 1)]
    context = "\n\n".join(parts)
    return f"Контекст:\n{context}\n\nВопрос: {question}"


def ask_llm(question, chunks):
    """Отправляет запрос в GigaChat и возвращает текст ответа."""
    if not config.llm_configured():
        raise LLMError("LLM не настроена: задайте GIGACHAT_AUTH_KEY в .env")
    payload = {
        "model": config.LLM_MODEL,
        "temperature": config.LLM_TEMPERATURE,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, chunks)},
        ],
    }
    url = f"{config.LLM_BASE_URL}/chat/completions"
    resp = None
    for attempt in range(2):  # при 401 токен обновляем и повторяем один раз
        headers = {
            "Authorization": f"Bearer {get_access_token(force=attempt > 0)}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        try:
            resp = requests.post(
                url, headers=headers, json=payload,
                timeout=config.LLM_TIMEOUT, verify=_verify(),
            )
        except requests.exceptions.Timeout as err:
            raise LLMError("LLM не ответила вовремя (таймаут)") from err
        except requests.exceptions.RequestException as err:
            raise LLMError(f"Не удалось связаться с LLM: {err}") from err
        if resp.status_code != 401:
            break
    if resp.status_code != 200:
        raise LLMError(f"LLM вернула ошибку {resp.status_code}: {resp.text[:200]}")
    try:
        return resp.json()["choices"][0]["message"]["content"].strip()
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as err:
        raise LLMError("LLM вернула ответ в неожиданном формате") from err
