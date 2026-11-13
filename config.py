"""Настройки проекта. Секреты читаются из файла .env (он не коммитится)."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# --- Пути ---
KNOWLEDGE_DIR = BASE_DIR / "knowledge_base"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "valorant_tactics"

# --- RAG ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "3"))

# --- LLM: GigaChat (Сбер) ---
# Authorization key — строка из личного кабинета developers.sber.ru (уже в base64).
GIGACHAT_AUTH_KEY = os.getenv("GIGACHAT_AUTH_KEY", "").strip()
GIGACHAT_SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
GIGACHAT_OAUTH_URL = os.getenv(
    "GIGACHAT_OAUTH_URL", "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://gigachat.devices.sberbank.ru/api/v1").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "GigaChat")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))

# Сертификаты Минцифры не входят в стандартный набор. Варианты:
#   GIGACHAT_CA_BUNDLE=путь к файлу russian_trusted_root_ca.cer  (рекомендуется)
#   GIGACHAT_VERIFY_SSL=false                                    (только для учебного запуска)
GIGACHAT_CA_BUNDLE = os.getenv("GIGACHAT_CA_BUNDLE", "").strip()
GIGACHAT_VERIFY_SSL = os.getenv("GIGACHAT_VERIFY_SSL", "true").strip().lower() not in ("0", "false", "no")

# --- Flask ---
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "5000"))
MAX_QUESTION_LEN = 500
MIN_TOP_K, MAX_TOP_K = 1, 10


def llm_configured() -> bool:
    """LLM считается настроенной, если есть ключ, адрес и модель."""
    return bool(GIGACHAT_AUTH_KEY and LLM_BASE_URL and LLM_MODEL)
