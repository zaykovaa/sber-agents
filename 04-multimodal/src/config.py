"""Конфигурация бота."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Путь к корню проекта
PROJECT_ROOT = Path(__file__).parent.parent

# Загружаем .env файл из корня проекта
env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)


def load_prompt(prompt_file_path: str, env_var: str = None) -> str:
    """Загружает промпт из файла или переменной окружения.
    
    Args:
        prompt_file_path: Путь к файлу с промптом (относительно корня проекта)
        env_var: Имя переменной окружения для переопределения промпта (опционально)
        
    Returns:
        Строка с содержимым промпта
    """
    # Сначала пробуем загрузить из переменной окружения напрямую
    if env_var:
        env_value = os.getenv(env_var)
        if env_value:
            return env_value
    
    # Если переменной нет, пробуем загрузить из файла
    prompt_path = PROJECT_ROOT / prompt_file_path if not os.path.isabs(prompt_file_path) else Path(prompt_file_path)
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    
    # Если файл не найден, возвращаем пустую строку
    return ""


# Telegram Bot
# Поддерживаем оба варианта для обратной совместимости
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    raise ValueError(
        "TELEGRAM_BOT_TOKEN (или TELEGRAM_TOKEN) не найден в .env. "
        "Добавьте одну из этих переменных в .env файл."
    )

# OpenRouter/OpenAI API (поддерживаем оба варианта для обратной совместимости)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY (или OPENAI_API_KEY) не найден в .env. "
        "Добавьте одну из этих переменных в .env файл."
    )

# Поддерживаем оба варианта для BASE_URL
OPENROUTER_BASE_URL = (
    os.getenv("OPENROUTER_BASE_URL") 
    or os.getenv("OPENAI_BASE_URL") 
    or "https://openrouter.ai/api/v1"
)

# Models (поддерживаем оба варианта для обратной совместимости)
MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv("MODEL_TEXT") or "openai/gpt-3.5-turbo"
MODEL_IMAGE = os.getenv("MODEL_IMAGE") or MODEL_NAME

# Bot Settings
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "10"))

# System Prompts
SYSTEM_PROMPT_TEXT = load_prompt(
    os.getenv("SYSTEM_PROMPT_TEXT_PATH", "prompts/system_prompt_text.txt"),
    "SYSTEM_PROMPT_TEXT"
)
SYSTEM_PROMPT_IMAGE = load_prompt(
    os.getenv("SYSTEM_PROMPT_IMAGE_PATH", "prompts/system_prompt_image.txt"),
    "SYSTEM_PROMPT_IMAGE"
)


# Для обратной совместимости используем текстовый промпт как основной
SYSTEM_PROMPT = SYSTEM_PROMPT_TEXT or (
    "Ты — профессиональный эксперт в области кино и сериалов, опытный советчик по фильмам. "
    "Твоя задача — помогать пользователям находить идеальный контент, знаешь тренды, жанры, без спойлеров. "
    "Общайся кратко, дружелюбно, профессионально."
)

