"""Обработчики команд бота."""
import logging
from aiogram import types
from aiogram.filters import Command
try:
    from ..storage import ConversationStorage
except ImportError:
    # Для запуска как скрипта
    from storage import ConversationStorage

logger = logging.getLogger(__name__)


class CommandHandlers:
    """Обработчики команд бота."""
    
    def __init__(self, storage: ConversationStorage):
        self.storage = storage
    
    async def start_handler(self, message: types.Message):
        """Обработчик команды /start."""
        self.storage.clear_conversation(message.from_user.id)
        await message.answer("👋 Привет! Я Эксперт по кино. Используйте /help для справки.")
        logger.info(f"/start от {message.from_user.id}")
    
    async def help_handler(self, message: types.Message):
        """Обработчик команды /help."""
        await message.answer(
            "Доступные команды:\n"
            "/start — приветствие\n"
            "/help — справка\n"
            "/clear — очистить историю\n\n"
            "Я умею:\n"
            "• Отвечать на текстовые вопросы о кино\n"
            "• Анализировать изображения (постеры, скриншоты и т.д.)"
        )
        logger.info(f"/help от {message.from_user.id}")
    
    async def clear_handler(self, message: types.Message):
        """Обработчик команды /clear."""
        self.storage.clear_conversation(message.from_user.id)
        await message.answer("История диалога очищена!")
        logger.info(f"/clear от {message.from_user.id}")

