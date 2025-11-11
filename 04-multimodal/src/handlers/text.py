"""Обработчик текстовых сообщений."""
import logging
from aiogram import types
try:
    from ..storage import ConversationStorage
    from ..services.llm import LLMService
except ImportError:
    # Для запуска как скрипта
    from storage import ConversationStorage
    from services.llm import LLMService

logger = logging.getLogger(__name__)


class TextHandler:
    """Обработчик текстовых сообщений."""
    
    def __init__(self, storage: ConversationStorage, llm_service: LLMService):
        self.storage = storage
        self.llm_service = llm_service
    
    async def handle(self, message: types.Message):
        """Обрабатывает текстовое сообщение.
        
        Args:
            message: Сообщение от пользователя
        """
        uid = message.from_user.id
        
        # Проверяем, что сообщение текстовое
        if not message.text or not message.text.strip():
            await message.answer("Пожалуйста, отправьте текстовое сообщение или изображение.")
            logger.warning(f"Получено не текстовое сообщение от пользователя {uid}")
            return
        
        # Добавляем сообщение пользователя в историю
        self.storage.add_message(uid, "user", message.text)
        
        # Генерируем ответ
        history = self.storage.get_conversation_history(uid, use_vision=False)
        response = await self.llm_service.generate_response(history, use_vision=False)
        
        # Проверяем, что ответ не пустой перед добавлением в историю
        if response and response.strip():
            self.storage.add_message(uid, "assistant", response)
            await message.answer(response)
            self.storage.increment_messages()
            logger.info(f"Ответ отправлен пользователю {uid}")
        else:
            error_msg = "Извините, произошла ошибка при генерации ответа. Попробуйте еще раз."
            await message.answer(error_msg)
            logger.error(f"Пустой ответ от LLM для пользователя {uid}")

