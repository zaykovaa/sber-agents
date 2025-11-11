"""Обработчик изображений."""
import logging
from aiogram import types
try:
    from ..storage import ConversationStorage
    from ..services.llm import LLMService
    from ..services.image import ImageService
except ImportError:
    # Для запуска как скрипта
    from storage import ConversationStorage
    from services.llm import LLMService
    from services.image import ImageService

logger = logging.getLogger(__name__)


class ImageHandler:
    """Обработчик изображений."""
    
    def __init__(
        self,
        storage: ConversationStorage,
        llm_service: LLMService,
        image_service: ImageService
    ):
        self.storage = storage
        self.llm_service = llm_service
        self.image_service = image_service
    
    async def handle(self, message: types.Message):
        """Обрабатывает изображение.
        
        Args:
            message: Сообщение с изображением от пользователя
        """
        uid = message.from_user.id
        
        if not message.photo:
            await message.answer("Не удалось обработать изображение.")
            return
        
        # Берем фото наибольшего размера
        photo = message.photo[-1]
        caption = message.caption or ""
        
        try:
            # Показываем, что обрабатываем изображение
            await message.answer("🖼️ Обрабатываю изображение...")
            
            # Загружаем и конвертируем изображение
            image_url = await self.image_service.download_image(photo)
            
            # Формируем мультимодальное сообщение
            content = self.image_service.create_multimodal_content(image_url, caption)
            
            # Добавляем сообщение в историю
            self.storage.add_message(uid, "user", content)
            
            # Генерируем ответ с использованием vision модели
            history = self.storage.get_conversation_history(uid, use_vision=True)
            response = await self.llm_service.generate_response(history, use_vision=True)
            
            # Проверяем, что ответ не пустой
            if response and response.strip():
                self.storage.add_message(uid, "assistant", response)
                await message.answer(response)
                self.storage.increment_messages()
                logger.info(f"Ответ на изображение отправлен пользователю {uid}")
            else:
                error_msg = "Извините, произошла ошибка при генерации ответа. Попробуйте еще раз."
                await message.answer(error_msg)
                logger.error(f"Пустой ответ от LLM для пользователя {uid}")
                
        except Exception as e:
            logger.error(f"Ошибка при обработке изображения: {e}")
            await message.answer("Извините, произошла ошибка при обработке изображения. Попробуйте еще раз.")

