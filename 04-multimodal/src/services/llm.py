"""Сервис для работы с LLM через OpenRouter API."""
import logging
import traceback
from typing import List, Dict
from openai import AsyncOpenAI
try:
    from ..config import (
        OPENROUTER_API_KEY,
        OPENROUTER_BASE_URL,
        MODEL_NAME,
        MODEL_IMAGE
    )
except ImportError:
    # Для запуска как скрипта
    from config import (
        OPENROUTER_API_KEY,
        OPENROUTER_BASE_URL,
        MODEL_NAME,
        MODEL_IMAGE
    )

logger = logging.getLogger(__name__)


class LLMService:
    """Сервис для взаимодействия с LLM."""
    
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL
        )
        self.model_name = MODEL_NAME
        self.model_image = MODEL_IMAGE
    
    async def generate_response(
        self,
        messages: List[Dict],
        use_vision: bool = False
    ) -> str:
        """Генерирует ответ от LLM на основе истории сообщений.
        
        Args:
            messages: История сообщений в формате OpenAI Chat API
            use_vision: Использовать ли vision-модель для обработки изображений
            
        Returns:
            Строка с ответом от LLM
        """
        # Фильтруем сообщения с пустым или None содержимым
        filtered_messages = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str) and content and str(content).strip():
                filtered_messages.append(msg)
            elif isinstance(content, list):
                # Мультимодальное сообщение
                filtered_messages.append(msg)
        
        if not filtered_messages:
            return "Не могу обработать пустое сообщение."
        
        # Выбираем модель в зависимости от наличия изображений
        model_to_use = self.model_image if use_vision else self.model_name
        
        # Логируем детали запроса для отладки
        logger.info(f"Запрос к модели: {model_to_use}, сообщений в истории: {len(filtered_messages)}")
        if use_vision:
            multimodal_count = sum(
                1 for msg in filtered_messages 
                if isinstance(msg.get("content"), list)
            )
            logger.info(f"Мультимодальных сообщений: {multimodal_count}")
        
        try:
            response = await self.client.chat.completions.create(
                model=model_to_use,
                messages=filtered_messages,
            )
            result = response.choices[0].message.content
            if not result or not result.strip():
                return "Получен пустой ответ от модели."
            result = result.strip()
            logger.info(f"Ответ сгенерирован (модель: {model_to_use})")
            return result
        except Exception as e:
            error_str = str(e)
            logger.error(f"LLM error (модель: {model_to_use}): {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Проверяем на различные типы ошибок
            if "403" in error_str or "unsupported_country" in error_str.lower() or "forbidden" in error_str.lower():
                return (
                    "⚠️ К сожалению, выбранная модель недоступна в вашем регионе. "
                    "Пожалуйста, попробуйте позже или используйте другую модель. "
                    "Для настройки модели измените MODEL_IMAGE в файле .env"
                )
            elif "400" in error_str or "bad_request" in error_str.lower():
                return (
                    "⚠️ Ошибка запроса к модели. Возможно, модель не поддерживает формат изображений или возникла проблема с данными. "
                    "Попробуйте другую модель или проверьте формат изображения."
                )
            elif "404" in error_str or "not_found" in error_str.lower():
                if "image" in error_str.lower() or "vision" in error_str.lower():
                    return (
                        f"⚠️ Модель '{model_to_use}' не поддерживает обработку изображений. "
                        "Используйте vision-модель, например:\n"
                        "• openai/gpt-4o-mini\n"
                        "• google/gemini-pro-vision\n"
                        "• anthropic/claude-3-haiku\n\n"
                        "Измените MODEL_IMAGE в файле .env"
                    )
                return (
                    f"⚠️ Модель '{model_to_use}' не найдена. "
                    "Проверьте правильность названия модели в MODEL_IMAGE в файле .env"
                )
            elif "401" in error_str or "unauthorized" in error_str.lower():
                return (
                    "⚠️ Ошибка авторизации. Проверьте правильность OPENROUTER_API_KEY в файле .env"
                )
            
            # Возвращаем детальное сообщение об ошибке для отладки
            error_msg = f"Ошибка генерации ответа: {error_str[:200]}"
            return error_msg

