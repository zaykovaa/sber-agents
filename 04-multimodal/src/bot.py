#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Telegram бот с ролью Эксперт кино.
Ведет диалог с пользователями и помогает с рекомендациями фильмов и сериалов.
"""
import logging
import asyncio
from aiogram import Bot, Dispatcher
from aiogram.filters import Command

try:
    from .config import TELEGRAM_BOT_TOKEN
    from .storage import ConversationStorage
    from .services.llm import LLMService
    from .services.image import ImageService
    from .handlers.commands import CommandHandlers
    from .handlers.text import TextHandler
    from .handlers.image import ImageHandler
except ImportError:
    # Для запуска как скрипта
    from config import TELEGRAM_BOT_TOKEN
    from storage import ConversationStorage
    from services.llm import LLMService
    from services.image import ImageService
    from handlers.commands import CommandHandlers
    from handlers.text import TextHandler
    from handlers.image import ImageHandler

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class FilmExpertBot:
    """Главный класс бота."""
    
    def __init__(self):
        # Инициализация бота
        self.bot = Bot(token=TELEGRAM_BOT_TOKEN)
        self.dp = Dispatcher()
        
        # Инициализация сервисов
        self.storage = ConversationStorage()
        self.llm_service = LLMService()
        self.image_service = ImageService(self.bot)
        
        # Инициализация обработчиков
        self.command_handlers = CommandHandlers(self.storage)
        self.text_handler = TextHandler(self.storage, self.llm_service)
        self.image_handler = ImageHandler(
            self.storage,
            self.llm_service,
            self.image_service
        )

    def register_handlers(self):
        """Регистрирует все обработчики сообщений."""
        # Команды
        self.dp.message(Command("start"))(self.command_handlers.start_handler)
        self.dp.message(Command("help"))(self.command_handlers.help_handler)
        self.dp.message(Command("clear"))(self.command_handlers.clear_handler)
        
        # Обработчик изображений (должен быть перед текстовым)
        self.dp.message(lambda m: m.photo is not None)(self.image_handler.handle)
        # Обработчик текстовых сообщений
        self.dp.message()(self.text_handler.handle)

    async def run(self):
        """Запускает бота."""
        self.register_handlers()
        logger.info("Бот запускается...")
        await self.dp.start_polling(self.bot)


async def main():
    """Главная функция для запуска бота."""
    bot = FilmExpertBot()
    try:
        await bot.run()
    except KeyboardInterrupt:
        logger.info("Бот останавливается...")
    finally:
        await bot.bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
