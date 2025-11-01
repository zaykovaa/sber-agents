#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Telegram бот с ролью Эксперт кино.
Ведет диалог с пользователями и помогает с рекомендациями фильмов и сериалов.
"""
import os
import logging
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
import asyncio

class FilmExpertBot:
    def __init__(self):
        load_dotenv()
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.bot = Bot(token=self.token)
        self.dp = Dispatcher()

    async def start_handler(self, message: types.Message):
        await message.answer("👋 Привет! Я Эксперт по кино. Используйте /help для справки.")
        self.logger.info(f"/start от {message.from_user.id}")

    async def help_handler(self, message: types.Message):
        await message.answer("Доступные команды:\n/start — приветствие\n/help — справка")
        self.logger.info(f"/help от {message.from_user.id}")

    def register_handlers(self):
        self.dp.message(Command("start"))(self.start_handler)
        self.dp.message(Command("help"))(self.help_handler)

    def run(self):
        self.register_handlers()
        self.logger.info("Бот запускается...")
        asyncio.run(self.dp.start_polling(self.bot))

