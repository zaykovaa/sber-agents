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
from openai import AsyncOpenAI

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
        self.model_name = os.getenv("MODEL_NAME", "openai/gpt-3.5-turbo")
        self.llm = AsyncOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        )

    async def start_handler(self, message: types.Message):
        await message.answer("👋 Привет! Я Эксперт по кино. Используйте /help для справки.")
        self.logger.info(f"/start от {message.from_user.id}")

    async def help_handler(self, message: types.Message):
        await message.answer("Доступные команды:\n/start — приветствие\n/help — справка")
        self.logger.info(f"/help от {message.from_user.id}")

    async def generate_response(self, user_id, text):
        try:
            response = await self.llm.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": text}],
            )
            result = response.choices[0].message.content.strip()
            # Можно логировать response.usage.total_tokens при необходимости
            return result
        except Exception as e:
            self.logger.error(f"LLM error: {e}")
            return "Ошибка генерации ответа, попробуйте еще раз."

    async def text_handler(self, message: types.Message):
        self.logger.info(f"Получено текстовое сообщение от {message.from_user.id}")
        response = await self.generate_response(message.from_user.id, message.text)
        await message.answer(response)

    def register_handlers(self):
        self.dp.message(Command("start"))(self.start_handler)
        self.dp.message(Command("help"))(self.help_handler)
        self.dp.message()(self.text_handler)  # все остальные текстовые сообщения

    def run(self):
        self.register_handlers()
        self.logger.info("Бот запускается...")
        asyncio.run(self.dp.start_polling(self.bot))

