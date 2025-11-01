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
from typing import Dict, List

SYSTEM_PROMPT = (
    "Ты — профессиональный эксперт в области кино и сериалов, опытный советчик по фильмам. "
    "Твоя задача — помогать пользователям находить идеальный контент, знаешь тренды, жанры, без спойлеров. "
    "Общайся кратко, дружелюбно, профессионально."
)

class FilmExpertBot:
    def __init__(self):
        load_dotenv()
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.token:
            self.logger.error("TELEGRAM_BOT_TOKEN не найден в .env, бот завершает работу.")
            exit(1)
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            self.logger.error("OPENROUTER_API_KEY не найден в .env, бот завершает работу.")
            exit(1)
        self.bot = Bot(token=self.token)
        self.dp = Dispatcher()
        self.model_name = os.getenv("MODEL_NAME", "openai/gpt-3.5-turbo")
        self.llm = AsyncOpenAI(
            api_key=api_key,
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        )
        self.conversations: Dict[int, List[Dict[str, str]]] = {}
        self.max_history = int(os.getenv("MAX_HISTORY_MESSAGES", "10"))
        self.stats = {"total_users": 0, "total_messages": 0}

    def get_conversation_history(self, user_id: int) -> List[Dict[str, str]]:
        if user_id not in self.conversations or not self.conversations[user_id]:
            self.conversations[user_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
            self.stats["total_users"] += 1
        elif self.conversations[user_id][0].get("role") != "system":
            self.conversations[user_id].insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        return self.conversations[user_id]

    def add_message(self, user_id: int, role: str, content: str):
        history = self.get_conversation_history(user_id)
        history.append({"role": role, "content": content})
        if len(history) > self.max_history:
            system_prompt = history[0]
            rest = history[-(self.max_history - 1):]
            self.conversations[user_id] = [system_prompt] + [m for m in rest if m.get("role") != "system"]

    def clear_conversation(self, user_id: int):
        self.conversations[user_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

    async def start_handler(self, message: types.Message):
        self.clear_conversation(message.from_user.id)
        await message.answer("👋 Привет! Я Эксперт по кино. Используйте /help для справки.")
        self.logger.info(f"/start от {message.from_user.id}")

    async def help_handler(self, message: types.Message):
        await message.answer("Доступные команды:\n/start — приветствие\n/help — справка\n/clear — очистить историю")
        self.logger.info(f"/help от {message.from_user.id}")

    async def clear_handler(self, message: types.Message):
        self.clear_conversation(message.from_user.id)
        await message.answer("История диалога очищена!")
        self.logger.info(f"/clear от {message.from_user.id}")

    async def generate_response(self, user_id: int) -> str:
        history = self.get_conversation_history(user_id)
        try:
            response = await self.llm.chat.completions.create(
                model=self.model_name,
                messages=history,
            )
            result = response.choices[0].message.content.strip()
            self.logger.info(f"Ответ сгенерирован для {user_id}")
            return result
        except Exception as e:
            self.logger.error(f"LLM error: {e}")
            return "Ошибка генерации ответа, попробуйте еще раз."

    async def text_handler(self, message: types.Message):
        uid = message.from_user.id
        self.add_message(uid, "user", message.text)
        response = await self.generate_response(uid)
        self.add_message(uid, "assistant", response)
        await message.answer(response)
        self.stats["total_messages"] += 1
        self.logger.info(f"Ответ отправлен пользователю {uid}")

    def register_handlers(self):
        self.dp.message(Command("start"))(self.start_handler)
        self.dp.message(Command("help"))(self.help_handler)
        self.dp.message(Command("clear"))(self.clear_handler)
        self.dp.message()(self.text_handler)

    def run(self):
        self.register_handlers()
        self.logger.info("Бот запускается...")
        asyncio.run(self.dp.start_polling(self.bot))

