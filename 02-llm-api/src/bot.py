#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI бот для взаимодействия с LLM через OpenRouter.
Демонстрирует работу с историей диалога, метриками и красивым выводом.
"""
# Убедитесь, что PYTHONIOENCODING=utf-8 установлен в вашем окружении
import os
import sys

# Принудительно устанавливаем UTF-8 кодировку для stdout/stderr
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')
from typing import List, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich import box


# Инициализация Rich консоли для красивого вывода
console = Console()

# Системный промпт - определяет роль и поведение ассистента
# ЗАДАНИЕ: Вставьте сюда ваш системный промпт, который определит поведение бота
# Например: "Ты — профессиональный банковский консультант..."
SYSTEM_PROMPT = """Ты — профессиональный консультант банка. 
Помогай клиентам с вопросами о счетах, картах, кредитах и вкладах. 
Отвечай вежливо, профессионально и по существу. 
Если не знаешь точного ответа — честно признайся и предложи обратиться к специалисту."""


class ChatBot:
    """Простой CLI бот для общения с LLM."""
    
    MAX_MESSAGES = 10  # Максимальное количество сообщений в истории
    
    def __init__(self):
        """Инициализация бота с загрузкой конфигурации."""
        # Загружаем переменные окружения из .env
        load_dotenv()
        
        # Получаем конфигурацию
        api_key = os.getenv("OPENROUTER_API_KEY")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.model_name = os.getenv("MODEL_NAME", "openai/gpt-3.5-turbo")
        
        if not api_key:
            console.print("[red]❌ Ошибка: OPENROUTER_API_KEY не найден в .env файле![/red]")
            sys.exit(1)
        
        # Очищаем API ключ от не-ASCII символов (например, "ваш-" из примера)
        # Простое решение: оставляем только ASCII символы
        cleaned_key = ''.join(c if ord(c) < 128 else '' for c in api_key)
        
        # Удаляем двойные дефисы и пробелы, которые могли появиться
        cleaned_key = cleaned_key.replace('--', '-').strip()
        
        # Если ключ содержит несколько вхождений 'sk-or-v1', берем последнюю часть
        if cleaned_key.count('sk-or-v1') > 1:
            # Находим последнее вхождение и берем все после него
            last_pos = cleaned_key.rfind('sk-or-v1')
            cleaned_key = cleaned_key[last_pos:]
        
        # Проверяем, что ключ начинается с sk-or-v1
        if not cleaned_key.startswith('sk-or-v1-'):
            console.print(f"[yellow]⚠️  Предупреждение: API ключ может быть некорректным после очистки[/yellow]")
            console.print(f"[yellow]Очищенный ключ (первые 20 символов): {cleaned_key[:20]}...[/yellow]\n")
        
        # Инициализируем OpenAI клиент для работы с OpenRouter
        self.client = OpenAI(
            api_key=cleaned_key,
            base_url=base_url,
        )
        
        # История диалога (список сообщений)
        self.conversation_history: List[Dict[str, str]] = []
        
        # Добавляем системный промпт в начало, если он задан
        if SYSTEM_PROMPT:
            self.conversation_history.append({
                "role": "system",
                "content": SYSTEM_PROMPT
            })
        
        # Метрики для отслеживания
        self.session_metrics = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "messages_count": 0,
        }
    
    def add_message(self, role: str, content: str):
        """Добавить сообщение в историю диалога."""
        # Добавляем новое сообщение
        self.conversation_history.append({
            "role": role,
            "content": content
        })
        
        # Ограничиваем количество сообщений, сохраняя системный промпт
        if len(self.conversation_history) > self.MAX_MESSAGES:
            # Находим индекс системного промпта (обычно первый элемент)
            system_index = -1
            for i, msg in enumerate(self.conversation_history):
                if msg.get("role") == "system":
                    system_index = i
                    break
            
            # Если есть системный промпт
            if system_index >= 0:
                # Берем системный промпт и последние N сообщений
                # Оставляем системный промпт + последние (MAX_MESSAGES-1) сообщений
                new_history = [self.conversation_history[system_index]]  # Системный промпт
                remaining_messages = self.conversation_history[system_index+1:]  # Все после системного промпта
                # Берем последние (MAX_MESSAGES-1) сообщений
                new_history.extend(remaining_messages[-(self.MAX_MESSAGES-1):])
                self.conversation_history = new_history
            else:
                # Если нет системного промпта, просто оставляем последние N сообщений
                self.conversation_history = self.conversation_history[-self.MAX_MESSAGES:]
    
    def clear_history(self):
        """Очистить историю диалога."""
        self.conversation_history = []
        # Восстанавливаем системный промпт, если он был задан
        if SYSTEM_PROMPT:
            self.conversation_history.append({
                "role": "system",
                "content": SYSTEM_PROMPT
            })
        console.print("[yellow]📝 История диалога очищена[/yellow]\n")
    
    def summarize_history(self):
        """Суммаризовать длинную историю диалога."""
        # Проверяем, есть ли что суммаризировать
        if len(self.conversation_history) <= 3:
            console.print("[yellow]История слишком короткая для суммаризации[/yellow]\n")
            return
        
        try:
            # 1. Находим системный промпт (если есть)
            system_index = -1
            for i, msg in enumerate(self.conversation_history):
                if msg.get("role") == "system":
                    system_index = i
                    break
            
            # 2. Определяем сообщения для суммаризации
            # Берем старые сообщения, кроме последних 3-4 (оставляем контекст)
            keep_recent = 3  # Сколько последних сообщений оставляем
            if system_index >= 0:
                messages_to_summarize = self.conversation_history[system_index+1:-keep_recent]
            else:
                messages_to_summarize = self.conversation_history[:-keep_recent]
            
            if len(messages_to_summarize) < 2:
                console.print("[yellow]Недостаточно сообщений для суммаризации[/yellow]\n")
                return
            
            # 3. Формируем промпт для суммаризации
            summary_prompt = "Пожалуйста, создай краткое резюме следующего диалога, сохраняя ключевые моменты и контекст:\n\n"
            
            # Форматируем сообщения для суммаризации
            formatted_messages = []
            for msg in messages_to_summarize:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    formatted_messages.append(f"Пользователь: {content}")
                elif role == "assistant":
                    formatted_messages.append(f"Ассистент: {content}")
            
            summary_prompt += "\n".join(formatted_messages)
            summary_prompt += "\n\nКраткое резюме:"
            
            # 4. Отправляем запрос на суммаризацию
            console.print("[yellow]📝 Суммаризирую историю диалога...[/yellow]")
            
            with console.status("[bold yellow]Суммаризация...", spinner="dots"):
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "Ты помогаешь создавать краткие резюме диалогов."},
                        {"role": "user", "content": summary_prompt}
                    ],
                )
            
            summary = response.choices[0].message.content
            
            # 5. Заменяем старые сообщения на резюме
            if system_index >= 0:
                # Сохраняем системный промпт
                system_prompt = self.conversation_history[system_index]
                # Берем последние сообщения
                recent_messages = self.conversation_history[-keep_recent:]
                # Создаем новую историю: системный промпт + резюме + недавние сообщения
                new_history = [
                    system_prompt,
                    {"role": "assistant", "content": f"[Резюме прошлых сообщений] {summary}"},
                ]
                new_history.extend(recent_messages)
                self.conversation_history = new_history
            else:
                # Нет системного промпта
                recent_messages = self.conversation_history[-keep_recent:]
                new_history = [
                    {"role": "assistant", "content": f"[Резюме прошлых сообщений] {summary}"},
                ]
                new_history.extend(recent_messages)
                self.conversation_history = new_history
            
            console.print("[green]✓ История успешно суммаризирована[/green]\n")
            
            # Показываем метрики
            if response.usage:
                self.display_metrics(response.usage.model_dump(), response.choices[0].finish_reason)
            
        except Exception as e:
            console.print("[red]Ошибка при суммаризации истории[/red]")
            self._safe_print_error(e)
    
    def display_metrics(self, usage: Optional[dict], finish_reason: Optional[str] = None):
        """Отобразить метрики и метаданные ответа."""
        if not usage:
            return
        
        # Извлекаем данные об использовании токенов
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        
        # Обновляем сессионные метрики
        self.session_metrics["total_prompt_tokens"] += prompt_tokens
        self.session_metrics["total_completion_tokens"] += completion_tokens
        self.session_metrics["total_tokens"] += total_tokens
        self.session_metrics["messages_count"] += 1
        
        # Создаем таблицу с метриками текущего ответа
        table = Table(title="📊 Метрики ответа", box=box.ROUNDED, show_header=True)
        table.add_column("Параметр", style="cyan")
        table.add_column("Значение", style="green")
        
        table.add_row("Модель", self.model_name)
        table.add_row("Prompt токены", str(prompt_tokens))
        table.add_row("Completion токены", str(completion_tokens))
        table.add_row("Всего токены", str(total_tokens))
        
        if finish_reason:
            table.add_row("Finish reason", finish_reason)
        
        console.print(table)
        
        # Таблица с накопленными метриками сессии
        session_table = Table(title="🎯 Статистика сессии", box=box.ROUNDED)
        session_table.add_column("Параметр", style="cyan")
        session_table.add_column("Значение", style="magenta")
        
        session_table.add_row("Сообщений", str(self.session_metrics["messages_count"]))
        session_table.add_row("Всего токенов", str(self.session_metrics["total_tokens"]))
        
        console.print(session_table)
        console.print()
    
    def display_stats(self):
        """Показать статистику сессии."""
        console.print("\n[bold cyan]📈 Статистика текущей сессии:[/bold cyan]")
        
        stats_table = Table(box=box.DOUBLE)
        stats_table.add_column("Метрика", style="cyan", no_wrap=True)
        stats_table.add_column("Значение", style="green")
        
        stats_table.add_row("Модель", self.model_name)
        stats_table.add_row("Сообщений в сессии", str(self.session_metrics["messages_count"]))
        stats_table.add_row("Сообщений в истории", str(len(self.conversation_history)))
        stats_table.add_row("Prompt токены", str(self.session_metrics["total_prompt_tokens"]))
        stats_table.add_row("Completion токены", str(self.session_metrics["total_completion_tokens"]))
        stats_table.add_row("Всего токены", str(self.session_metrics["total_tokens"]))
        
        console.print(stats_table)
        console.print()
    
    def _safe_print_error(self, e: Exception):
        """Безопасный вывод ошибки без проблем с кодировкой."""
        error_type = type(e).__name__
        
        # Попытка 1: безопасный ASCII вывод
        try:
            # Фильтруем только ASCII символы ДО формирования строки
            safe_type = ''.join(c if ord(c) < 128 else '' for c in error_type)
            print(f"Ошибка при обращении к LLM ({safe_type})", file=sys.stderr)
            print(file=sys.stderr)
        except Exception:
            pass
        
        # Попытка 2: выводим тип через байты
        try:
            error_bytes = error_type.encode('ascii', errors='replace')
            print(f"Ошибка типа: {error_bytes.decode('ascii')}", file=sys.stderr)
            print(file=sys.stderr)
        except Exception:
            # Попытка 3: просто выводим что есть ошибка
            try:
                sys.stderr.write("Ошибка при обращении к LLM\n")
                sys.stderr.write("\n")
            except Exception:
                pass
    
    def send_message(self, user_message: str) -> Optional[str]:
        """Отправить сообщение в LLM и получить ответ."""
        # Добавляем сообщение пользователя в историю
        self.add_message("user", user_message)
        
        try:
            # Показываем индикатор загрузки
            with console.status("[bold green]🤔 Думаю...", spinner="dots"):
                # Отправляем запрос с полной историей диалога
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self.conversation_history,
                )
            
            # Извлекаем ответ
            assistant_message = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            
            # Добавляем ответ в историю
            self.add_message("assistant", assistant_message)
            
            # Отображаем ответ
            console.print(Panel(
                Markdown(assistant_message),
                title="🤖 Ассистент",
                border_style="blue",
                padding=(1, 2)
            ))
            
            # Показываем метрики
            self.display_metrics(response.usage.model_dump() if response.usage else None, finish_reason)
            
            return assistant_message
            
        except Exception as e:
            # Безопасная обработка ошибок с дополнительной защитой
            try:
                self._safe_print_error(e)
            except Exception:
                # Даже если вывод ошибки упал, продолжаем работу
                pass
            # Удаляем последнее сообщение пользователя из истории, так как запрос не удался
            try:
                if self.conversation_history and self.conversation_history[-1]["role"] == "user":
                    self.conversation_history.pop()
            except Exception:
                pass
            return None
    
    def show_welcome(self):
        """Показать приветственное сообщение."""
        welcome_text = """
# 🤖 CLI LLM Бот

Образовательный проект для работы с LLM через OpenRouter API.

**Доступные команды:**
- `/exit` - выход из программы
- `/clear` - очистить историю диалога
- `/summarize` - суммаризировать историю диалога
- `/stats` - показать статистику сессии
- `/help` - показать эту справку

Начните диалог с вопроса или сообщения!
        """
        console.print(Panel(
            Markdown(welcome_text),
            title="📖 Справка",
            border_style="green",
            padding=(1, 2)
        ))
        
        if not SYSTEM_PROMPT:
            console.print("[yellow]⚠️  Системный промпт не задан. Отредактируйте SYSTEM_PROMPT в src/bot.py[/yellow]\n")
        else:
            console.print("[green]✓ Системный промпт активен[/green]\n")
    
    def run(self):
        """Запустить основной цикл бота (REPL)."""
        self.show_welcome()
        
        try:
            while True:
                # Получаем ввод пользователя
                try:
                    user_input = console.input("[bold cyan]👤 Вы:[/bold cyan] ").strip()
                except EOFError:
                    break
                
                if not user_input:
                    continue
                
                # Обработка команд
                if user_input.startswith("/"):
                    command = user_input.lower()
                    
                    if command == "/exit":
                        console.print("[yellow]👋 До свидания![/yellow]")
                        break
                    
                    elif command == "/clear":
                        self.clear_history()
                        continue
                    
                    elif command == "/summarize":
                        self.summarize_history()
                        continue
                    
                    elif command == "/stats":
                        self.display_stats()
                        continue
                    
                    elif command == "/help":
                        self.show_welcome()
                        continue
                    
                    else:
                        console.print(f"[red]❌ Неизвестная команда: {user_input}[/red]")
                        console.print("[yellow]Используйте /help для справки[/yellow]\n")
                        continue
                
                # Отображаем сообщение пользователя
                console.print(Panel(
                    user_input,
                    title="👤 Вы",
                    border_style="cyan",
                    padding=(1, 2)
                ))
                
                # Отправляем сообщение и получаем ответ
                self.send_message(user_input)
        
        except KeyboardInterrupt:
            console.print("\n[yellow]👋 Прервано пользователем. До свидания![/yellow]")
        
        # Показываем финальную статистику
        if self.session_metrics["messages_count"] > 0:
            console.print("\n[bold green]📊 Финальная статистика сессии:[/bold green]")
            self.display_stats()


def main():
    """Точка входа в программу."""
    bot = ChatBot()
    bot.run()


if __name__ == "__main__":
    main()

