# Правила разработки кода

Этот документ содержит правила для code-ассистента при генерации кода для проекта.

## Основные принципы

### KISS и YAGNI
- **Один файл** `src/bot.py` — вся логика бота
- **Один класс** `FilmExpertBot` — вся функциональность
- **Никаких абстракций** сверх необходимого
- **Без "на будущее"** — только то, что нужно сейчас
- **Минимум файлов** — только необходимое

### Асинхронность
- **Весь код async/await** — aiogram и LLM клиент асинхронные
- **Нет блокирующих операций** — используем async вызовы
- **Простой polling** — `bot.run_polling()` для получения сообщений

## Структура кода

### Класс бота
```python
class FilmExpertBot:
    def __init__(self):
        # Инициализация: загрузка конфигурации, клиенты
        pass
    
    # Обработчики команд
    async def start_handler(self, message): pass
    async def help_handler(self, message): pass
    async def clear_handler(self, message): pass
    
    # Обработчик текста
    async def message_handler(self, message): pass
    
    # Логика LLM
    async def generate_response(self, user_id, text): pass
    
    # Запуск
    def run(self): pass
```

### История диалога
- `conversations: Dict[int, List[Dict[str, str]]]` — в памяти
- Формат: `{"role": "system|user|assistant", "content": "..."}`
- Ограничение: максимум 10 сообщений (настраивается)
- При очистке: сохраняем системный промпт

## Работа с LLM

### Запрос
```python
response = await self.client.chat.completions.create(
    model=self.model_name,
    messages=history,  # Полная история
)
```

### Правила
- **Всегда полная история** — весь контекст диалога
- **Стандартный API** — без дополнительных параметров
- **Прямой прокидывание** — без обработки/валидации
- **Без суммаризаций** — не сокращаем историю

## Обработка ошибок

### Принципы
- **Логируем ошибку** — `logger.error(...)`
- **Не падаем** — бот продолжает работу
- **Сообщаем пользователю** — простое сообщение об ошибке
- **История не повреждается** — не удаляем сообщения при ошибке

### Не делаем
- ❌ Сложные retry логики
- ❌ Кэширование ответов
- ❌ Фолбэки на другие модели

## Логирование

### Формат
```python
import logging

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
```

### Что логируем
- `INFO`: начало работы, получение сообщений, отправка ответов
- `ERROR`: ошибки LLM, конфигурации

### Не логируем
- ❌ Содержимое сообщений полностью
- ❌ В файлы
- ❌ Детальные метрики токенов

## Конфигурация

### Загрузка
```python
from dotenv import load_dotenv
load_dotenv()

token = os.getenv("TELEGRAM_BOT_TOKEN")
api_key = os.getenv("OPENROUTER_API_KEY")
model = os.getenv("MODEL_NAME", "openai/gpt-3.5-turbo")
```

### Принципы
- **Читаем из .env** — `python-dotenv`
- **Проверяем обязательные** — если нет, логируем и выходим
- **Дефолты для опциональных** — указаны в коде
- **Без валидации** — только проверка наличия

## Именование

### Стиль
- **Функции/методы**: `snake_case` (async функции с `async def`)
- **Классы**: `PascalCase`
- **Константы**: `UPPER_SNAKE_CASE`
- **Простые имена**: понятно из контекста

### Примеры
- ✅ `async def message_handler(message)` 
- ✅ `async def generate_response(user_id, text)`
- ✅ `MAX_HISTORY_MESSAGES = 10`

## Аннотации типов

### Принцип
- **Базовые типы** — `str`, `int`, `List`, `Dict`
- **Для сложных** — typing imports
- **Не перегружаем** — только где понятность

### Примеры
```python
from typing import Dict, List, Optional

conversations: Dict[int, List[Dict[str, str]]]
async def get_history(user_id: int) -> List[Dict[str, str]]:
    ...
```

## Импорты

### Порядок
1. Стандартная библиотека
2. Внешние библиотеки
3. Локальные модули (если будут)

### Пример
```python
import os
import logging
from typing import Dict, List

from dotenv import load_dotenv
from openai import AsyncOpenAI
from aiogram import Bot, Dispatcher, Router

# Локальные
from src.bot import FilmExpertBot
```

## Комментарии

### Принцип
- **Минимум комментариев** — код должен говорить сам
- **Docstrings** — только для публичных методов
- **Неочевидная логика** — краткое пояснение

## Тестирование

### Принцип
- **Нет unit тестов** для MVP
- **Ручная проверка** в Telegram
- **Логи** для отладки

## Запрещено

- ❌ Создавать новые файлы без согласования
- ❌ Добавлять middleware, роутеры, handlers
- ❌ Использовать БД или файловое хранилище
- ❌ Создавать абстракции для "гибкости"
- ❌ Оптимизировать без необходимости
- ❌ Добавлять features из YAGNI списка
- ❌ Использовать сложные паттерны

## Разрешается

- ✅ Простой код в одном файле
- ✅ Прямая логика без слоев
- ✅ Минимальная обработка ошибок
- ✅ In-memory хранение истории
- ✅ Простой polling вместо webhook
- ✅ Стандартный logging

