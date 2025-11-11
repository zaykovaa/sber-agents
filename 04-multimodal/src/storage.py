"""Хранение истории диалогов и статистики."""
from typing import Dict, List
try:
    from .config import SYSTEM_PROMPT_TEXT, SYSTEM_PROMPT_IMAGE, MAX_HISTORY_MESSAGES
except ImportError:
    # Для запуска как скрипта
    from config import SYSTEM_PROMPT_TEXT, SYSTEM_PROMPT_IMAGE, MAX_HISTORY_MESSAGES


class ConversationStorage:
    """Хранилище истории диалогов."""
    
    def __init__(self):
        self.conversations: Dict[int, List[Dict[str, str]]] = {}
        self.stats = {"total_users": 0, "total_messages": 0}
    
    def get_conversation_history(self, user_id: int, use_vision: bool = False) -> List[Dict[str, str]]:
        """Получает историю диалога для пользователя.
        
        Args:
            user_id: ID пользователя в Telegram
            use_vision: Использовать ли промпт для изображений
            
        Returns:
            Список сообщений в формате OpenAI Chat API
        """
        system_prompt = SYSTEM_PROMPT_IMAGE if use_vision else SYSTEM_PROMPT_TEXT
        if not system_prompt:
            # Fallback на текстовый промпт
            system_prompt = SYSTEM_PROMPT_TEXT or (
                "Ты — профессиональный эксперт в области кино и сериалов, "
                "опытный советчик по фильмам."
            )
        
        if user_id not in self.conversations or not self.conversations[user_id]:
            self.conversations[user_id] = [{"role": "system", "content": system_prompt}]
            self.stats["total_users"] += 1
        elif self.conversations[user_id][0].get("role") != "system":
            self.conversations[user_id].insert(0, {"role": "system", "content": system_prompt})
        return self.conversations[user_id]
    
    def add_message(self, user_id: int, role: str, content, use_vision: bool = False):
        """Добавляет сообщение в историю диалога.
        
        Args:
            user_id: ID пользователя в Telegram
            role: Роль сообщения ('user', 'assistant', 'system')
            content: Содержимое сообщения (строка или список для мультимодальных)
            use_vision: Использовать ли промпт для изображений (определяется автоматически, если content - список)
        """
        # Пропускаем сообщения с пустым или None содержимым
        if isinstance(content, str) and (not content or not content.strip()):
            return
        
        # Определяем, нужно ли использовать vision промпт
        if isinstance(content, list):
            use_vision = True
        
        history = self.get_conversation_history(user_id, use_vision=use_vision)
        history.append({"role": role, "content": content})
        
        # Ограничиваем длину истории
        if len(history) > MAX_HISTORY_MESSAGES:
            system_prompt = history[0]
            # Берем последние (MAX_HISTORY_MESSAGES - 1) сообщений, исключая системное
            rest = [
                m for m in history[1:] 
                if m.get("role") != "system"
            ][-(MAX_HISTORY_MESSAGES - 1):]
            self.conversations[user_id] = [system_prompt] + rest
    
    def clear_conversation(self, user_id: int, use_vision: bool = False):
        """Очищает историю диалога для пользователя.
        
        Args:
            user_id: ID пользователя в Telegram
            use_vision: Использовать ли промпт для изображений
        """
        system_prompt = SYSTEM_PROMPT_IMAGE if use_vision else SYSTEM_PROMPT_TEXT
        if not system_prompt:
            system_prompt = SYSTEM_PROMPT_TEXT or (
                "Ты — профессиональный эксперт в области кино и сериалов, "
                "опытный советчик по фильмам."
            )
        self.conversations[user_id] = [{"role": "system", "content": system_prompt}]
    
    def increment_messages(self):
        """Увеличивает счетчик обработанных сообщений."""
        self.stats["total_messages"] += 1

