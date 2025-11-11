"""Сервис для обработки изображений из Telegram."""
import base64
import logging
from aiogram import Bot
from aiogram.types import PhotoSize

logger = logging.getLogger(__name__)


class ImageService:
    """Сервис для работы с изображениями."""
    
    def __init__(self, bot: Bot):
        self.bot = bot
    
    async def download_image(self, photo: PhotoSize) -> str:
        """Загружает изображение из Telegram и конвертирует в base64.
        
        Args:
            photo: Объект PhotoSize из Telegram
            
        Returns:
            Data URL строка с base64 изображением (data:image/jpeg;base64,...)
        """
        try:
            # Скачиваем файл
            file = await self.bot.get_file(photo.file_id)
            file_data = await self.bot.download_file(file.file_path)
            
            # Читаем байты и конвертируем в base64
            image_bytes = file_data.read()
            
            # Проверяем размер изображения (некоторые модели имеют ограничения)
            max_size_mb = 20  # Максимальный размер ~20MB
            size_mb = len(image_bytes) / (1024 * 1024)
            if size_mb > max_size_mb:
                logger.warning(f"Изображение слишком большое: {size_mb:.2f}MB")
            
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            # Определяем MIME тип на основе расширения
            mime_type = "image/jpeg"
            if file.file_path:
                ext = file.file_path.lower().split('.')[-1]
                if ext == 'png':
                    mime_type = "image/png"
                elif ext == 'gif':
                    mime_type = "image/gif"
                elif ext == 'webp':
                    mime_type = "image/webp"
            
            logger.info(f"Изображение загружено: {size_mb:.2f}MB, тип: {mime_type}")
            return f"data:{mime_type};base64,{base64_image}"
        except Exception as e:
            logger.error(f"Ошибка при загрузке изображения: {e}")
            raise
    
    def create_multimodal_content(self, image_url: str, caption: str = None) -> list:
        """Создает мультимодальное содержимое для отправки в LLM.
        
        Args:
            image_url: Data URL изображения
            caption: Текстовая подпись к изображению (опционально)
            
        Returns:
            Список с текстовым и графическим содержимым
        """
        content = []
        
        if caption and caption.strip():
            content.append({
                "type": "text",
                "text": caption
            })
        else:
            content.append({
                "type": "text",
                "text": "Что изображено на этом фото? Расскажи о нем."
            })
        
        content.append({
            "type": "image_url",
            "image_url": {
                "url": image_url
            }
        })
        
        return content


