.PHONY: setup run clean help

help:
	@echo "Доступные команды:"
	@echo "  make setup  - Установить зависимости через uv"
	@echo "  make run    - Запустить Telegram бот"
	@echo "  make clean  - Очистить временные файлы"

setup:
	@echo "Установка зависимостей..."
	uv sync
	@echo "✓ Зависимости установлены"
	@echo ""
	@echo "Не забудьте создать .env файл на основе .env.example"
	@echo "  cp .env.example .env"
	@echo "  # затем отредактируйте .env и добавьте ваши API ключи"

run:
	@if [ ! -f .env ]; then \
		echo "❌ Файл .env не найден!"; \
		echo "Скопируйте .env.example в .env и добавьте ваши API ключи:"; \
		echo "  cp .env.example .env"; \
		exit 1; \
	fi
	uv run python src/bot.py

clean:
	@echo "Очистка временных файлов..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	@echo "✓ Очистка завершена"

