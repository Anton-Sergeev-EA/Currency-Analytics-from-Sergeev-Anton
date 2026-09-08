.PHONY: help build up down restart logs shell clean deploy backup

# Цвета для вывода.
GREEN=\033[0;32m
BLUE=\033[0;34m
RED=\033[0;31m
NC=\033[0m # No Color

help: ## Показать помощь.
	@printf "${BLUE}Доступные команды:${NC}\n"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "${GREEN}%-20s${NC} %s\n", $$1, $$2}'

build: ## Собрать образы.
	docker-compose build

up: ## Запустить все сервисы.
	docker-compose up -d
	@echo "Сервисы запущены."

down: ## Остановить все сервисы.
	docker-compose down

restart: down up ## Перезапустить все сервисы.

logs: ## Показать логи всех сервисов.
	docker-compose logs -f

logs-app: ## Показать логи только приложения.
	docker-compose logs -f currency-analytics

shell: ## Зайти в shell приложения.
	docker-compose exec currency-analytics /bin/bash

shell-redis: ## Зайти в Redis CLI.
	docker-compose exec redis redis-cli

shell-db: ## Зайти в PostgreSQL.
	docker-compose exec db psql -U $$POSTGRES_USER -d $$POSTGRES_DB

clean: ## Остановить и удалить все volumes.
	docker-compose down -v
	@echo "Все данные удалены."

deploy: ## Развернуть в продакшене.
	./deploy.sh

backup: ## Создать бэкап данных.
	@mkdir -p backups
	@tar -czf backups/backup_$$(date +%Y%m%d_%H%M%S).tar.gz data/ models/
	@echo "Бэкап создан в backups/"

status: ## Показать статус контейнеров.
	docker-compose ps

test: ## Запустить тесты (если есть).
	docker-compose exec currency-analytics pytest tests/
	