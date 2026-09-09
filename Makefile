.PHONY: help build up down restart logs logs-app shell clean deploy backup status test

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

# shell-redis / shell-db used to be here, exec-ing into "redis" and "db"
# compose services. Neither service has ever existed in docker-compose.yml:
# Redis runs outside Compose (REDIS_URL points at the host's
# 172.17.0.1:6379) and there is no Postgres anywhere in this project - see
# src/core/config.py and docker-compose.prod.yml for the matching cleanup.
# Both targets always failed with "no such service".

clean: ## Остановить и удалить все volumes.
	docker-compose down -v
	@echo "Все данные удалены."

deploy: ## Развернуть в продакшене (Docker Compose, prod overlay).
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build

backup: ## Создать бэкап данных.
	@mkdir -p backups
	@tar -czf backups/backup_$$(date +%Y%m%d_%H%M%S).tar.gz data/
	@echo "Бэкап создан в backups/"

status: ## Показать статус контейнеров.
	docker-compose ps

test: ## Запустить тесты (pip install -r requirements-dev.txt first).
	pytest tests/ -v
