.PHONY: help install run test clean

help:
	@echo "Commands:"
	@echo "  install  - Install dependencies"
	@echo "  run      - Run application"
	@echo "  test     - Run tests"
	@echo "  clean    - Clean cache"

install:
	pip install -r requirements.txt

run:
	python -m src.main

test:
	python -m pytest tests/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf logs/*.log
	