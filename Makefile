.PHONY: build run test clean

build:
	docker build -t potibo-bot .

run:
	docker run --rm --env-file .env potibo-bot

test:
	python3 -m pytest app/tests/ -v

clean:
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true
