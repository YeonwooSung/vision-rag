.PHONY: all dev prod create_collection docker_build docker_run docker_logs docker_stop


include .env

# ------------------------------------------------------------------------------
# Run Targets
# ------------------------------------------------------------------------------
create_collection:
	uv run python scripts/create_collection.py


# Run the server in development mode with hot-reloading.
dev:
	uv run uvicorn vision_rag.main:app:app \
		--host 0.0.0.0 \
		--port 8000 \
		--reload

# Run the server in production mode without reloading.
prod:
	uv run uvicorn vision_rag.main:app \
		--host 0.0.0.0 \
		--port 8000

# ------------------------------------------------------------------------------
# Docker
# ------------------------------------------------------------------------------

docker_build:
	docker build -t vision_rag:latest .

docker_run:
	docker run --rm --name vision_rag --env-file .env -p 8000:8000 -d vision_rag:latest

docker_logs:
	docker logs -f --tail=100 vision_rag

docker_stop:
	docker stop vision_rag
