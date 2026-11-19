# One image for the gateway and the mock model server (same package, different command).
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# Dependencies first, so code changes do not reinstall them.
COPY pyproject.toml README.md ./
RUN mkdir -p src/vizwiz && touch src/vizwiz/__init__.py \
    && pip install -e ".[serve]" \
    && rm -rf src
COPY configs ./configs
COPY src ./src

RUN useradd --create-home --uid 10001 app
USER app

EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2)"
CMD ["uvicorn", "--factory", "vizwiz.serve.gateway:create_app", "--host", "0.0.0.0", "--port", "8080"]
