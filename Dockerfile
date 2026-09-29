FROM python:3.11-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

COPY showcase ./showcase
COPY models ./models

ENV PATH="/app/.venv/bin:$PATH" \
    DJANGO_DEBUG=0 \
    DJANGO_ALLOWED_HOSTS=* \
    PYTHONUNBUFFERED=1

EXPOSE 8000
WORKDIR /app/showcase
CMD ["gunicorn", "config.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "180"]
