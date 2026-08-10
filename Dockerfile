FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

RUN python -m pip install --no-cache-dir '.[minio]'

EXPOSE 8000

CMD ["uvicorn", "vehicle_intelligence.interfaces.api:app", "--host", "0.0.0.0", "--port", "8000"]
