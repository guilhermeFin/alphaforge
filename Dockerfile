FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY research ./research
COPY api ./api
COPY app ./app
COPY accounts ./accounts
RUN pip install --no-cache-dir ".[api,app,llm,data]"

EXPOSE 8000 8501
