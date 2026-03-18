# Swing Trade — FastAPI Backend (Production)
# Deploy to Fly.io, Railway, or any Docker host

FROM python:3.11-slim

# System deps for psycopg2
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code (.dockerignore excludes frontend, venv, etc.)
COPY . .

# Data dir for SQLite fallback (if no DATABASE_URL)
RUN mkdir -p /app/data

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
