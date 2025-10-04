FROM python:3.13-slim

# Update system packages and pip to the latest version for security
RUN apt-get update && apt-get upgrade -y && pip install --upgrade pip

# Устанавливаем рабочую директорию
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Устанавливаем переменную окружения
ENV PYTHONPATH=/app

# Запускаем приложение
ENTRYPOINT ["python", "main.py"]