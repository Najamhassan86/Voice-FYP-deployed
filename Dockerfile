FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY voice-backend/requirements.txt ./voice-backend/requirements.txt
RUN pip install --no-cache-dir -r voice-backend/requirements.txt

COPY . .

WORKDIR /app/voice-backend

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
