FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
WORKDIR /app/code

ENV DB_PATH=/home/student_scoring.db

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--timeout", "300", "app:app"]