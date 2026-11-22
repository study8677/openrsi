FROM python:3.12-slim

RUN pip install --no-cache-dir pytest==8.4.1

WORKDIR /app

COPY kvstore.py /app/kvstore.py
COPY test_smoke.py /app/test_smoke.py
