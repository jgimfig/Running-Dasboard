FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/data \
    CACHE_DIR=/cache

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py data_loader.py ./
COPY assets/ assets/

# Run as an unprivileged user; /cache holds the Apple Health parse cache
RUN useradd --create-home --uid 1000 runner \
    && mkdir -p /data /cache \
    && chown runner:runner /cache
USER runner

EXPOSE 8050

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8050/', timeout=4)" || exit 1

# --preload loads the data once before forking, so slow first parses don't hit the worker timeout.
# A single worker keeps one in-memory copy of the data; threads handle concurrent callbacks.
CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:8050", "--preload", \
     "--workers", "1", "--threads", "4", "--timeout", "120"]
