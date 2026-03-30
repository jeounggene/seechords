FROM python:3.12-slim

# System deps: ffmpeg for audio conversion, libsndfile for librosa
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || \
    (echo "WARNING: full install failed, retrying without essentia" && \
     grep -v essentia requirements.txt | pip install --no-cache-dir -r /dev/stdin)

# Copy server code
COPY . .

# Create persistent-data parent (Fly volume will mount at /data)
RUN mkdir -p /data/uploads

EXPOSE 8080

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--timeout", "600", "--workers", "1"]
