# Production image: Flask server + BTC chord model + Transformer+CRF freeze5 fallback.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || \
    (echo "WARNING: full install failed, retrying without essentia" && \
     grep -v essentia requirements.txt | pip install --no-cache-dir -r /dev/stdin)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir https://github.com/CPJKU/beat_this/archive/main.zip

COPY server/ /app/
COPY training/v2 /app/training/v2
COPY training/shared /app/training/shared
COPY training/models/chord_transformer_crf_freeze5.pt /app/training/models/chord_transformer_crf_freeze5.pt

ENV FLASK_ENV=production
ENV PYTHONPATH=/app/training
ENV SEECHORDS_MODEL_DIR=/app/training/models
ENV USE_BTC=1
ENV USE_FREEZE5=1
ENV USE_BEAT_THIS=1

# Download BTC checkpoint at build time (MIT-licensed, from ptnghia-j/ChordMini)
RUN python -c "\
import urllib.request, os; \
os.makedirs('/app/btc_model', exist_ok=True); \
url = 'https://github.com/ptnghia-j/ChordMini/raw/main/checkpoints/btc_model_best.pth'; \
urllib.request.urlretrieve(url, '/app/btc_model/btc_model_best.pth'); \
print('Downloaded BTC checkpoint')"

# Pre-cache Beat This! small0 checkpoint (avoids cold-start download)
RUN python -c "\
import torch; \
torch.hub.load_state_dict_from_url( \
    'https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/small0.ckpt', \
    file_name='beat_this-small0.ckpt', map_location='cpu'); \
print('Cached beat_this small0 checkpoint')"

RUN mkdir -p /data/uploads

EXPOSE 8080

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--timeout", "600", "--workers", "1"]
