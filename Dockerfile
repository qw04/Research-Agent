FROM python:3.11-slim

WORKDIR /app

COPY requirements.lock.txt .
RUN pip install --no-cache-dir -r requirements.lock.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Bake the embedding model into the image so the always-on pod never depends
# on Hugging Face Hub being reachable at an arbitrary restart (node reboot,
# OOM-kill recovery). HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE (set at deploy time)
# then guarantee no code path can silently fall back to a network call.
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY src/ src/
COPY data/ data/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
