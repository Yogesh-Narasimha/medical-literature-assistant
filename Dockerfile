# Image for the API and the Streamlit UI. Ollama runs as a separate service (see docker-compose.yml).
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY eval ./eval
COPY app.py .

# data/ (downloaded labels, chunks, FAISS index) is mounted as a volume, not baked into the image.
ENV OLLAMA_HOST=http://ollama:11434
EXPOSE 8000 8501
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
