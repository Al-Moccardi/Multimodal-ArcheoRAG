# ============================================
# Pompeii Multimodal Framework — Dockerfile
# ============================================
# Note: Ollama must run on the host or as a separate container.
# Set OLLAMA_HOST=http://host.docker.internal:11434 for Docker Desktop
# or OLLAMA_HOST=http://ollama:11434 with docker-compose.

FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create directories
RUN mkdir -p \
    knowledge_base/ceramics/historical \
    knowledge_base/ceramics/cataloguing \
    knowledge_base/paintings/historical \
    knowledge_base/paintings/cataloguing \
    knowledge_base/architecture/historical \
    knowledge_base/architecture/cataloguing \
    vector_stores \
    outputs/metadata \
    outputs/annotations \
    outputs/images

EXPOSE 7860

CMD ["python", "app.py"]
