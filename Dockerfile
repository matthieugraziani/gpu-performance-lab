# ── Base image ────────────────────────────────────────────────────────────────
# PyTorch avec support CUDA (compatible RTX 40xx / Ada Lovelace)
FROM pytorch/pytorch:2.2.0-cuda12.9-cudnn9.3-runtime

# ── Metadata ──────────────────────────────────────────────────────────────────
LABEL maintainer="benchmark"
LABEL description="CPU & GPU Benchmark + Streamlit Dashboard"

# ── Dépendances système ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Répertoire de travail ─────────────────────────────────────────────────────
WORKDIR /app

# ── Dépendances Python ────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copie du projet ───────────────────────────────────────────────────────────
COPY notebooks/ ./notebooks/
COPY app.py .

# ── Création des dossiers de sortie ──────────────────────────────────────────
RUN mkdir -p data assets

# ── Port Streamlit ────────────────────────────────────────────────────────────
EXPOSE 8501

# ── Healthcheck ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Commande par défaut : dashboard ──────────────────────────────────────────
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
