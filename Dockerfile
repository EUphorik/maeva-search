FROM python:3.11-slim

WORKDIR /app

# Installation des dépendances système (nécessaires pour certaines bibliothèques comme faiss)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copie des requirements en premier pour profiter du cache Docker
COPY requirements.txt .

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie du reste du code source
COPY . .

# Port exposé par Streamlit
EXPOSE 8501

# Healthcheck pour s'assurer que Streamlit tourne bien
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Commande de démarrage par défaut
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

ENV PYTHONUNBUFFERED=1
