# 🏖️ POC — Moteur de recherche sémantique de séjours

Prototype fonctionnel d'un moteur de recherche sémantique de séjours de vacances.  

---

## 🚀 Installation (Alternative : Environnement local)

```bash
# 1. Cloner / dézipper le projet
cd semantic-search

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Windows : venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
streamlit run app.py
```

---

## 🐳 Installation via Docker (Recommandé)

Le projet dispose d'une configuration `docker-compose` prête à l'emploi. L'application sera accessible sur le port **8502**.

```bash
# 1. Cloner / dézipper le projet
cd semantic-search

# 2. Définir vos variables d'environnement (si un .env est requis, ex: pour l'API Key OpenSearch)
# N'oubliez pas de configurer vos clés dans le fichier .env avant le lancement

# 3. Construire et démarrer les conteneurs
docker-compose up -d --build

# 4. Accéder à l'application
# Ouvrez votre navigateur sur http://localhost:8502
```

Pour consulter les logs du conteneur en direct :
```bash
docker-compose logs -f app
```

Pour arrêter complètement l'application :
```bash
docker-compose down
```

