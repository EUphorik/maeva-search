import json
import os
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import streamlit as st
from dotenv import load_dotenv

from utils.search_engine import init_client


@st.cache_data
def load_dataset() -> List[Dict]:
    """Charge le dataset de séjours depuis le fichier JSON."""
    return json.loads(Path("data/sejours.json").read_text())


def render_sidebar() -> Tuple[Optional[str], int]:
    """Affiche et gère toute la logique de la barre latérale.

    Retourne:
        api_key: clé API OpenAI saisie par l'utilisateur (ou None si vide)
        top_k: nombre de résultats demandés
    """
    with st.sidebar:
        # Chargement de la clé API depuis le fichier .env (ou les variables d'env)
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")

        st.markdown("### ⚙️ Configuration")
        st.markdown("---")

        if api_key:
            init_client(api_key)
            st.success("✅ Clé API OpenAI chargée avec succès !")
        else:
            st.error("⚠️ Clé API OpenAI introuvable. Ajoutez OPENAI_API_KEY dans votre fichier .env")


        st.markdown("---")
        top_k = st.slider("Nombre de résultats", 20, 200, 50, 10)
        st.markdown("---")
        st.markdown(f"""
        <div style="font-size:0.78rem; color:#9ca3af; line-height:1.7;">
        <strong>Modèles utilisés</strong><br>
        🧠 Intention : <code>{os.getenv("LLM_MODEL")}</code><br>
        🔢 Embeddings : <code>{os.getenv("EMBEDDING_MODEL")}({os.getenv("EMBEDDING_DIM")})</code><br>
        </div>
        """, unsafe_allow_html=True)

    return api_key or None, top_k
