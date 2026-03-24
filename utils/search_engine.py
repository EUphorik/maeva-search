import html
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import Client as LangSmithClient
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from dotenv import load_dotenv
from opensearchpy import OpenSearch
from utils.format import build_residence_photo

# Charger le .env AVANT de lire les variables d'environnement
load_dotenv()

def clean_html(raw_html: str) -> str:
    """Nettoyage du HTML pour la description (inclut le décodage des entités)."""
    if not isinstance(raw_html, str):
        return ""
    # On supprime les balises HTML
    clean = re.sub('<[^<]+?>', ' ', raw_html)
    # On décode les entités HTML (&nbsp; etc.)
    clean = html.unescape(clean)
    # On normalise les espaces
    clean = re.sub(r'\s+', ' ', clean)
    return clean.strip()

# ── Configuration ────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
INTENT_MODEL    = os.getenv("LLM_MODEL")
EMBEDDING_DIM   = os.getenv("EMBEDDING_DIM")

# Clients LangChain / Qdrant (initialisés paresseusement)
_llm: ChatOpenAI | None = None
_embeddings: OpenAIEmbeddings | None = None
_opensearch_client: OpenSearch | None = None


class FiltersOutput(BaseModel):
    """Structure typée des filtres renvoyés par le LLM.

    On modélise explicitement chaque champ pour générer un JSON Schema
    compatible avec l'API OpenAI (évite les erreurs de response_format).
    """

    zone_geo_label: Optional[str] = Field(
        None,
        description="Lieu extrait de la requête : ville, région ou POI. Ex: 'Arcachon', 'Bretagne', 'Disneyland Paris'"
    )
    is_poi: bool = Field(
        False,
        description="True uniquement si le lieu est un point d'intérêt précis et délimité (monument, parc, lac nommé). False pour une ville ou une région."
    )
    pays: Optional[str] = Field(None, description="Pays du séjour si mentionné ou clairement déduit. Ex: 'France'")
    type_logement: Optional[str] = Field(None, description="Type de logement demandé. Ex: 'villa', 'appartement', 'chalet'")
    nb_personnes: Optional[int] = Field(None, description="Nombre de personnes pour le séjour")
    prix_min: Optional[float] = Field(None, description="Prix minimum par nuit en euros si mentionné")
    prix_max: Optional[float] = Field(None, description="Prix maximum par nuit en euros si mentionné")
    date_debut: Optional[str] = Field(None, description="Date d'arrivée au format YYYY-MM-DD")
    date_fin: Optional[str] = Field(None, description="Date de départ au format YYYY-MM-DD")
    ambiance: Optional[list[str]] = Field(None, description="Ambiances souhaitées. Ex: ['Famille', 'Amis', 'Couple']")
    activites: Optional[list[str]] = Field(None, description="Activités souhaitées. Ex: ['Surf', 'Randonnée', 'Ski']")
    activity_inferred: bool = False
    villes_expanded: List[str] = []


class IntentOutput(BaseModel):
    semantic_query: str
    filters: FiltersOutput
    intent_summary: str

class GeoExpansionOutput(BaseModel):
    """Structure des résultats d'expansion géographique."""
    villes: List[str] = []
    departements: List[str] = []

def init_client(api_key: str) -> None:
    """Initialise les clients LangChain et active LangSmith si configuré.

    - Force la clé OpenAI dans l'env pour les libs LangChain
    - Active le tracing LangSmith si LANGSMITH_API_KEY est présent
    """
    global _llm, _embeddings

    # Charge les variables depuis .env si ce n'est pas déjà fait
    load_dotenv()

    # Clé OpenAI pour LangChain / langchain-openai (priorité au paramètre)
    os.environ["OPENAI_API_KEY"] = api_key

    # LangSmith / LangChain tracing v2 : entièrement piloté par .env
    # Si LANGSMITH_API_KEY est présent, LangChain enverra les traces
    # en fonction de LANGCHAIN_TRACING_V2 / LANGCHAIN_ENDPOINT / LANGCHAIN_PROJECT

    # Clients paresseux
    if _llm is None:
        _llm = ChatOpenAI(model=INTENT_MODEL, temperature=0)
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL,dimensions=EMBEDDING_DIM)


def _ensure_clients() -> None:
    """S'assure que les clients LangChain sont initialisés.

    Utile si d'autres scripts utilisent ce module sans passer par init_client.
    """
    global _llm, _embeddings
    if _llm is None:
        _llm = ChatOpenAI(model=INTENT_MODEL, temperature=0)
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL,dimensions=EMBEDDING_DIM)




def _get_opensearch_client() -> OpenSearch:
    global _opensearch_client

    if _opensearch_client is not None:
        return _opensearch_client

    host = os.getenv("OPENSEARCH_HOST") or os.getenv("OPENSEARCH_URL")
    user = os.getenv("OPENSEARCH_USERNAME")
    password = os.getenv("OPENSEARCH_PASSWORD")

    if not host:
        raise RuntimeError("La variable OPENSEARCH_HOST (ou OPENSEARCH_URL) doit être définie pour utiliser OpenSearch")

    use_ssl = host.startswith("https")
    
    conn_params = {
        "hosts": [host],
        "use_ssl": use_ssl,
        "verify_certs": False,
        "ssl_show_warn": False
    }
    
    if user and password:
        conn_params["http_auth"] = (user, password)

    _opensearch_client = OpenSearch(**conn_params)
    return _opensearch_client

# ── Construction du texte à vectoriser ───────────────────────────────────────
def build_embedding_text(sejour: dict) -> str:
    """
    Construit le texte envoyé à OpenAI pour créer l'embedding.

    Règles :
    - On n'inclut un champ que s'il est renseigné (pas de "Activités : ." vide)
    - On utilise tous les champs de localisation disponibles (ville, région, dept)
    - Les champs numériques utiles (surface, sdb) enrichissent la recherche
      pour des requêtes comme "grand appartement" ou "avec 2 salles de bain"
    """
    parts = []

    # ── Titre + description (signal le plus fort) ────────────────────────────
    titre = (sejour.get("titre") or "").strip()
    if titre:
        parts.append(titre + ".")

    description = (sejour.get("description") or "").strip()
    if description:
        parts.append(description)

    # ── Localisation (du plus précis au plus large) ──────────────────────────
    loc_parts = []
    for field in ["ville", "departement", "region", "pays"]:
        val = (sejour.get(field) or "").strip()
        if val:
            loc_parts.append(val)
    if loc_parts:
        parts.append("Destination : " + ", ".join(loc_parts) + ".")

    # ── Type de logement ─────────────────────────────────────────────────────
    type_log = (sejour.get("type_logement") or "").strip()
    if type_log:
        parts.append(f"Type de logement : {type_log}.")

    # ── Capacité ─────────────────────────────────────────────────────────────
    cap_parts = []
    nb_pers = sejour.get("nb_personnes")
    if nb_pers:
        cap_parts.append(f"{nb_pers} personnes")
    nb_ch = sejour.get("nb_chambres")
    if nb_ch:
        cap_parts.append(f"{nb_ch} chambre{'s' if int(nb_ch) > 1 else ''}")
    nb_sdb = sejour.get("nb_sdb")
    if nb_sdb:
        cap_parts.append(f"{nb_sdb} salle{'s' if int(nb_sdb) > 1 else ''} de bain")
    surface = sejour.get("surface_m2")
    if surface:
        cap_parts.append(f"{surface} m²")
    if cap_parts:
        parts.append("Capacité : " + ", ".join(cap_parts) + ".")

    # ── Équipements (seulement si liste non vide) ────────────────────────────
    equipements = [e for e in (sejour.get("equipements") or []) if e]
    if equipements:
        parts.append("Équipements : " + ", ".join(equipements) + ".")

    # ── Activités (seulement si liste non vide) ──────────────────────────────
    activites = [a for a in (sejour.get("activites") or []) if a]
    if activites:
        parts.append("Activités à proximité : " + ", ".join(activites) + ".")

    # ── Tags / ambiance (seulement si liste non vide) ────────────────────────
    tags = [t for t in (sejour.get("tags") or []) if t]
    if tags:
        parts.append("Ambiance : " + ", ".join(tags) + ".")

    return " ".join(parts)


# ── Extraction d'intention ────────────────────────────────────────────────────


def extract_intent(query: str) -> dict:
    """
    Utilise le LLM pour extraire l'intention structurée d'une requête.
    Retourne un dict avec semantic_query, filters et intent_summary.
    """
    _ensure_clients()

    try:
        _t_intent_start = time.perf_counter()
        
        client = LangSmithClient()
        prompt = client.pull_prompt("search_intent_v0:866ff27a")
        
        # Date du jour pour aider à l'extraction
        current_date = datetime.now().strftime("%Y-%m-%d")

        llm = ChatOpenAI(
            model=INTENT_MODEL,
            temperature=0,
            reasoning_effort="low",
        ).with_structured_output(IntentOutput)
        
        llm_chain = prompt | llm
        response: IntentOutput = llm_chain.invoke(input={
            "question": query, 
            "current_date": current_date
        })
        
        print(f"⏱️  Intent Extraction : {time.perf_counter() - _t_intent_start:.2f}s")
        return response.model_dump()
    except Exception as e:
        raise RuntimeError(f"Erreur lors de l'extraction d'intention : {e}") from e

@lru_cache(maxsize=256)
def expand_geo_zone(zone_label: str) -> dict:
    """Expansion géographique structurée à partir d'un libellé de zone.

    Résultat mis en cache (LRU 256 entrées) — même zone = 0 appel LLM supplémentaire.
    Retourne un dictionnaire avec 'villes' et 'departements'.
    """
    print(f"Expansion géographique pour : {zone_label}")
    if not zone_label:
        return {"villes": [], "departements": []}

    _ensure_clients()

    try:
        client = LangSmithClient()
        prompt = client.pull_prompt("geo_expension_v0")

        llm = ChatOpenAI(
            model=INTENT_MODEL,
            temperature=0,
            reasoning_effort='low'
        ).with_structured_output(GeoExpansionOutput)
        
        chain = prompt | llm
        response: GeoExpansionOutput = chain.invoke(input={"zone_label": zone_label})
        result = response.model_dump()
        
        # Post-traitement : suppression des tirets dans les noms de départements
        result["departements"] = [d.replace("-", " ") for d in result.get("departements", [])]
        return result
    except Exception as e:
        print(f"Erreur expansion géo : {e}")
        return {"villes": [], "departements": []}


# ── Appels OpenAI Embeddings ──────────────────────────────────────────────────
def get_embedding(text: str) -> list[float]:
    _ensure_clients()
    cleaned = text.replace("\n", " ")
    return _embeddings.embed_query(cleaned)  # type: ignore[union-attr]


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Calcule les embeddings en batch en respectant la limite de tokens.

    Utilise OpenAIEmbeddings de LangChain et découpe en batches pour
    éviter de dépasser les limites de tokens par requête.
    """
    _ensure_clients()

    all_embeddings: list[list[float]] = []
    batch_size = 100

    for start in range(0, len(texts), batch_size):
        batch_texts = [t.replace("\n", " ") for t in texts[start:start + batch_size]]
        batch_embeddings = _embeddings.embed_documents(batch_texts)  # type: ignore[union-attr]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


# ── Normalisation ────────────────────────────────────────────────────────────
def _normalize(s: str) -> str:
    """Normalisation souple pour les comparaisons de chaînes.

    - Minuscule
    - Suppression des accents
    - Suppression des caractères non alphanumériques (espaces, tirets...)
    """
    s = s.lower()
    for a, b in [("é","e"),("è","e"),("ê","e"),("à","a"),("â","a"),
                 ("ô","o"),("û","u"),("ù","u"),("î","i"),("ï","i"),("ç","c")]:
        s = s.replace(a, b)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


# ── Recherche sémantique + filtres extraits ───────────────────────────────────
def _apply_hard_filters(
    sejour: dict,
    filters: dict
) -> bool:
    """Applique tous les filtres durs hors géographie (déjà gérée en amont)."""
    nb_pers = filters.get("nb_personnes")
    if nb_pers and int(sejour.get("nb_personnes") or 0) < int(nb_pers):
        return False

    surface_min = filters.get("surface_min")
    if surface_min and int(sejour.get("surface_m2") or 0) < int(surface_min):
        return False

    # Note: On laisse prix_min/max en post-filtrage car certains champs 
    # peuvent manquer dans l'index ou avoir des formats inattendus.
    # Toutefois, la recherche hybride fera le gros du travail.
    prix_min = filters.get("prix_min")
    prix_max = filters.get("prix_max")
    if prix_min is not None or prix_max is not None:
        try:
            prix_nuit = float(sejour.get("prix_nuit") or 0)
            if prix_min is not None and prix_nuit < float(prix_min):
                return False
            if prix_max is not None and prix_nuit > float(prix_max):
                return False
        except (TypeError, ValueError):
            pass

    # type_log = filters.get("type_logement")
    # if type_log and not _str_match(sejour.get("type_logement") or "", [type_log]):
    #     return False

    return True




def semantic_search_with_intent(
    intent: dict,
    top_k: int = 5,
    index_name: str | None = None,
    metadata: list[dict] | None = None,
) -> list[dict]:
    _t0 = time.perf_counter()

    filters = intent.get("filters") or {}
    semantic_query = intent.get("semantic_query") or ""

    if not semantic_query.strip():
        return []

    client = _get_opensearch_client()
    index_target = index_name or os.getenv("OPENSEARCH_INDEX")

    zone_label = filters.get("zone_geo_label")
    pays_filter = filters.get("pays")
    villes_filter: list[str] = []
    depts_filter: list[str] = []

    # ── Parallélisation : expand_geo_zone + get_embedding lancés simultanément ──
    # Ces deux appels sont indépendants → on les exécute en parallèle avec
    # ThreadPoolExecutor pour économiser ~5-10s sur le temps total.
    clean_zone_label = zone_label.lstrip("~").strip() if zone_label else None

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_geo = executor.submit(expand_geo_zone, clean_zone_label) if clean_zone_label else None
        future_vec = executor.submit(get_embedding, semantic_query)

        # Récupération embedding
        _t_vec_start = time.perf_counter()
        query_vec = future_vec.result()
        print(f"⏱️  Embedding        : {time.perf_counter() - _t_vec_start:.2f}s")

        # Récupération expansion géo (0s si cache LRU hit)
        if future_geo:
            _t_geo_start = time.perf_counter()
            geo_data = future_geo.result()
            _geo_elapsed = time.perf_counter() - _t_geo_start
            villes_filter = list(geo_data.get("villes", []))
            depts_filter = list(geo_data.get("departements", []))
            if villes_filter:
                intent.setdefault("filters", {})["villes_expanded"] = villes_filter
            cached = expand_geo_zone.cache_info().hits > 0
            print(f"⏱️  Geo-expansion    : {_geo_elapsed:.2f}s {'(cache ✅)' if cached else '(LLM call)'}")
            print(f"🗺️  villes={villes_filter}, depts={depts_filter}")

    print(f"⏱️  [Parallèle total]: {time.perf_counter() - _t0:.2f}s")
    
    # ── Étape 1 : Hybride sur l'index sémantique ──
    semantic_index = os.getenv("OPENSEARCH_SEMENTIC_INDEX")
    if not semantic_index:
        raise RuntimeError("La variable OPENSEARCH_SEMENTIC_INDEX n'est pas définie dans l'environnement")

    must_filters_step1 = []
    
    # Filtre Géographique (Ville OU Département)
    geo_should = []
    for ville in villes_filter:
        geo_should.append({"match": {"city": {"query": ville, "operator": "and"}}})
        
    if not filters.get("is_poi", False):
        for dept in depts_filter:
            geo_should.append({"match": {"residence_department.fr": {"query": dept, "operator": "and"}}})

    if geo_should:
        must_filters_step1.append({
            "bool": {
                "should": geo_should,
                "minimum_should_match": 1
            }
        })
    
    if pays_filter:
        must_filters_step1.append({"match": {"country": pays_filter}})

    # Construction de la requête KNN hybride pour l'index sémantique
    knn_clause = {
        "my_vector": {
            "vector": query_vec,
            "k": 100 # On prend plus de résultats pour avoir de la marge sur les filtres durs
        }
    }

    if must_filters_step1:
        knn_clause["my_vector"]["filter"] = {
            "bool": {
                "must": must_filters_step1
            }
        }

    body_step1 = {
        "size": 100,
        "_source": ["product_id"],
        "query": {
            "knn": knn_clause
        }
    }
    
    response_step1 = client.search(index=semantic_index, body=body_step1)
    hits_step1 = response_step1["hits"]["hits"]
    
    product_ids = []
    product_scores = {}
    for h in hits_step1:
        pid = h.get("_source", {}).get("product_id") or h.get("_id")
        if pid:
            product_ids.append(pid)
            product_scores[str(pid)] = h.get("_score", 0.0)

    print(f"Étape 1 (Index sémantique) a retourné {len(product_ids)} IDs")

    if not product_ids:
        return []

    # ── Étape 2 : Recherche sur l'index global avec filtres durs ──
    must_filters_step2 = [
        {
            "bool": {
                "should": [
                    {"terms": {"product_id": product_ids}},
                    {"terms": {"product_id.keyword": product_ids}},
                    {"terms": {"_id": product_ids}}
                ],
                "minimum_should_match": 1
            }
        }
    ]

    # Filtre de prix (directement dans OpenSearch sur l'index global)
    prix_min = filters.get("prix_min")
    prix_max = filters.get("prix_max")
    if prix_min is not None or prix_max is not None:
        price_range = {}
        if prix_min is not None: price_range["gte"] = prix_min
        if prix_max is not None: price_range["lte"] = prix_max
        must_filters_step2.append({"range": {"min_price_per_night": price_range}})

    # On peut aussi ajouter d'autres filtres durs ici si nécessaire,
    # mais le post-filtrage Python s'en chargera via `_apply_hard_filters`

    body_step2 = {
        "size": top_k * 5, # Assez pour le filtrage python
        "query": {
            "bool": {
                "must": must_filters_step2
            }
        }
    }

    response_step2 = client.search(index=index_target, body=body_step2)
    hits_step2 = response_step2["hits"]["hits"]

    # Remettre les résultats de l'étape 2 dans l'ordre des scores de l'étape 1
    hits_step2.sort(
        key=lambda h: product_scores.get(str(h.get("_source", {}).get("product_id") or h.get("_id")), 0.0),
        reverse=True
    )

    print(f"Étape 2 (Index global) a retourné {len(hits_step2)} hits avant filtrage dur")
    results: list[dict] = []
    
    for hit in hits_step2:
        source = hit["_source"]
        
        # Extraction et nettoyage de la description
        desc_raw = source.get("description", {})
        if isinstance(desc_raw, dict):
            description = desc_raw.get("fr", "")
        else:
            description = str(desc_raw)
        description = clean_html(description)

        # Extraction detail logement
        accommodation = source.get("accommodation", {})
        nb_chambres = accommodation.get("bedrooms_count")
        nb_salles_de_bain = accommodation.get("bathrooms_count")
        surface = accommodation.get("surface_m2")

        main_image = source.get("main_image_url", "")

        pid = source.get("product_id") or hit["_id"]
        original_score = product_scores.get(str(pid), hit["_score"])

        # Mapping vers le format attendu par app.py 
        sejour = {
            "id": source.get("product_id"),
            "titre": source.get("label", "Sans titre"),
            "description": description,
            "ville": source.get("city") or source.get("station_ville") or "N/A",
            "region": source.get("residence_region_label", {}).get("fr") or source.get("region") or "",
            "type_logement": source.get("type_logement") or "Location",
            "nb_personnes": source.get("capacity", {}).get("total") if isinstance(source.get("capacity"), dict) else source.get("capacity", 0),
            "prix_nuit": source.get("min_price_per_night") or source.get("prix_nuit") or 0.0,
            "score": original_score * 100 if original_score <= 1.0 else original_score, # Utiliser le score de l'étape 1
            "equipements": [f.get("label") for f in source.get("residence_facilities", []) if isinstance(f, dict) and f.get("label")],
            "tags": source.get("tags") or [],
            "chambres": nb_chambres,
            "salles_de_bain": nb_salles_de_bain,
            "surface": surface,
            "image": build_residence_photo(main_image)
        }

        if not _apply_hard_filters(sejour, filters):
            continue

        results.append(sejour)
        
        if len(results) >= top_k:
            break

    print(f"OpenSearch a retourné {len(results)} résultats après filtrage dur complet")
    return results

def semantic_search_with_intent_v2(
    intent: dict,
    top_k: int = 5,
    index_name: str | None = None,
    metadata: list[dict] | None = None,
) -> list[dict]:
    _t0 = time.perf_counter()

    filters = intent.get("filters") or {}
    semantic_query = intent.get("semantic_query") or ""

    if not semantic_query.strip():
        return []

    client = _get_opensearch_client()
    index_target = index_name or os.getenv("OPENSEARCH_INDEX")
    semantic_index = os.getenv("OPENSEARCH_SEMENTIC_INDEX")
    if not semantic_index:
        raise RuntimeError("OPENSEARCH_SEMENTIC_INDEX n'est pas définie")

    zone_label = filters.get("zone_geo_label")
    pays_filter = filters.get("pays")
    villes_filter: list[str] = []
    depts_filter: list[str] = []

    clean_zone_label = zone_label.lstrip("~").strip() if zone_label else None

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_geo = executor.submit(expand_geo_zone, clean_zone_label) if clean_zone_label else None
        future_vec = executor.submit(get_embedding, semantic_query)

        _t_vec_start = time.perf_counter()
        query_vec = future_vec.result()
        print(f"⏱️  Embedding        : {time.perf_counter() - _t_vec_start:.2f}s")

        if future_geo:
            _t_geo_start = time.perf_counter()
            geo_data = future_geo.result()
            villes_filter = list(geo_data.get("villes", []))
            depts_filter = list(geo_data.get("departements", []))
            if villes_filter:
                intent.setdefault("filters", {})["villes_expanded"] = villes_filter
            cached = expand_geo_zone.cache_info().hits > 0
            print(f"⏱️  Geo-expansion    : {time.perf_counter() - _t_geo_start:.2f}s {'(cache ✅)' if cached else '(LLM call)'}")
            print(f"🗺️  villes={villes_filter}, depts={depts_filter}")

    print(f"⏱️  [Parallèle total]: {time.perf_counter() - _t0:.2f}s")

    # ── Étape 1 : Hybrid search sur l'index sémantique ──────────────────────────

    must_filters_step1 = []

    # FIX 1 : term/terms sur keyword, pas match
    geo_should = []
    for ville in villes_filter:
        geo_should.append({"term": {"city": ville.replace("-", " ").upper()}})

    # if not filters.get("is_poi", False):
    #     for dept in depts_filter:
    #         geo_should.append({"term": {"department": dept}})

    if geo_should:
        must_filters_step1.append({
            "bool": {"should": geo_should, "minimum_should_match": 1}
        })

    # FIX 2 : term sur keyword pour pays
    if pays_filter:
        must_filters_step1.append({"term": {"country": pays_filter}})


    # FIX 3 : Hybrid search BM25 + KNN avec le pipeline RRF
    knn_query = {
        "knn": {
            "my_vector": {
                "vector": query_vec,
                "k": 100,
                **({"filter": {"bool": {"must": must_filters_step1}}} if must_filters_step1 else {})
            }
        }
    }

    bm25_query = {
        "bool": {
            "must": [{"match": {"contenu_texte": {"query": semantic_query, "analyzer": "french"}}}],
            **({"filter": must_filters_step1} if must_filters_step1 else {})
        }
    }

    body_step1 = {
        # "size": 100,
        "_source": ["product_id"],
        "query": {
            "hybrid": {
                "queries": [bm25_query, knn_query]
            }
        }
    }

    print(f"body_step1: {body_step1}")
    response_step1 = client.search(index=semantic_index,
                                   body=body_step1,
                                   params={"search_pipeline": "hybrid_pipeline"})
    hits_step1 = response_step1["hits"]["hits"]

    product_ids = []
    product_scores = {}
    for h in hits_step1:
        pid = str(h.get("_source", {}).get("product_id") or h.get("_id"))
        if pid:
            product_ids.append(pid)
            product_scores[pid] = h.get("_score", 0.0)

    print(f"Étape 1 (Index sémantique) → {len(product_ids)} IDs")

    if not product_ids:
        return []

    # ── Étape 2 : Filtres durs sur l'index global ────────────────────────────────

    # FIX 4 : terms simple sur keyword — pas de triple should défensif
    must_filters_step2 = [
        {"terms": {"product_id": product_ids}}
    ]

    # Prix
    prix_min = filters.get("prix_min")
    prix_max = filters.get("prix_max")
    if prix_min is not None or prix_max is not None:
        price_range = {}
        if prix_min is not None: price_range["gte"] = prix_min
        if prix_max is not None: price_range["lte"] = prix_max
        must_filters_step2.append({"range": {"min_price_per_night": price_range}})

    # Capacité
    nb_personnes = filters.get("nb_personnes")
    if nb_personnes:
        must_filters_step2.append({"range": {"capacity.total": {"gte": nb_personnes}}})

    # FIX 5 : filtres durs dans OpenSearch, pas en Python
    nb_chambres_min = filters.get("nb_chambres")
    if nb_chambres_min:
        must_filters_step2.append({"range": {"accommodation.bedrooms_count": {"gte": nb_chambres_min}}})

    # Filtrage par dates et disponibilités
    date_debut = filters.get("date_debut")
    date_fin = filters.get("date_fin")
    
    # On privilégie la durée explicite si extraite, sinon on la calcule
    duree = filters.get("duree")
    if date_debut and date_fin and not duree:
        try:
            d1 = datetime.strptime(str(date_debut), "%Y-%m-%d")
            d2 = datetime.strptime(str(date_fin), "%Y-%m-%d")
            duree = (d2 - d1).days
        except (ValueError, TypeError):
            pass

    if date_debut or duree:
        nested_must = []
        if date_debut:
            nested_must.append({"term": {"availabilities.start_date": date_debut}})
        if duree:
            nested_must.append({"term": {"availabilities.duration": int(duree)}})
            
        must_filters_step2.append({
            "nested": {
                "path": "availabilities",
                "query": {
                    "bool": {
                        "must": nested_must
                    }
                }
            }
        })
    elif duree:
        # Si on n'a que la durée mais pas de date précise, on peut chercher dans un champ à plat
        must_filters_step2.append({"term": {"available_durations": int(duree)}})

    body_step2 = {
        "size": top_k,  # plus besoin de x5 — les filtres sont dans OpenSearch
        "query": {
            "bool": {"must": must_filters_step2}
        }
    }

    response_step2 = client.search(index=index_target, body=body_step2)
    hits_step2 = response_step2["hits"]["hits"]

    # Réappliquer l'ordre sémantique de l'étape 1
    hits_step2.sort(
        key=lambda h: product_scores.get(
            str(h.get("_source", {}).get("product_id") or h.get("_id")), 0.0
        ),
        reverse=True
    )

    print(f"Étape 2 (Index global) → {len(hits_step2)} hits")

    results = []
    for hit in hits_step2:
        source = hit["_source"]

        desc_raw = source.get("description", {})
        description = clean_html(
            desc_raw.get("fr", "") if isinstance(desc_raw, dict) else str(desc_raw)
        )

        accommodation = source.get("accommodation", {})
        pid = str(source.get("product_id") or hit["_id"])

        results.append({
            "id":           source.get("product_id"),
            "titre":        source.get("label", "Sans titre"),
            "description":  description,
            "ville":        source.get("city") or "N/A",
            "region":       source.get("residence_region_label", {}).get("fr") or "",
            "type_logement": source.get("type_logement") or "Location",
            "nb_personnes": source.get("capacity", {}).get("total") if isinstance(source.get("capacity"), dict) else source.get("capacity", 0),
            "prix_nuit":    source.get("min_price_per_night") or 0.0,
            "score":        product_scores.get(pid, 0.0) * 100,
            "equipements":  [f.get("label") for f in source.get("residence_facilities", []) if isinstance(f, dict) and f.get("label")],
            "tags":         source.get("tags") or [],
            "chambres":     accommodation.get("bedrooms_count"),
            "salles_de_bain": accommodation.get("bathrooms_count"),
            "surface":      accommodation.get("surface_m2"),
            "image":        build_residence_photo(source.get("main_image_url", ""))
        })

    print(f"→ {len(results)} résultats finaux")
    return results


# def semantic_search_with_intent(
#     intent: dict,
#     metadata: list[dict] | None,
#     top_k: int = 5,
#     collection_name: str | None = None,
# ) -> list[dict]:
#     """Point d'entrée unifié pour la recherche sémantique avec OpenSearch.

#     Utilise l'index OpenSearch défini par OPENSEARCH_INDEX ou `collection_name`.
#     """
#     return _semantic_search_with_intent_opensearch(
#         intent=intent,
#         top_k=top_k,
#         index_name=collection_name,
#         metadata=metadata,
#     )
