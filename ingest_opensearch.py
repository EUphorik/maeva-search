import argparse
import html
import os
import re

from dotenv import load_dotenv
from opensearchpy import OpenSearch, helpers

from utils.search_engine import (
    get_embeddings_batch,
    init_client,
)

def clean_html(raw_html: str) -> str:
    if not isinstance(raw_html, str):
        return ""
    # On supprime les balises HTML
    clean = re.sub('<[^<]+?>', ' ', raw_html)
    # On décode les entités HTML (&nbsp; -> espace, etc.)
    clean = html.unescape(clean)
    # On normalise les espaces
    clean = re.sub(r'\s+', ' ', clean)
    return clean.strip()

def build_embedding_text_for_json(product: dict) -> str:
    parts = []
    
    label = product.get("label", "")
    if label and isinstance(label, str):
        parts.append(label.strip() + ".")
        
    desc_dict = product.get("description", {})
    desc_fr = desc_dict.get("fr", "") if isinstance(desc_dict, dict) else ""
    if desc_fr:
        parts.append(clean_html(desc_fr))
        
    dest_type = product.get("destination_type", "")
    if dest_type and isinstance(dest_type, str):
        parts.append(f"Destination: {dest_type.strip()}.")
        
    city = product.get("city", "")
    region_dict = product.get("residence_region_label", {})
    region_fr = region_dict.get("fr", "") if isinstance(region_dict, dict) else ""
    country = product.get("country", "")
    
    loc_parts = [str(p).strip() for p in [city, region_fr, country] if p and isinstance(p, str) and p.strip()]
    if loc_parts:
        parts.append("Lieu: " + ", ".join(loc_parts) + ".")
        
    res_type_list = product.get("residence_type", [])
    type_logs = [t.get("label") for t in res_type_list if isinstance(t, dict) and t.get("label")] if isinstance(res_type_list, list) else []
    if type_logs:
        parts.append(f"Type: {', '.join(type_logs)}.")
        
    facilities = []
    for f in (product.get("residence_facilities", []) if isinstance(product.get("residence_facilities"), list) else []):
        if isinstance(f, dict) and f.get("label"): facilities.append(f.get("label"))
    for f in (product.get("product_facilities", []) if isinstance(product.get("product_facilities"), list) else []):
        if isinstance(f, dict) and f.get("label"): facilities.append(f.get("label"))
        
    facilities = [str(f) for f in set(facilities) if f]
    if facilities:
        parts.append(f"Équipements: {', '.join(facilities)}.")
        
    capacity = product.get("capacity", {})
    total_capacity = capacity.get("total") if isinstance(capacity, dict) else None
    if total_capacity:
        parts.append(f"Capacité: {total_capacity} personnes.")
        
    min_price = product.get("min_price_per_night")
    if min_price:
        parts.append(f"Prix à partir de {min_price}€ / nuit.")

    # Points d'intérêts
    poi_list = product.get("poi_labels") or [] # S'assure d'avoir au moins une liste vide
    if isinstance(poi_list, list):
        poi_list = [str(p.get("label")).strip() for p in poi_list if isinstance(p, dict) and p.get("label")]
        if poi_list:
            parts.append(f"Points d'intérêts: {', '.join(poi_list)}.")
        
    return " ".join(parts)


def ingest_to_opensearch(index_name: str):
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "http://localhost:9200")
    OPENSEARCH_USERNAME = os.getenv("OPENSEARCH_USERNAME")
    OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD")

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY doit être défini dans le .env")

    init_client(OPENAI_API_KEY)

    # Déterminer si on utilise SSL basé sur l'URL
    use_ssl = OPENSEARCH_HOST.startswith('https')

    # Configuration de la connexion
    connection_params = {
        'hosts': [OPENSEARCH_HOST],
        'http_auth': (OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
        'use_ssl': use_ssl,
        'verify_certs': False,  # Set to True in production with valid certificates
        'ssl_show_warn': False
    }

    print(f"🔌 Connexion à OpenSearch: {OPENSEARCH_HOST} (SSL: {use_ssl}, User: {OPENSEARCH_USERNAME})")
    client = OpenSearch(**connection_params)

    print(f"📥 Récupération des documents depuis l'index '{index_name}'...")
    
    try:
        page = client.search(
            index=index_name,
            scroll='2m',
            size=100,
            body={
                "query": {
                    "bool": {
                        "must_not": [
                            {"exists": {"field": "my_vector"}}
                        ]
                    }
                },
                "_source": True
            }
        )
    except Exception as e:
        print(f"❌ Erreur lors de la lecture de l'index: {e}")
        return

    sid = page['_scroll_id']
    scroll_size = len(page['hits']['hits'])
    
    docs_to_update = []
    
    while scroll_size > 0:
        for hit in page['hits']['hits']:
            docs_to_update.append((hit['_id'], hit['_source']))
            
        page = client.scroll(scroll_id=sid, scroll='2m')
        sid = page['_scroll_id']
        scroll_size = len(page['hits']['hits'])
        
    client.clear_scroll(scroll_id=sid)
    
    if not docs_to_update:
        print("⚠️ Aucun document trouvé dans l'index.")
        return
        
    print(f"✅ {len(docs_to_update)} documents récupérés. Génération des embeddings en cours...")
    
    batch_size = 100
    for i in range(0, len(docs_to_update), batch_size):
        batch = docs_to_update[i:i + batch_size]
        texts = [build_embedding_text_for_json(doc) for _, doc in batch]
        
        try:
            embeddings = get_embeddings_batch(texts)
        except Exception as e:
            print(f"❌ Erreur pendant le calcul des embeddings: {e}")
            break
            
        actions = []
        for (doc_id, _), vector in zip(batch, embeddings):
            actions.append({
                "_op_type": "update",
                "_index": index_name,
                "_id": doc_id,
                "doc": {
                    "my_vector": vector
                }
            })
            
        try:
            helpers.bulk(client, actions)
            print(f"   → Mise à jour effectuée pour les documents {i+1} à {i+len(batch)}/{len(docs_to_update)}")
        except Exception as e:
            print(f"❌ Erreur lors de la mise à jour (bulk) dans OpenSearch: {e}")
            break

    print(f"🎉 Ingestion terminée avec succès !")

def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default=os.getenv("OPENSEARCH_INDEX", "semantic_search_index"))
    args = parser.parse_args()
    ingest_to_opensearch(args.index)

if __name__ == "__main__":
    main()
