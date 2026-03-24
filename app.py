import os
import time

import streamlit as st
from utils.search_engine import extract_intent, semantic_search_with_intent, semantic_search_with_intent_v2
from utils.sidebar import render_sidebar

st.set_page_config(
    page_title="🔍 Recherche Sémantique Séjours",
    page_icon="🏖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRODUCT_ROOT_URL = os.getenv("PRODUCT_ROOT_URL", "").rstrip("/")

st.title("🌊 Recherche sémantique ⛰")
# ── Initialisation des états de session ─────────────────────────────────────
for key, default in [
    ("index", None), ("metadata", None),
    ("results", []), ("intent", None), ("last_query", ""),
    ("search_duration", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar ───────────────────────────────────────────────────────────────────
api_key, top_k = render_sidebar()

# ── Barre de recherche ────────────────────────────────────────────────────────
col_q, col_btn = st.columns([5, 1])
with col_q:
    query = st.text_input(
        "Recherche",
        placeholder='Ex : "villa avec piscine pour 6 personnes en bord de mer dans le sud" ou "escapade romantique insolite en forêt"',
        label_visibility="collapsed",
    )
with col_btn:
    search_btn = st.button("🔍 Rechercher", width="stretch", type="primary")

st.caption(
    "💡 Essayez : \"chalet ski famille 8 personnes alpes\" · "
    "\"séjour romantique bord de mer sud de la France\" · "
    "\"nature calme écologie\" · \"luxe vue mer méditerranée\" · "
    "\"city-break gastronomie vin bordeaux\""
)

# ── Lancement de la recherche ─────────────────────────────────────────────────
run_search = (search_btn and query.strip()) or (
    query.strip() and query != st.session_state.last_query and search_btn
)

if search_btn and query.strip():

    if not api_key:
        st.error("⚠️ Clé API manquante.")
    else:
        st.session_state.last_query = query
        start_time = time.perf_counter()

        with st.spinner("Analyse de l'intention et expansion géographique…"):
            try:
                intent = extract_intent(query)
                st.session_state.intent = intent
            except Exception as e:
                st.error(f"Erreur extraction intention : {e}")
                st.stop()

        with st.spinner("Recherche sémantique…"):
            try:
                results = semantic_search_with_intent_v2(
                    intent=intent,
                    metadata=st.session_state.metadata,
                    top_k=top_k,
                )
                st.session_state.results = results
                st.session_state.search_duration = time.perf_counter() - start_time
            except Exception as e:
                st.error(f"Erreur recherche : {e}")


# ── Affichage des résultats ───────────────────────────────────────────────────
intent  = st.session_state.intent
results = st.session_state.results

if intent and st.session_state.last_query:
    filters = intent.get("filters", {})

    # ── Bloc intention détectée ───────────────────────────────────────────────
    geo = filters.get("zone_geo_label") or filters.get("pays")
    type_log = filters.get("type_logement")
    nb = filters.get("nb_personnes")
    prix_min = filters.get("prix_min", None)
    prix_max = filters.get("prix_max", None)
    ambiances = filters.get("ambiance") or []
    activites = filters.get("activites") or []
    date_debut = filters.get("date_debut")
    date_fin = filters.get("date_fin")
    semantic_q = intent.get("semantic_query", "")

    with st.container(border=True):
        st.caption("🧠 Intention détectée")
        st.write(intent.get("intent_summary", ""))

        filtres_lignes: list[str] = []
        if geo:
            is_poi = filters.get("is_poi", False)
            poi_tag = " (POI 🏰)" if is_poi else ""
            if filters.get("activity_inferred") and geo.startswith("~"):
                label = geo[1:]
                filtres_lignes.append(f"📍 Zone déduite : **{label}**{poi_tag}")
            else:
                filtres_lignes.append(f"📍 Zone : **{geo}**{poi_tag}")
        if type_log:
            filtres_lignes.append(f"🏠 Type de logement : **{type_log}**")
        if nb:
            filtres_lignes.append(f"👥 Capacité : **{nb} personnes**")
        if prix_min:
            filtres_lignes.append(f"💰 Prix min : **{prix_min} €/nuit**")
        if prix_max:
            filtres_lignes.append(f"💰 Prix max : **{prix_max} €/nuit**")
        if ambiances:
            filtres_lignes.append("✨ Ambiance : " + ", ".join(f"`{v}`" for v in ambiances))
        if activites:
            filtres_lignes.append("🎯 Activités : " + ", ".join(f"`{a}`" for a in activites))
        if date_debut:
            if date_fin:
                filtres_lignes.append(f"📅 Dates : du **{date_debut}** au **{date_fin}**")
            else:
                filtres_lignes.append(f"📅 Arrivée le : **{date_debut}**")

        if filtres_lignes:
            st.markdown("\n".join(f"- {l}" for l in filtres_lignes))
        else:
            st.caption("Aucun filtre détecté — recherche purement sémantique")

        villes_expanded = filters.get("villes_expanded") or []
        if villes_expanded:
            villes_str = ", ".join(villes_expanded)
            st.caption(f"🗺️ Communes couvertes : **{villes_str}**")

        if semantic_q:
            st.caption(f"🔢 Requête vectorisée : _{semantic_q}_")

    # ── Résultats ─────────────────────────────────────────────────────────────
    if results:
        n = len(results)
        duration = st.session_state.get("search_duration")
        if duration is not None:
            st.caption(
                f"{n} séjour{'s' if n > 1 else ''} correspondant{'s' if n > 1 else ''} "
                f"(traité en {duration:.2f} s)"
            )
        else:
            st.caption(f"{n} séjour{'s' if n > 1 else ''} correspondant{'s' if n > 1 else ''}")

        for idx, sejour in enumerate(results):
            rank = idx + 1
            score = sejour.get("score", 0)
            equips = " · ".join(sejour.get("equipements", [])[:5])
            extra = "…" if len(sejour.get("equipements", [])) > 5 else ""
            tags = sejour.get("tags", [])
            
            # Troncature de la description pour le punch visuel
            desc = sejour.get("description", "")
            if len(desc) > 280:
                desc = desc[:277] + "..."

            with st.container(border=True):
                col_img, col_info = st.columns([0.3, 0.7], gap="medium")
                
                with col_img:
                    # Affichage de l'image avec un ratio fixe si possible (via CSS global ou juste st.image)
                    st.image(sejour.get("image"), width="stretch")
                    st.caption(f"Score de pertinence : **{score:.1f}%**")
                    st.progress(min(score / 100, 1.0))

                with col_info:
                    title_col, rank_col = st.columns([5, 1])
                    with title_col:
                        st.subheader(f"{sejour['titre']}")
                    with rank_col:
                        st.markdown(f"**#{rank}**")

                    st.markdown(f"📍 **{sejour['ville']}**, {sejour['region']}  ·  *{sejour['type_logement']}*")
                    
                    # Infos primordiales sous forme de badges/lignes denses
                    c1, c2, c3, c4 = st.columns(4)
                    c1.markdown(f"💰 **{sejour.get('prix_nuit', '-')} €** /nuit")
                    c2.markdown(f"👥 **{sejour.get('nb_personnes', '-')}** pers.")
                    c3.markdown(f"🛏️ **{sejour.get('chambres', '-')}** ch.")
                    c4.markdown(f"📏 **{sejour.get('surface', '-')} m²**")

                    st.write(desc)

                    if tags:
                        st.markdown(" ".join(f"`{t}`" for t in tags[:6]))

                    st.caption(f"➕ {equips}{extra}")

                    # Pied de carte : lien
                    article_cle = sejour.get("id")
                    if PRODUCT_ROOT_URL and article_cle:
                        url = f"{PRODUCT_ROOT_URL}{article_cle}"
                        st.link_button("Découvrir ce séjour", url, type="secondary", width="content")
    else:
        geo_label = filters.get("zone_geo_label") or filters.get("pays") or ""
        zone_msg = f" dans la zone **{geo_label}**" if geo_label else ""
        st.info(
            "🔍 Aucun séjour trouvé" +
            zone_msg +
            ". Les filtres géographiques sont appliqués strictement — essayez une zone plus large."
        )

elif not st.session_state.last_query:
    st.info("✨ Décrivez votre séjour idéal ci-dessus")