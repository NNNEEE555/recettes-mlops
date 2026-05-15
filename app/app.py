import requests
import streamlit as st


API_URL = "https://recettes-mlops-production.up.railway.app"


st.set_page_config(
    page_title="Prévision des recettes CCISTTA",
    page_icon="📊",
    layout="wide"
)


st.title("📊 Solution MLOps de Prévision des Recettes CCISTTA")

st.markdown("""
Cette interface permet d'utiliser le modèle de prévision des recettes sans passer par Swagger.
Elle communique avec l'API FastAPI déployée sur Railway.
""")


# =========================
# Sidebar
# =========================
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choisir une page",
    [
        "Accueil",
        "Prévision globale",
        "Prévision par segment",
        "Informations techniques"
    ]
)


# =========================
# Page Accueil
# =========================
if page == "Accueil":
    st.header("🏠 Accueil")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Modèle", "XGBoost")

    with col2:
        st.metric("API", "FastAPI")

    with col3:
        st.metric("Déploiement", "Railway")

    st.subheader("🎯 Objectif")
    st.write("""
    Cette solution permet d'automatiser la prévision des recettes de la CCISTTA
    à partir d'un pipeline MLOps intégrant nettoyage, feature engineering,
    modélisation, API, orchestration et déploiement cloud.
    """)

    st.subheader("🔎 Vérification de l'API")

    if st.button("Tester la connexion API"):
        try:
            response = requests.get(f"{API_URL}/health", timeout=10)

            if response.status_code == 200:
                st.success("✅ API disponible et fonctionnelle")
                st.json(response.json())
            else:
                st.error(f"❌ API non disponible : {response.status_code}")

        except Exception as e:
            st.error(f"Erreur de connexion à l'API : {e}")

    st.subheader("🔗 Liens utiles")
    st.markdown(f"- [Swagger API]({API_URL}/docs)")
    st.markdown("- Dashboard Power BI : à intégrer ou ouvrir séparément")


# =========================
# Page Prévision globale
# =========================
elif page == "Prévision globale":
    st.header("📈 Prévision globale des recettes")

    st.write("""
    Cette page permet de générer une prévision globale des recettes pour une période future.
    """)

    date_cible = st.text_input(
        "Date cible",
        value="2026-01",
        help="Format attendu : YYYY-MM"
    )

    if st.button("Générer la prévision globale"):
        payload = {
            "date": date_cible
        }

        try:
            response = requests.post(
                f"{API_URL}/predict/global/auto",
                json=payload,
                timeout=20
            )

            if response.status_code == 200:
                result = response.json()

                prediction = result.get("prediction_recettes")

                st.success("✅ Prévision générée avec succès")

                st.metric(
                    label=f"Recettes prévues pour {date_cible}",
                    value=f"{prediction:,.2f} DH"
                )

                st.subheader("Réponse API")
                st.json(result)

            else:
                st.error(f"Erreur API : {response.status_code}")
                st.text(response.text)

        except Exception as e:
            st.error(f"Erreur lors de l'appel API : {e}")


# =========================
# Page Prévision par segment
# =========================
elif page == "Prévision par segment":
    st.header("🧩 Prévision par segment")

    st.write("""
    Cette page permet de générer une prévision selon un segment :
    secteur, produit, antenne ou type d'adhérent.
    """)

    col1, col2 = st.columns(2)

    with col1:
        segment_type = st.selectbox(
            "Type de segment",
            ["secteur", "produit", "antenne", "adherent"]
        )

    with col2:
        date_cible = st.text_input(
            "Date cible",
            value="2026-01",
            help="Format attendu : YYYY-MM"
        )

    valeurs_par_segment = {
        "secteur": [
            "Commerce",
            "Industrie",
            "Service",
            "Hors secteur"
        ],
        "produit": [
            "Attestation Professionnelle",
            "Carte de Commerce",
            "Certificat d'origine",
            "Certificat de commercialisation libre",
            "Légalisation",
            "Location de salles"
        ],
        "antenne": [
            "Tanger",
            "Tétouan",
            "Larache",
            "Al Hoceima"
        ],
        "adherent": [
            "Particulier",
            "PM",
            "PP"
        ]
    }

    segment_value = st.selectbox(
        "Valeur du segment",
        valeurs_par_segment[segment_type]
    )

    if st.button("Générer la prévision segmentée"):
        payload = {
            "date": date_cible,
            "segment_type": segment_type,
            "segment_value": segment_value
        }

        try:
            response = requests.post(
                f"{API_URL}/predict/segment/auto",
                json=payload,
                timeout=20
            )

            if response.status_code == 200:
                result = response.json()

                prediction = result.get("prediction_recettes")

                st.success("✅ Prévision segmentée générée avec succès")

                st.metric(
                    label=f"{segment_type} - {segment_value}",
                    value=f"{prediction:,.2f} DH"
                )

                st.subheader("Réponse API")
                st.json(result)

            else:
                st.error(f"Erreur API : {response.status_code}")
                st.text(response.text)

        except Exception as e:
            st.error(f"Erreur lors de l'appel API : {e}")


# =========================
# Page Informations techniques
# =========================
elif page == "Informations techniques":
    st.header("⚙️ Informations techniques")

    st.subheader("Architecture")
    st.code("""
Excel / PostgreSQL / Neon
        ↓
Airflow Pipeline
        ↓
Feature Engineering
        ↓
XGBoost
        ↓
FastAPI Railway
        ↓
Streamlit App
        ↓
Utilisateur métier
    """)

    st.subheader("Endpoints utilisés")

    st.markdown("""
| Endpoint | Méthode | Rôle |
|---|---|---|
| `/health` | GET | Vérifier l'état de l'API |
| `/predict/global/auto` | POST | Prévision globale |
| `/predict/segment/auto` | POST | Prévision par segment |
""")

    st.subheader("Stack technique")

    st.markdown("""
- Python
- Streamlit
- FastAPI
- XGBoost
- PostgreSQL / Neon
- Docker
- Railway
- Airflow
- Power BI
""")