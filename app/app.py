import requests
import streamlit as st
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

API_URL = "https://recettes-mlops-production.up.railway.app"
LOGO_PATH = Path(__file__).parent / "logo_ccistta.png"
st.set_page_config(
    page_title="Prévision des recettes CCISTTA",
    page_icon="📊",
    layout="wide"
)

# =========================
# THEME CCISTTA
# =========================
st.markdown("""
<style>
    .stApp {
        background-color: #F7F9FC;
    }

    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E5E7EB;
    }

    h1, h2, h3 {
        color: #0B4F8A;
        font-family: Arial, sans-serif;
    }

    .main-card {
        background-color: white;
        padding: 25px;
        border-radius: 14px;
        border-left: 6px solid #0B4F8A;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.06);
        margin-bottom: 20px;
    }

    .kpi-card {
        background-color: white;
        padding: 20px;
        border-radius: 14px;
        text-align: center;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.06);
        border-top: 4px solid #008C45;
    }

    .kpi-title {
        font-size: 15px;
        color: #555;
    }

    .kpi-value {
        font-size: 24px;
        font-weight: bold;
        color: #0B4F8A;
    }

    .stButton>button {
        background-color: #0B4F8A;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
    }

    .stButton>button:hover {
        background-color: #008C45;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.image(str(LOGO_PATH), width=120)
    st.markdown("### CCISTTA")
    st.caption("Solution MLOps de prévision des recettes")

    page = st.radio(
        "Navigation",
        [
            "Accueil",
            "Prévision globale",
            "Prévision par segment",
            "Visualisation générale",
            "Informations techniques"
        ]
    )


# =========================
# HEADER
# =========================
col_logo, col_title = st.columns([1, 6])

with col_logo:
    st.image(str(LOGO_PATH), width=90)

with col_title:
    st.title("Solution MLOps de Prévision des Recettes")
    st.markdown("### Chambre de Commerce, d’Industrie et de Services Tanger-Tétouan-Al Hoceima")


# =========================
# PAGE ACCUEIL
# =========================
if page == "Accueil":
    st.markdown("""
    <div class="main-card">
        <h3>Objectif de la solution</h3>
        <p>
        Cette application permet d'exploiter un modèle de prévision des recettes à travers
        une interface simple, accessible et professionnelle. Elle communique avec une API
        FastAPI déployée sur Railway afin de générer des prévisions globales ou segmentées.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Modèle</div>
            <div class="kpi-value">XGBoost</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">API</div>
            <div class="kpi-value">FastAPI</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Déploiement</div>
            <div class="kpi-value">Cloud</div>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Vérification de l’API")

    if st.button("Tester la connexion API"):
        try:
            response = requests.get(f"{API_URL}/health", timeout=10)

            if response.status_code == 200:
                st.success("API disponible et fonctionnelle")
                st.json(response.json())
            else:
                st.error(f"API non disponible : {response.status_code}")

        except Exception as e:
            st.error(f"Erreur de connexion à l'API : {e}")


# =========================
# PAGE PREVISION GLOBALE
# =========================
elif page == "Prévision globale":
    st.header("Prévision globale des recettes")

    st.markdown("""
    <div class="main-card">
        Cette page permet de générer une prévision globale des recettes pour une période future.
    </div>
    """, unsafe_allow_html=True)

    date_cible = st.text_input(
        "Date cible",
        value="2026-01",
        help="Format attendu : YYYY-MM"
    )

    if st.button("Générer la prévision globale"):
        payload = {"date": date_cible}

        try:
            response = requests.post(
                f"{API_URL}/predict/global/auto",
                json=payload,
                timeout=20
            )

            if response.status_code == 200:
                result = response.json()
                prediction = result.get("prediction_recettes")

                st.success("Prévision générée avec succès")

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
# PAGE PREVISION SEGMENT
# =========================
elif page == "Prévision par segment":
    st.header("Prévision par segment")

    st.markdown("""
    <div class="main-card">
        Cette page permet de générer une prévision selon un segment :
        secteur, produit, antenne ou type d'adhérent.
    </div>
    """, unsafe_allow_html=True)

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
        "secteur": ["Commerce", "Industrie", "Service", "Hors secteur"],
        "produit": [
            "Attestation Professionnelle",
            "Carte de Commerce",
            "Certificat d'origine",
            "Certificat de commercialisation libre",
            "Légalisation",
            "Location de salles"
        ],
        "antenne": ["Tanger", "Tétouan", "Larache", "Al Hoceima"],
        "adherent": ["Particulier", "PM", "PP"]
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

                st.success("Prévision segmentée générée avec succès")

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
# PAGE VISUALISATION GENERALE
# =========================
elif page == "Visualisation générale":
    st.header("Visualisation générale")

    st.markdown("""
    <div class="main-card">
        Cette partie présente une vue générale et synthétique des recettes.
        Elle sert à donner une lecture rapide de l’évolution, sans entrer dans une analyse détaillée.
    </div>
    """, unsafe_allow_html=True)

    data = pd.DataFrame({
        "Mois": ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin"],
        "Recettes": [120000, 135000, 128000, 150000, 162000, 158000],
        "Transactions": [320, 350, 330, 390, 410, 400]
    })

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Recettes moyennes", "142 166 DH")

    with col2:
        st.metric("Transactions moyennes", "366")

    with col3:
        st.metric("Tendance", "Positive")

    st.subheader("Évolution générale des recettes")
    fig, ax = plt.subplots()
    ax.plot(data["Mois"], data["Recettes"], marker="o")
    ax.set_xlabel("Mois")
    ax.set_ylabel("Recettes")
    ax.set_title("Évolution générale des recettes")
    st.pyplot(fig)

    st.subheader("Vue générale des transactions")
    fig, ax = plt.subplots()
    ax.bar(data["Mois"], data["Transactions"])
    ax.set_xlabel("Mois")
    ax.set_ylabel("Transactions")
    ax.set_title("Vue générale des transactions")
    st.pyplot(fig)

    st.info("Cette visualisation est indicative. Elle peut être remplacée plus tard par des données réelles issues de PostgreSQL ou de l’API.")


# =========================
# PAGE INFORMATIONS TECHNIQUES
# =========================
elif page == "Informations techniques":
    st.header("Informations techniques")

    st.subheader("Architecture générale")

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
Streamlit Cloud
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
- GitHub
- Streamlit Cloud
""")