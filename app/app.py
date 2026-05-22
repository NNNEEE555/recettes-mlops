import requests
import streamlit as st
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

API_URL = "https://recettes-mlops-production.up.railway.app"
BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = Path(__file__).parent / "logo_ccistta.png"
RECETTES_MENSUELLES_PATH = BASE_DIR / "data" / "recettes_mensuelles.xlsx"
RECETTES_FEATURES_PATH = BASE_DIR / "data" / "recettes_features.xlsx"
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
        Cette page présente une vue synthétique des recettes et des transactions.
        Les indicateurs sont calculés automatiquement à partir des données mensuelles réelles.
    </div>
    """, unsafe_allow_html=True)

    try:
        data = pd.read_excel(RECETTES_MENSUELLES_PATH)

        data["date"] = pd.to_datetime(data["date"])
        data["annee"] = data["date"].dt.year
        data["month"] = data["date"].dt.month

        mois_noms = {
            1: "Jan", 2: "Fév", 3: "Mar", 4: "Avr",
            5: "Mai", 6: "Juin", 7: "Juil", 8: "Août",
            9: "Sep", 10: "Oct", 11: "Nov", 12: "Déc"
        }

        data["mois_nom"] = data["month"].map(mois_noms)

        annees = sorted(data["annee"].dropna().unique())

        annee_selectionnee = st.selectbox(
            "Choisir l'année à afficher",
            annees,
            index=len(annees) - 1
        )

        data_filtre = data[data["annee"] == annee_selectionnee].copy()
        data_filtre = data_filtre.sort_values("month")

        recette_totale = data_filtre["recettes"].sum()
        transaction_totale = data_filtre["transactions"].sum()
        recette_moyenne = data_filtre["recettes"].mean()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Recette totale",
                f"{recette_totale:,.2f} DH"
            )

        with col2:
            st.metric(
                "Transactions totales",
                f"{transaction_totale:,.0f}"
            )

        with col3:
            st.metric(
                "Recette moyenne mensuelle",
                f"{recette_moyenne:,.2f} DH"
            )

        st.subheader(f"Évolution mensuelle des recettes - {annee_selectionnee}")

        fig, ax = plt.subplots()
        ax.plot(
            data_filtre["mois_nom"],
            data_filtre["recettes"],
            marker="o"
        )
        ax.set_xlabel("Mois")
        ax.set_ylabel("Recettes en DH")
        ax.set_title(f"Recettes mensuelles - {annee_selectionnee}")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        st.subheader(f"Transactions mensuelles - {annee_selectionnee}")

        fig, ax = plt.subplots()
        ax.bar(
            data_filtre["mois_nom"],
            data_filtre["transactions"]
        )
        ax.set_xlabel("Mois")
        ax.set_ylabel("Nombre de transactions")
        ax.set_title(f"Transactions mensuelles - {annee_selectionnee}")
        ax.grid(True, axis="y", alpha=0.3)
        st.pyplot(fig)

        with st.expander("Afficher les données mensuelles"):
            st.dataframe(
                data_filtre[["date", "recettes", "transactions", "trimestre"]],
                use_container_width=True
            )

    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
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