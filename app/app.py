import requests
import streamlit as st
import pandas as pd
from pathlib import Path
import os

API_URL = os.getenv("API_URL", "https://recettes-mlops-production.up.railway.app")
BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = Path(__file__).parent / "logo_ccistta.png"
RECETTES_MENSUELLES_PATH = BASE_DIR / "data" / "recettes_mensuelles.csv"
RECETTES_FEATURES_PATH = BASE_DIR / "data" / "recettes_features.csv"
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
        width: 250px !important;
        min-width: 250px !important;
        max-width: 250px !important;
        background: #FFFFFF !important;
        border-right: 1px solid #E5E7EB;
    }

    section[data-testid="stSidebar"] img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        margin-top: 12px;
        margin-bottom: 25px;
    }

    h1, h2, h3 {
        color: #0B2545;
        font-family: Arial, sans-serif;
    }

    .main-card {
        background-color: white;
        padding: 22px;
        border-radius: 18px;
        border-left: 6px solid #0B4F8A;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.06);
        margin-bottom: 20px;
    }

    div[role="radiogroup"] > label {
        background: transparent !important;
        border-radius: 10px !important;
        padding: 8px 10px !important;
        margin-bottom: 4px !important;
    }

    div[role="radiogroup"] > label:hover {
        background: #F1F5F9 !important;
    }

    div[role="radiogroup"] p {
        font-size: 14px !important;
        font-weight: 500 !important;
        color: #1F2937 !important;
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
    st.image(str(LOGO_PATH), width=110)

    st.markdown("""
    <div style="margin-top:10px;">
        <div style="font-size:16px; font-weight:700; color:#0B2545;">
            CCISTTA
        </div>
        <div style="font-size:12px; color:#6B7280; line-height:1.5; margin-top:8px;">
            Solution MLOps de<br>
            prévision des recettes
        </div>
    </div>

    <div style="font-size:12px; font-weight:700; color:#0B2545; margin-top:25px; margin-bottom:8px;">
        Navigation
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "",
        [
            "🏠 Accueil",
            "📈 Prévision globale",
            "🕒 Prévision par segment",
            "📊 Visualisation générale",
            "ℹ️ Informations techniques"
        ],
        label_visibility="collapsed"
    )

    page = page.replace("🏠 ", "")
    page = page.replace("📈 ", "")
    page = page.replace("🕒 ", "")
    page = page.replace("📊 ", "")
    page = page.replace("ℹ️ ", "")

    st.markdown("""
    <div style="
        background:#EFF6FF;
        border:1px solid #D7E8FF;
        border-radius:12px;
        padding:12px;
        margin-top:70px;
        color:#0B4F8A;
        font-size:12px;
        font-weight:600;
        line-height:1.9;
    ">
        🛡️ Solution fiable<br>
        🔐 Sécurisée<br>
        📈 Évolutive
    </div>
    """, unsafe_allow_html=True)


# =========================
# PAGE ACCUEIL
# =========================
if page == "Accueil":

    st.title("Solution MLOps de Prévision des Recettes")
    st.subheader("Chambre de Commerce, d’Industrie et de Services Tanger-Tétouan-Al Hoceima")

    st.markdown("### 🛡️ Solution MLOps")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Modèle", "XGBoost")
        st.caption("Algorithme performant de machine learning")

    with col2:
        st.metric("API", "FastAPI")
        st.caption("API RESTful rapide et scalable")

    with col3:
        st.metric("Données", "Historiques")
        st.caption("Données mensuelles fiables et consolidées")

    with col4:
        st.metric("Déploiement", "Cloud")
        st.caption("Déploiement sécurisé et haute disponibilité")

    st.markdown("---")

    st.markdown("### Architecture de la solution")

    a1, a2, a3, a4, a5 = st.columns(5)

    with a1:
        st.info("🗄️ **Données**\n\nHistoriques")

    with a2:
        st.info("🧠 **Modèle**\n\nXGBoost")

    with a3:
        st.info("⚡ **API**\n\nFastAPI")

    with a4:
        st.info("🖥️ **Interface**\n\nStreamlit")

    with a5:
        st.info("👥 **Utilisateurs**\n\nDécision")

    st.markdown("---")

    st.success(
        "📈 **Anticipez, analysez et pilotez vos recettes en toute confiance.**\n\n"
        "Accédez aux prévisions globales ou segmentées et suivez l’évolution des recettes en temps réel."
    )

    st.button("🚀 Commencez à prévoir")


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
        data = pd.read_csv(RECETTES_MENSUELLES_PATH)

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

        recettes_totales = data_filtre["recettes"].sum()
        transactions_totales = data_filtre["transactions"].sum()
        recette_moyenne = data_filtre["recettes"].mean()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Recettes totales",
                f"{recettes_totales:,.0f} DH"
            )

        with col2:
            st.metric(
                "Transactions totales",
                f"{transactions_totales:,.0f}"
            )

        with col3:
            st.metric(
                "Recette moyenne mensuelle",
                f"{recette_moyenne:,.0f} DH"
            )

        st.markdown("### Tableau de bord mensuel")

        col_graph1, col_graph2 = st.columns(2)

        # =========================
        # GRAPHE RECETTES
        # =========================
        with col_graph1:

            st.subheader("Évolution mensuelle des recettes")

            chart_recettes = data_filtre.set_index(
                "mois_nom"
            )[["recettes"]]

            st.line_chart(
                chart_recettes,
                height=300,
                use_container_width=True
            )

        # =========================
        # GRAPHE TRANSACTIONS
        # =========================
        with col_graph2:

            st.subheader("Évolution mensuelle des transactions")

            chart_transactions = data_filtre.set_index(
                "mois_nom"
            )[["transactions"]]

            st.line_chart(
                chart_transactions,
                height=300,
                use_container_width=True
            )

        # =========================
        # EVOLUTION ANNUELLE
        # =========================
        st.markdown("### Évolution annuelle globale")

        data_annuelle = (
            data.groupby("annee", as_index=False)
            .agg({
                "recettes": "sum",
                "transactions": "sum"
            })
            .sort_values("annee")
        )

        col_year1, col_year2 = st.columns(2)

        # =========================
        # RECETTES ANNUELLES
        # =========================
        with col_year1:

            st.subheader("Recettes annuelles")

            chart_recettes_annuelles = (
                data_annuelle.set_index("annee")[["recettes"]]
            )

            st.line_chart(
                chart_recettes_annuelles,
                height=300,
                use_container_width=True
            )

        # =========================
        # TRANSACTIONS ANNUELLES
        # =========================
        with col_year2:

            st.subheader("Transactions annuelles")

            chart_transactions_annuelles = (
                data_annuelle.set_index("annee")[["transactions"]]
            )

            st.line_chart(
                chart_transactions_annuelles,
                height=300,
                use_container_width=True
            )

    except Exception as e:
        st.error(
            f"Erreur lors du chargement des données : {e}"
        )

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