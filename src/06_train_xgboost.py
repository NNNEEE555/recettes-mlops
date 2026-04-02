from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from db_connection import engine


BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


GLOBAL_FEATURES = [
    "annee",
    "month",
    "trimestre",
    "lag_1",
    "lag_3",
    "lag_6",
    "moyenne_mobile",
    "transactions",
    "transactions_lag_1",
    "transactions_lag_3",
    "transactions_rolling_3",
    "transactions_diff",
    "month_sin",
    "month_cos",
    "trend",
]

SEGMENT_NUM_FEATURES = [
    "annee",
    "month",
    "trimestre",
    "lag_1",
    "lag_3",
    "lag_6",
    "moyenne_mobile",
    "transactions",
    "transactions_lag_1",
    "transactions_lag_3",
    "transactions_rolling_3",
    "transactions_diff",
    "month_sin",
    "month_cos",
    "trend",
]

SEGMENT_CAT_FEATURES = [
    "segment_type",
    "segment_value",
]


def build_xgb_model():
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.2,
        reg_lambda=2.0,
        random_state=42
    )

# 1. Données globales
def load_global_data():
    query = """
SELECT 
    annee,
    month,
    trimestre,
    lag_1,
    lag_3,
    lag_6,
    moyenne_mobile,
    transactions,
    transactions_lag_1,
    transactions_lag_3,
    transactions_rolling_3,
    transactions_diff,
    month_sin,
    month_cos,
    trend,
    recettes,
    date
FROM recettes_mensuelles
ORDER BY date
"""
    df = pd.read_sql(query, engine)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df = df.dropna(subset=GLOBAL_FEATURES + ["recettes"])
    return df

# 2. Données segmentées
def load_segment_tables():
    queries = {
        "secteur": """
    SELECT 
        secteur AS segment_value,
        annee,
        month,
        trimestre,
        lag_1,
        lag_3,
        lag_6,
        moyenne_mobile,
        transactions,
        transactions_lag_1,
        transactions_lag_3,
        transactions_rolling_3,
        transactions_diff,
        month_sin,
        month_cos,
        trend,
        recettes,
        date
    FROM recettes_secteur
""",
        "produit": """
    SELECT 
        produit AS segment_value,
        annee,
        month,
        trimestre,
        lag_1,
        lag_3,
        lag_6,
        moyenne_mobile,
        transactions,
        transactions_lag_1,
        transactions_lag_3,
        transactions_rolling_3,
        transactions_diff,
        month_sin,
        month_cos,
        trend,
        recettes,
        date
    FROM recettes_produit
""",
        "antenne": """
    SELECT 
        antenne AS segment_value,
        annee,
        month,
        trimestre,
        lag_1,
        lag_3,
        lag_6,
        moyenne_mobile,
        transactions,
        transactions_lag_1,
        transactions_lag_3,
        transactions_rolling_3,
        transactions_diff,
        month_sin,
        month_cos,
        trend,
        recettes,
        date
    FROM recettes_antenne
""",
        "adherent": """
    SELECT 
        personne_physique AS segment_value,
        annee,
        month,
        trimestre,
        lag_1,
        lag_3,
        lag_6,
        moyenne_mobile,
        transactions,
        transactions_lag_1,
        transactions_lag_3,
        transactions_rolling_3,
        transactions_diff,
        month_sin,
        month_cos,
        trend,
        recettes,
        date
    FROM recettes_adherent
""",
    }

    frames = []

    for segment_type, query in queries.items():
        df = pd.read_sql(query, engine)
        df["segment_type"] = segment_type
        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)
    df_all["date"] = pd.to_datetime(df_all["date"])
    df_all = df_all.sort_values(["segment_type", "segment_value", "date"])

    df_all = df_all.dropna(subset=SEGMENT_NUM_FEATURES + SEGMENT_CAT_FEATURES + ["recettes"])
    return df_all

# 3. Entraînement global
def train_global_model():
    df = load_global_data()

    X = df[GLOBAL_FEATURES]
    y = np.log1p(df["recettes"])

    print("Dimensions globales après dropna :", df.shape)

    model = build_xgb_model()
    model.fit(X, y)

    payload = {
        "model": model,
        "features": GLOBAL_FEATURES,
        "target_transform": "log1p"
    }

    joblib.dump(payload, MODELS_DIR / "xgb_global.pkl")
    print("[OK] Modèle global sauvegardé : xgb_global.pkl")

# 4. Entraînement segmenté
def train_segment_model():
    df = load_segment_tables()

    X = df[SEGMENT_NUM_FEATURES + SEGMENT_CAT_FEATURES]
    y = np.log1p(df["recettes"])

    print("Dimensions segmentées après dropna :", df.shape)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), SEGMENT_CAT_FEATURES),
            ("num", "passthrough", SEGMENT_NUM_FEATURES),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", build_xgb_model())
        ]
    )

    pipeline.fit(X, y)

    payload = {
        "pipeline": pipeline,
        "num_features": SEGMENT_NUM_FEATURES,
        "cat_features": SEGMENT_CAT_FEATURES,
        "target_transform": "log1p"
    }

    joblib.dump(payload, MODELS_DIR / "xgb_segment.pkl")
    print("[OK] Modèle segmenté sauvegardé : xgb_segment.pkl")


def main():
    print("\n=== ENTRAINEMENT XGBOOST ===\n")

    print("1) Entraînement du modèle global...")
    train_global_model()

    print("\n2) Entraînement du modèle segmenté...")
    train_segment_model()

    print("\n=== FIN ENTRAINEMENT XGBOOST ===")


if __name__ == "__main__":
    main()