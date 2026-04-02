from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

from db_connection import engine

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs" / "forecasts"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


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


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0

    if mask.sum() == 0:
        mape = np.nan
    else:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    return round(mae, 4), round(rmse, 4), round(mape, 4) if not np.isnan(mape) else None


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
        """
    }

    frames = []

    for segment_type, query in queries.items():
        df = pd.read_sql(query, engine)
        df["segment_type"] = segment_type
        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)
    df_all["date"] = pd.to_datetime(df_all["date"])
    df_all = df_all.sort_values(["segment_type", "segment_value", "date"])

    df_all = df_all.dropna(
        subset=SEGMENT_NUM_FEATURES + SEGMENT_CAT_FEATURES + ["recettes"]
    )

    return df_all


def evaluate_global():
    payload = joblib.load(MODELS_DIR / "xgb_global.pkl")
    model = payload["model"]
    target_transform = payload.get("target_transform", None)

    df = load_global_data()

    train_df = df.iloc[:-3].copy()
    test_df = df.iloc[-3:].copy()

    X_train = train_df[GLOBAL_FEATURES]
    X_test = test_df[GLOBAL_FEATURES]

    if target_transform == "log1p":
        y_train = np.log1p(train_df["recettes"])
    else:
        y_train = train_df["recettes"]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Retour à l'échelle réelle
    if target_transform == "log1p":
        y_pred = np.expm1(y_pred)

    # Graphique Réel vs Prédit
    plt.figure(figsize=(10, 5))
    plt.plot(test_df["date"], test_df["recettes"], label="Réel", marker="o")
    plt.plot(test_df["date"], y_pred, label="Prédit", marker="o")
    plt.title("Réel vs Prédit - Modèle Global XGBoost")
    plt.xlabel("Date")
    plt.ylabel("Recettes")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    mae, rmse, mape = compute_metrics(test_df["recettes"], y_pred)

    return [{
        "label": "xgb_global",
        "segment_type": None,
        "segment_value": None,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "mae": mae,
        "rmse": rmse,
        "mape": mape
    }]



def evaluate_segments():
    payload = joblib.load(MODELS_DIR / "xgb_segment.pkl")
    pipeline = payload["pipeline"]
    target_transform = payload.get("target_transform", None)

    df = load_segment_tables()
    results = []

    for (segment_type, segment_value), group in df.groupby(["segment_type", "segment_value"]):
        group = group.sort_values("date")

        if len(group) < 6:
            continue

        if len(group) >= 24:
            test_size = 3
        else:
            test_size = 1

        train_df = group.iloc[:-test_size].copy()
        test_df = group.iloc[-test_size:].copy()

        X_train = train_df[SEGMENT_NUM_FEATURES + SEGMENT_CAT_FEATURES]
        X_test = test_df[SEGMENT_NUM_FEATURES + SEGMENT_CAT_FEATURES]

        if target_transform == "log1p":
            y_train = np.log1p(train_df["recettes"])
        else:
            y_train = train_df["recettes"]

        y_test = test_df["recettes"]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Retour à l'échelle réelle
        if target_transform == "log1p":
            y_pred = np.expm1(y_pred)

        # Graphique Réel vs Prédit (SEGMENT)
        plt.figure(figsize=(8, 4))
        plt.plot(test_df["date"], y_test.values, label="Réel", marker="o")
        plt.plot(test_df["date"], y_pred, label="Prédit", marker="o")
        plt.title(f"{segment_type} - {segment_value}")
        plt.xlabel("Date")
        plt.ylabel("Recettes")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        mae, rmse, mape = compute_metrics(y_test, y_pred)

        results.append({
            "label": f"xgb_{segment_type}_{segment_value}",
            "segment_type": segment_type,
            "segment_value": segment_value,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "mae": mae,
            "rmse": rmse,
            "mape": mape
        })

    return results


def main():
    print("\n=== EVALUATION XGBOOST ===\n")

    results = []

    # 1) Evaluation globale
    global_results = evaluate_global()
    results.extend(global_results)

    # 2) Evaluation détaillée des segments
    segment_results = evaluate_segments()
    results.extend(segment_results)

    # Convertir en DataFrame pour créer les résumés
    results_df = pd.DataFrame(results)

    # Résumé global de tous les segments
    df_segments = results_df[results_df["label"] != "xgb_global"].copy()

    if not df_segments.empty:
        segments_global_row = {
            "label": "xgb_segments_global",
            "segment_type": "ALL",
            "segment_value": "ALL",
            "n_train": round(df_segments["n_train"].mean(), 2),
            "n_test": round(df_segments["n_test"].mean(), 2),
            "mae": round(df_segments["mae"].mean(), 4),
            "rmse": round(df_segments["rmse"].mean(), 4),
            "mape": round(df_segments["mape"].mean(), 4),
        }

        results = [global_results[0], segments_global_row] + segment_results

        # Résumé par type de segment
        segment_type_rows = []

        for seg_type in sorted(df_segments["segment_type"].dropna().unique()):
            df_type = df_segments[df_segments["segment_type"] == seg_type]

            row = {
                "label": f"xgb_{seg_type}_global",
                "segment_type": seg_type,
                "segment_value": "ALL",
                "n_train": round(df_type["n_train"].mean(), 2),
                "n_test": round(df_type["n_test"].mean(), 2),
                "mae": round(df_type["mae"].mean(), 4),
                "rmse": round(df_type["rmse"].mean(), 4),
                "mape": round(df_type["mape"].mean(), 4),
            }

            segment_type_rows.append(row)

        # ordre final : global → segments global → résumé par type → détails
        results = [global_results[0], segments_global_row] + segment_type_rows + segment_results

    # DataFrame final
    results_df = pd.DataFrame(results)

    csv_path = OUTPUTS_DIR / "xgboost_evaluation_metrics.csv"
    xlsx_path = OUTPUTS_DIR / "xgboost_evaluation_metrics.xlsx"

    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    results_df.to_excel(xlsx_path, index=False)

    print(results_df.head(30))
    print(f"\n[OK] CSV sauvegardé : {csv_path}")
    print(f"[OK] Excel sauvegardé : {xlsx_path}")
    print("\n=== FIN EVALUATION XGBOOST ===")


if __name__ == "__main__":
    main()