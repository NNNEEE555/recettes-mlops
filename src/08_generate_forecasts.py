from pathlib import Path
import joblib
import pandas as pd

from db_connection import engine


BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs" / "forecasts"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

FORECAST_MONTHS = 12


GLOBAL_FEATURES = [
    "annee",
    "month",
    "trimestre",
    "lag_1",
    "lag_3",
    "moyenne_mobile",
    "transactions",
]

SEGMENT_NUM_FEATURES = [
    "annee",
    "month",
    "trimestre",
    "lag_1",
    "lag_3",
    "moyenne_mobile",
    "transactions",
]

SEGMENT_CAT_FEATURES = [
    "segment_type",
    "segment_value",
]


def save_outputs(df: pd.DataFrame, base_filename: str):
    csv_path = OUTPUTS_DIR / f"{base_filename}.csv"
    xlsx_path = OUTPUTS_DIR / f"{base_filename}.xlsx"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_excel(xlsx_path, index=False)

    print(f"[OK] Sauvegardé : {csv_path}")
    print(f"[OK] Sauvegardé : {xlsx_path}")


def load_global_history():
    query = """
        SELECT annee, month, trimestre, lag_1, lag_3, moyenne_mobile, transactions, recettes, date
        FROM recettes_mensuelles
        ORDER BY date
    """
    df = pd.read_sql(query, engine)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_segment_tables():
    queries = {
        "secteur": """
            SELECT secteur AS segment_value, annee, month, recettes, transactions, date
            FROM recettes_secteur
        """,
        "produit": """
            SELECT produit AS segment_value, annee, month, recettes, transactions, date
            FROM recettes_produit
        """,
        "antenne": """
            SELECT antenne AS segment_value, annee, month, recettes, transactions, date
            FROM recettes_antenne
        """,
        "adherent": """
            SELECT personne_physique AS segment_value, annee, month, recettes, transactions, date
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
    df_all = df_all.sort_values(["segment_type", "segment_value", "date"]).reset_index(drop=True)
    return df_all


def forecast_global():
    payload = joblib.load(MODELS_DIR / "xgb_global.pkl")
    model = payload["model"]

    history = load_global_history().copy()
    forecasts = []

    for _ in range(FORECAST_MONTHS):
        last_date = history["date"].iloc[-1]
        next_date = last_date + pd.offsets.MonthEnd(1)

        lag_1 = history["recettes"].iloc[-1]
        lag_3 = history["recettes"].iloc[-3] if len(history) >= 3 else lag_1
        moyenne_mobile = history["recettes"].tail(3).mean()
        transactions = history["transactions"].tail(3).mean()

        row = pd.DataFrame([{
            "annee": next_date.year,
            "month": next_date.month,
            "trimestre": ((next_date.month - 1) // 3) + 1,
            "lag_1": lag_1,
            "lag_3": lag_3,
            "moyenne_mobile": moyenne_mobile,
            "transactions": transactions
        }])

        pred = float(model.predict(row)[0])

        forecasts.append({
            "date": next_date,
            "yhat": pred
        })

        history = pd.concat([
            history,
            pd.DataFrame([{
                "annee": next_date.year,
                "month": next_date.month,
                "trimestre": ((next_date.month - 1) // 3) + 1,
                "lag_1": lag_1,
                "lag_3": lag_3,
                "moyenne_mobile": moyenne_mobile,
                "transactions": transactions,
                "recettes": pred,
                "date": next_date
            }])
        ], ignore_index=True)

    df_forecast = pd.DataFrame(forecasts)
    save_outputs(df_forecast, "xgb_forecast_global")

    annual = df_forecast.copy()
    annual["annee"] = pd.to_datetime(annual["date"]).dt.year
    annual = annual.groupby("annee", as_index=False)["yhat"].sum()
    save_outputs(annual, "xgb_forecast_annual")


def forecast_segments():
    payload = joblib.load(MODELS_DIR / "xgb_segment.pkl")
    pipeline = payload["pipeline"]

    df = load_segment_tables()
    forecasts = []

    for (segment_type, segment_value), group in df.groupby(["segment_type", "segment_value"]):
        group = group.sort_values("date").reset_index(drop=True)

        if len(group) < 4:
            continue

        local_history = group.copy()
        local_history["trimestre"] = ((local_history["month"] - 1) // 3) + 1

        for _ in range(FORECAST_MONTHS):
            last_date = local_history["date"].iloc[-1]
            next_date = last_date + pd.offsets.MonthEnd(1)

            lag_1 = local_history["recettes"].iloc[-1]
            lag_3 = local_history["recettes"].iloc[-3] if len(local_history) >= 3 else lag_1
            moyenne_mobile = local_history["recettes"].tail(3).mean()
            transactions = local_history["transactions"].tail(3).mean()

            row = pd.DataFrame([{
                "segment_type": segment_type,
                "segment_value": segment_value,
                "annee": next_date.year,
                "month": next_date.month,
                "trimestre": ((next_date.month - 1) // 3) + 1,
                "lag_1": lag_1,
                "lag_3": lag_3,
                "moyenne_mobile": moyenne_mobile,
                "transactions": transactions
            }])

            pred = float(pipeline.predict(row)[0])

            forecasts.append({
                "segment_type": segment_type,
                "segment_value": segment_value,
                "date": next_date,
                "yhat": pred
            })

            local_history = pd.concat([
                local_history,
                pd.DataFrame([{
                    "segment_type": segment_type,
                    "segment_value": segment_value,
                    "annee": next_date.year,
                    "month": next_date.month,
                    "trimestre": ((next_date.month - 1) // 3) + 1,
                    "transactions": transactions,
                    "recettes": pred,
                    "date": next_date
                }])
            ], ignore_index=True)

    result = pd.DataFrame(forecasts)
    save_outputs(result, "xgb_forecast_segments")


def main():
    print("\n=== GENERATION DES PREVISIONS XGBOOST ===\n")

    print("1) Prévision globale mensuelle...")
    forecast_global()

    print("\n2) Prévision annuelle...")
    print("   -> calculée par somme des prévisions mensuelles globales")

    print("\n3) Prévisions par segments...")
    forecast_segments()

    print("\n=== FIN GENERATION XGBOOST ===")


if __name__ == "__main__":
    main()