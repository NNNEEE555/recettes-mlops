import math
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.db_connection import engine
from api.schemas import GlobalForecastInput, SegmentForecastInput
from api.model_loader import global_model, segment_model
def read_sql_safe(query: str, params=None):
    try:
        with engine.connect() as conn:
            return pd.read_sql(query, conn, params=params)
    except Exception as e:
        engine.dispose()
        raise HTTPException(
            status_code=500,
            detail=f"Erreur base de données : {str(e)}"
        )

app = FastAPI(
    title="API de prévision des recettes",
    description="API pour prédire les recettes globales et segmentées",
    version="2.0.0"
)

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
    "moyenne_mobile",
    "transactions",
]

SEGMENT_CAT_FEATURES = [
    "segment_type",
    "segment_value",
]


@app.get("/health")
def health_check():
    return {"status": "ok"}


def parse_target_month(date_str: str) -> pd.Timestamp:
    try:
        return pd.to_datetime(date_str, format="%Y-%m")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Le format de date doit être YYYY-MM, par exemple 2026-01"
        )


def load_global_history() -> pd.DataFrame:
    query = """
        SELECT annee, month, trimestre, lag_1, lag_3, lag_6, moyenne_mobile,
               transactions, transactions_lag_1, transactions_lag_3,
               transactions_rolling_3, transactions_diff,
               month_sin, month_cos, trend, recettes, date
        FROM recettes_mensuelles
        ORDER BY date
    """
    df = read_sql_safe(query)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_segment_history(segment_type: str, segment_value: str) -> pd.DataFrame:
    if segment_type == "secteur":
        query = """
            SELECT secteur AS segment_value, annee, month, recettes, transactions, date
            FROM recettes_secteur
            WHERE secteur = %(segment_value)s
            ORDER BY date
        """
    elif segment_type == "produit":
        query = """
            SELECT produit AS segment_value, annee, month, recettes, transactions, date
            FROM recettes_produit
            WHERE produit = %(segment_value)s
            ORDER BY date
        """
    elif segment_type == "antenne":
        query = """
            SELECT antenne AS segment_value, annee, month, recettes, transactions, date
            FROM recettes_antenne
            WHERE antenne = %(segment_value)s
            ORDER BY date
        """
    elif segment_type == "adherent":
        query = """
            SELECT personne_physique AS segment_value, annee, month, recettes, transactions, date
            FROM recettes_adherent
            WHERE personne_physique = %(segment_value)s
            ORDER BY date
        """
    else:
        raise HTTPException(
            status_code=400,
            detail="segment_type doit être : secteur, produit, antenne ou adherent"
        )

    df = read_sql_safe(
    query,
    params={"segment_value": segment_value}
)
    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Aucune donnée trouvée pour {segment_type} = {segment_value}"
        )

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["trimestre"] = ((df["month"] - 1) // 3) + 1
    return df


def build_global_features(history: pd.DataFrame, next_date: pd.Timestamp) -> pd.DataFrame:
    if len(history) < 6:
        raise HTTPException(
            status_code=400,
            detail="Historique insuffisant pour calculer les variables globales (minimum 6 mois)."
        )

    recettes_series = history["recettes"]
    transactions_series = history["transactions"]

    lag_1 = float(recettes_series.iloc[-1])
    lag_3 = float(recettes_series.iloc[-3])
    lag_6 = float(recettes_series.iloc[-6])
    moyenne_mobile = float(recettes_series.tail(3).mean())

    transactions = float(transactions_series.tail(3).mean())
    transactions_lag_1 = float(transactions_series.iloc[-1])
    transactions_lag_3 = float(transactions_series.iloc[-3])
    transactions_rolling_3 = float(transactions_series.tail(3).mean())
    transactions_diff = float(transactions_series.iloc[-1] - transactions_series.iloc[-2])

    month_value = int(next_date.month)
    trend = len(history) + 1

    row = pd.DataFrame([{
        "annee": int(next_date.year),
        "month": month_value,
        "trimestre": ((month_value - 1) // 3) + 1,
        "lag_1": lag_1,
        "lag_3": lag_3,
        "lag_6": lag_6,
        "moyenne_mobile": moyenne_mobile,
        "transactions": transactions,
        "transactions_lag_1": transactions_lag_1,
        "transactions_lag_3": transactions_lag_3,
        "transactions_rolling_3": transactions_rolling_3,
        "transactions_diff": transactions_diff,
        "month_sin": math.sin(2 * math.pi * month_value / 12),
        "month_cos": math.cos(2 * math.pi * month_value / 12),
        "trend": trend,
    }])

    return row[GLOBAL_FEATURES]


def build_segment_features(
    history: pd.DataFrame,
    next_date: pd.Timestamp,
    segment_type: str,
    segment_value: str
) -> pd.DataFrame:

    if len(history) < 6:
        raise HTTPException(
            status_code=400,
            detail="Historique insuffisant pour les features segment (min 6 mois)."
        )

    recettes_series = history["recettes"]
    transactions_series = history["transactions"]

    month_value = int(next_date.month)

    row = pd.DataFrame([{
        "segment_type": segment_type,
        "segment_value": segment_value,

        "annee": int(next_date.year),
        "month": month_value,
        "trimestre": ((month_value - 1) // 3) + 1,

        "lag_1": float(recettes_series.iloc[-1]),
        "lag_3": float(recettes_series.iloc[-3]),
        "lag_6": float(recettes_series.iloc[-6]),

        "moyenne_mobile": float(recettes_series.tail(3).mean()),

        "transactions": float(transactions_series.tail(3).mean()),
        "transactions_lag_1": float(transactions_series.iloc[-1]),
        "transactions_lag_3": float(transactions_series.iloc[-3]),
        "transactions_rolling_3": float(transactions_series.tail(3).mean()),
        "transactions_diff": float(transactions_series.iloc[-1] - transactions_series.iloc[-2]),

        "month_sin": math.sin(2 * math.pi * month_value / 12),
        "month_cos": math.cos(2 * math.pi * month_value / 12),

        "trend": len(history) + 1
    }])

    return row

@app.post("/predict/global/auto")
def predict_global_auto(payload: GlobalForecastInput):
    target_date = parse_target_month(payload.date)
    history = load_global_history()

    last_known_date = history["date"].max()
    if target_date <= last_known_date:
        raise HTTPException(
            status_code=400,
            detail=f"La date demandée ({payload.date}) doit être postérieure à la dernière date connue ({last_known_date.strftime('%Y-%m')})."
        )

    local_history = history.copy()

    while local_history["date"].max() < target_date:
        next_date = local_history["date"].max() + pd.offsets.MonthBegin(1)

        row = build_global_features(local_history, next_date)
        pred = float(global_model.predict(row)[0])

        new_row = row.copy()
        new_row["recettes"] = pred
        new_row["date"] = next_date

        local_history = pd.concat([local_history, new_row], ignore_index=True)

    final_row = local_history.loc[local_history["date"] == target_date].iloc[0]

    return {
        "type": "global_auto",
        "target_date": payload.date,
        "prediction_recettes": round(float(final_row["recettes"]), 2)
    }


@app.post("/predict/segment/auto")
def predict_segment_auto(payload: SegmentForecastInput):
    target_date = parse_target_month(payload.date)
    history = load_segment_history(payload.segment_type, payload.segment_value)

    last_known_date = history["date"].max()
    if target_date <= last_known_date:
        raise HTTPException(
            status_code=400,
            detail=f"La date demandée ({payload.date}) doit être postérieure à la dernière date connue ({last_known_date.strftime('%Y-%m')}) pour ce segment."
        )

    local_history = history.copy()

    while local_history["date"].max() < target_date:
        next_date = local_history["date"].max() + pd.offsets.MonthBegin(1)

        row = build_segment_features(
            local_history,
            next_date,
            payload.segment_type,
            payload.segment_value
        )

        pred = float(segment_model.predict(row)[0])

        new_row = pd.DataFrame([{
            "segment_value": payload.segment_value,
            "annee": int(next_date.year),
            "month": int(next_date.month),
            "recettes": pred,
            "transactions": float(local_history["transactions"].tail(3).mean()),
            "date": next_date,
            "trimestre": ((next_date.month - 1) // 3) + 1,
        }])

        local_history = pd.concat([local_history, new_row], ignore_index=True)

    final_row = local_history.loc[local_history["date"] == target_date].iloc[0]

    return {
        "type": "segment_auto",
        "segment_type": payload.segment_type,
        "segment_value": payload.segment_value,
        "target_date": payload.date,
        "prediction_recettes": round(float(final_row["recettes"]), 2)
    }
@app.get("/")
def root():
    return {"message": "API Recettes fonctionne"}