import os
import numpy as np
import pandas as pd
from db_connection import engine

# Fonctions de feature engineering
def add_time_features(df, date_col="date", month_col="month"):
    df = df.copy()
    df["trimestre"] = df[date_col].dt.quarter
    df["month_sin"] = np.sin(2 * np.pi * df[month_col] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df[month_col] / 12)
    return df


def add_lag_features(df, target_col="recettes", group_cols=None):
    df = df.copy()

    if group_cols is None:
        df = df.sort_values("date").reset_index(drop=True)

        # ===== RECETTES =====
        df["lag_1"] = df[target_col].shift(1)
        df["lag_3"] = df[target_col].shift(3)
        df["lag_6"] = df[target_col].shift(6)

        df["moyenne_mobile"] = df[target_col].shift(1).rolling(3).mean()
        df["rolling_6"] = df[target_col].shift(1).rolling(6).mean()

        # ===== TRANSACTIONS =====
        df["transactions_lag_1"] = df["transactions"].shift(1)
        df["transactions_lag_3"] = df["transactions"].shift(3)
        df["transactions_rolling_3"] = df["transactions"].shift(1).rolling(3).mean()

        # Variation (très utile)
        df["transactions_diff"] = df["transactions"] - df["transactions"].shift(1)

        # ===== TREND =====
        df["trend"] = np.arange(len(df))

    else:
        df = df.sort_values(group_cols + ["date"]).reset_index(drop=True)

        # ===== RECETTES =====
        df["lag_1"] = df.groupby(group_cols)[target_col].shift(1)
        df["lag_3"] = df.groupby(group_cols)[target_col].shift(3)
        df["lag_6"] = df.groupby(group_cols)[target_col].shift(6)

        df["moyenne_mobile"] = (
            df.groupby(group_cols)[target_col]
            .transform(lambda s: s.shift(1).rolling(3).mean())
        )

        df["rolling_6"] = (
            df.groupby(group_cols)[target_col]
            .transform(lambda s: s.shift(1).rolling(6).mean())
        )

        # ===== TRANSACTIONS =====
        df["transactions_lag_1"] = df.groupby(group_cols)["transactions"].shift(1)
        df["transactions_lag_3"] = df.groupby(group_cols)["transactions"].shift(3)

        df["transactions_rolling_3"] = (
            df.groupby(group_cols)["transactions"]
            .transform(lambda s: s.shift(1).rolling(3).mean())
        )

        df["transactions_diff"] = (
            df.groupby(group_cols)["transactions"].diff()
        )

        # ===== TREND =====
        df["trend"] = df.groupby(group_cols).cumcount()

    return df

# 1. Charger les données propres
df = pd.read_sql("SELECT * FROM recettes_clean", engine)

print("Dimensions initiales :", df.shape)

# 2. Créer la date complète
df["date"] = pd.to_datetime(
    df["annee"].astype(str) + "-" +
    df["month"].astype(str) + "-" +
    df["jour"].astype(str),
    errors="coerce"
)

# Trier chronologiquement
df = df.sort_values("date")

# 3. Créer trimestre et année-mois
df["trimestre"] = df["date"].dt.quarter

df["année_mois"] = df["date"].dt.to_period("M").dt.to_timestamp("M")

# 4. Sauvegarder le dataset enrichi transactionnel
df.to_sql(
    "recettes_features",
    engine,
    schema="public",
    if_exists="replace",
    index=False
)

df_excel = df.copy()
df_excel["date"] = df_excel["date"].dt.date

df_excel.to_excel(
    "data/processed/recettes_features.xlsx",
    index=False
)

print("recettes_features créé")

# 5. Recettes mensuelles
recettes_mensuelles = df.groupby(
    ["annee", "month", "année_mois"],
    as_index=False
).agg(
    recettes=("montant", "sum"),
    transactions=("id_vente", "count")
)

# Créer une vraie date mensuelle
recettes_mensuelles["date"] = (
    pd.to_datetime(
        recettes_mensuelles["annee"].astype(str) + "-" +
        recettes_mensuelles["month"].astype(str) + "-01"
    ) + pd.offsets.MonthEnd(0)
)
recettes_mensuelles["année_mois"] = pd.to_datetime(recettes_mensuelles["année_mois"])
recettes_mensuelles = recettes_mensuelles.sort_values("date")

# 6. Variables temporelles pour la prévision
recettes_mensuelles = add_time_features(
    recettes_mensuelles,
    date_col="date",
    month_col="month"
)

recettes_mensuelles = add_lag_features(
    recettes_mensuelles,
    target_col="recettes",
    group_cols=None
)

# 7. Sauvegarder recettes mensuelles
recettes_mensuelles.to_sql(
    "recettes_mensuelles",
    engine,
    schema="public",
    if_exists="replace",
    index=False
)

recettes_mensuelles_excel = recettes_mensuelles.copy()
recettes_mensuelles_excel["date"] = recettes_mensuelles_excel["date"].dt.date

recettes_mensuelles_excel.to_excel(
    "data/processed/recettes_mensuelles.xlsx",
    index=False
)

print("recettes_mensuelles créé")

# 8. Recettes par produit
recettes_produit = df.groupby(
    ["produit", "annee", "month", "année_mois"],
    as_index=False
).agg(
    recettes=("montant", "sum"),
    transactions=("id_vente", "count")
)

recettes_produit["date"] = (
    pd.to_datetime(
        recettes_produit["annee"].astype(str) + "-" +
        recettes_produit["month"].astype(str) + "-01"
    ) + pd.offsets.MonthEnd(0)
)
recettes_produit["année_mois"] = pd.to_datetime(recettes_produit["année_mois"])
recettes_produit = recettes_produit.sort_values(["produit", "date"])
recettes_produit = add_time_features(
    recettes_produit,
    date_col="date",
    month_col="month"
)

recettes_produit = add_lag_features(
    recettes_produit,
    target_col="recettes",
    group_cols=["produit"]
)
recettes_produit.to_sql(
    "recettes_produit",
    engine,
    schema="public",
    if_exists="replace",
    index=False
)

recettes_produit_excel = recettes_produit.copy()
recettes_produit_excel["date"] = recettes_produit_excel["date"].dt.date

recettes_produit_excel.to_excel(
    "data/processed/recettes_produit.xlsx",
    index=False
)

print("recettes_produit créé")

# 9. Recettes par secteur
recettes_secteur = df.groupby(
    ["secteur", "annee", "month", "année_mois"],
    as_index=False
).agg(
    recettes=("montant", "sum"),
    transactions=("id_vente", "count")
)

recettes_secteur["date"] = (
    pd.to_datetime(
        recettes_secteur["annee"].astype(str) + "-" +
        recettes_secteur["month"].astype(str) + "-01"
    ) + pd.offsets.MonthEnd(0)
)
recettes_secteur["année_mois"] = pd.to_datetime(recettes_secteur["année_mois"])
recettes_secteur = recettes_secteur.sort_values(["secteur", "date"])
recettes_secteur = add_time_features(
    recettes_secteur,
    date_col="date",
    month_col="month"
)

recettes_secteur = add_lag_features(
    recettes_secteur,
    target_col="recettes",
    group_cols=["secteur"]
)
recettes_secteur.to_sql(
    "recettes_secteur",
    engine,
    schema="public",
    if_exists="replace",
    index=False
)

recettes_secteur_excel = recettes_secteur.copy()
recettes_secteur_excel["date"] = recettes_secteur_excel["date"].dt.date

recettes_secteur_excel.to_excel(
    "data/processed/recettes_secteur.xlsx",
    index=False
)

print("recettes_secteur créé")

# 10. Recettes par antenne
recettes_antenne = df.groupby(
    ["antenne", "annee", "month", "année_mois"],
    as_index=False
).agg(
    recettes=("montant", "sum"),
    transactions=("id_vente", "count")
)

recettes_antenne["date"] = (
    pd.to_datetime(
        recettes_antenne["annee"].astype(str) + "-" +
        recettes_antenne["month"].astype(str) + "-01"
    ) + pd.offsets.MonthEnd(0)
)
recettes_antenne["année_mois"] = pd.to_datetime(recettes_antenne["année_mois"])
recettes_antenne = recettes_antenne.sort_values(["antenne", "date"])
recettes_antenne = add_time_features(
    recettes_antenne,
    date_col="date",
    month_col="month"
)

recettes_antenne = add_lag_features(
    recettes_antenne,
    target_col="recettes",
    group_cols=["antenne"]
)
recettes_antenne.to_sql(
    "recettes_antenne",
    engine,
    schema="public",
    if_exists="replace",
    index=False
)

recettes_antenne_excel = recettes_antenne.copy()
recettes_antenne_excel["date"] = recettes_antenne_excel["date"].dt.date

recettes_antenne_excel.to_excel(
    "data/processed/recettes_antenne.xlsx",
    index=False
)

print("recettes_antenne créé")

# 11. Recettes par type d’adhérent
recettes_adherent = df.groupby(
    ["personne_physique", "annee", "month", "année_mois"],
    as_index=False
).agg(
    recettes=("montant", "sum"),
    transactions=("id_vente", "count")
)

recettes_adherent["date"] = (
    pd.to_datetime(
        recettes_adherent["annee"].astype(str) + "-" +
        recettes_adherent["month"].astype(str) + "-01"
    ) + pd.offsets.MonthEnd(0)
)
recettes_adherent["année_mois"] = pd.to_datetime(recettes_adherent["année_mois"])
recettes_adherent = recettes_adherent.sort_values(["personne_physique", "date"])
recettes_adherent = add_time_features(
    recettes_adherent,
    date_col="date",
    month_col="month"
)

recettes_adherent = add_lag_features(
    recettes_adherent,
    target_col="recettes",
    group_cols=["personne_physique"]
)
recettes_adherent.to_sql(
    "recettes_adherent",
    engine,
    schema="public",
    if_exists="replace",
    index=False
)

recettes_adherent_excel = recettes_adherent.copy()
recettes_adherent_excel["date"] = recettes_adherent_excel["date"].dt.date

recettes_adherent_excel.to_excel(
    "data/processed/recettes_adherent.xlsx",
    index=False
)

print("recettes_adherent créé")

print("\n Phase Feature Engineering terminée avec succès.")