import os
import pandas as pd
from db_connection import engine

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

df["année_mois"] = df["date"].dt.to_period("M").astype(str)

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
recettes_mensuelles["date"] = pd.to_datetime(
    recettes_mensuelles["annee"].astype(str) + "-" +
    recettes_mensuelles["month"].astype(str) + "-01"
)

recettes_mensuelles = recettes_mensuelles.sort_values("date")

# 6. Variables temporelles pour la prévision
recettes_mensuelles["trimestre"] = recettes_mensuelles["date"].dt.quarter

# Recettes mois précédent
recettes_mensuelles["lag_1"] = recettes_mensuelles["recettes"].shift(1)

# Recettes 3 mois avant
recettes_mensuelles["lag_3"] = recettes_mensuelles["recettes"].shift(3)

# Moyenne mobile sur 3 mois
recettes_mensuelles["moyenne_mobile"] = (
    recettes_mensuelles["recettes"].rolling(window=3).mean()
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

recettes_produit["date"] = pd.to_datetime(
    recettes_produit["annee"].astype(str) + "-" +
    recettes_produit["month"].astype(str) + "-01"
)

recettes_produit = recettes_produit.sort_values(["produit", "date"])

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

recettes_secteur["date"] = pd.to_datetime(
    recettes_secteur["annee"].astype(str) + "-" +
    recettes_secteur["month"].astype(str) + "-01"
)

recettes_secteur = recettes_secteur.sort_values(["secteur", "date"])

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

recettes_antenne["date"] = pd.to_datetime(
    recettes_antenne["annee"].astype(str) + "-" +
    recettes_antenne["month"].astype(str) + "-01"
)

recettes_antenne = recettes_antenne.sort_values(["antenne", "date"])

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

recettes_adherent["date"] = pd.to_datetime(
    recettes_adherent["annee"].astype(str) + "-" +
    recettes_adherent["month"].astype(str) + "-01"
)

recettes_adherent = recettes_adherent.sort_values(["personne_physique", "date"])

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