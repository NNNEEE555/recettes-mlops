import pandas as pd
from sqlalchemy import create_engine

# connexion PostgreSQL
from db_connection import engine

# lire les données brutes
df = pd.read_sql("SELECT * FROM recettes_raw", engine)

# 1️: Correction colonne MOY_PAIMENT

df["moy_paiment"] = df["moy_paiment"].replace({
    "4": "TGR"
})

# 2️: Correction colonne produit

  # normaliser texte
df["produit"] = df["produit"].astype(str).str.strip()

  # certificat d'origine
df.loc[
    df["produit"].str.contains("certificat", case=False, na=False),
    "produit"
] = "Certificat d'origine"

df.loc[
    (df["produit"].isna()) | (df["produit"] == "") | (df["produit"] == "None"),
    "produit"
] = "Certificat d'origine"

  # location de salles
df.loc[
    df["produit"].str.contains("location de salles", case=False, na=False),
    "produit"
] = "Location de salles"

# 3️: Correction colonne Secteur

df["secteur"] = df["secteur"].astype(str).str.strip()

valeurs_commerce = [
"ALCommerce013",
"ALCommerce017",
"ALCommerce014",
"ALCommerce016",
"ALCommerce015",
"ALCommerce 18",
"ALCommerce",
"ALCommerce01",
"CommerceertIndustriefIndustrieCommerceat n 2",
"CommerceertIndustriefIndustrieCommerceat n 3",
"ALCommerce/05/2025",
"ALCommerce/04/2025"
]

df.loc[
    df["secteur"].isin(valeurs_commerce),
    "secteur"
] = "Commerce"

# 4️: supprimer doublons

df = df.drop_duplicates()

# 5: Imputer personne_physique pour produit = Location de salles


df.loc[
    (df["produit"] == "Location de salles") &
    ((df["personne_physique"].isna()) | (df["personne_physique"] == "")),
    "personne_physique"
] = "Particulier"


# 6: Imputer secteur

  # normaliser la colonne secteur
df["secteur"] = df["secteur"].astype(str).str.strip()

  # transformer les fausses valeurs vides en NA
df["secteur"] = df["secteur"].replace(["", "None", "nan", "NaN"], pd.NA)

  # cases vides -> Hors secteur
df.loc[
    (df["secteur"].isna()) | (df["secteur"] == ""),
    "secteur"
] = "Hors secteur"

  # valeurs comme 2024 ou 2025 -> Hors secteur
df.loc[
    df["secteur"].astype(str).str.fullmatch(r"\d{4}", na=False),
    "secteur"
] = "Hors secteur"

# 7: Imputer la référence produit
df["produit_ref"] = df["produit_ref"].fillna("Inconnu")

#  sauvegarder dataset propre

df.to_sql(
    "recettes_clean",
    engine,
    schema="public",
    if_exists="replace",
    index=False
)
# exporter dataset propre
df.to_excel(
    "data/processed/recettes_clean.xlsx",
    index=False
)

print("Nettoyage terminé")
print("Dimensions :", df.shape)

