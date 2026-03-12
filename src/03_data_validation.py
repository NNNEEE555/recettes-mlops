import pandas as pd
from sqlalchemy import create_engine

# connexion à la base
from db_connection import engine

# charger les données nettoyées
df = pd.read_sql("SELECT * FROM recettes_clean", engine)

print("Dimensions du dataset :", df.shape)

# Vérifier les valeurs manquantes
print("\nValeurs manquantes par colonne :")
print(df.isna().sum())

# Vérifier les catégories
print("\nProduits uniques :")
print(df["produit"].value_counts())

print("\nSecteurs uniques :")
print(df["secteur"].value_counts())

# Vérifier les doublons
print("\nNombre de doublons :", df.duplicated().sum())

# Vérifier les types de colonnes
print("\nTypes de données :")
print(df.dtypes)

# Statistiques sur les valeurs numériques
print("\nStatistiques numériques :")
print(df.describe())