import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from db_connection import engine

# Charger les données propres
df = pd.read_sql("SELECT * FROM recettes_clean", engine)

print("Dimensions du dataset :", df.shape)

# 1. Evolution annuelle des recettes

recettes_annuelles = df.groupby("annee").agg(
    recettes_totales=("montant", "sum"),
    transactions=("id_vente", "count")
)

# taux de croissance
recettes_annuelles["croissance_%"] = recettes_annuelles["recettes_totales"].pct_change() * 100

print("\nEvolution annuelle des recettes")
print(recettes_annuelles)

plt.figure(figsize=(8,5))
plt.plot(recettes_annuelles.index, recettes_annuelles["recettes_totales"], marker="o")
plt.title("Evolution annuelle des recettes")
plt.xlabel("Année")
plt.ylabel("Recettes totales")
plt.grid(True)
plt.show()

# 2. Analyse de la saisonnalité mensuelle

recettes_mensuelles = df.groupby("month").agg(
    recettes_totales=("montant", "sum"),
    transactions=("id_vente", "count")
)

print("\nRecettes par mois")
print(recettes_mensuelles)

mois_max = recettes_mensuelles["recettes_totales"].idxmax()
mois_min = recettes_mensuelles["recettes_totales"].idxmin()

print("\nMois avec recettes maximales :", mois_max)
print("Mois avec recettes minimales :", mois_min)

plt.figure(figsize=(8,5))
plt.plot(recettes_mensuelles.index, recettes_mensuelles["recettes_totales"], marker="o")
plt.title("Saisonnalité mensuelle des recettes")
plt.xlabel("Mois")
plt.ylabel("Recettes")
plt.grid(True)
plt.show()

# 3. Analyse par type d'adhérent

recettes_adherent = df.groupby("personne_physique").agg(
    recettes_totales=("montant", "sum"),
    transactions=("id_vente", "count")
)

recettes_adherent["part_%"] = (
    recettes_adherent["recettes_totales"] /
    recettes_adherent["recettes_totales"].sum()
) * 100

print("\nAnalyse par type d'adhérent")
print(recettes_adherent)

plt.figure(figsize=(6,6))
plt.pie(
    recettes_adherent["recettes_totales"],
    labels=recettes_adherent.index,
    autopct="%1.1f%%"
)
plt.title("Part des recettes par type d'adhérent")
plt.show()

# 4. Analyse par secteur

recettes_secteur = df.groupby("secteur").agg(
    recettes_totales=("montant", "sum"),
    transactions=("id_vente", "count")
)

recettes_secteur["part_%"] = (
    recettes_secteur["recettes_totales"] /
    recettes_secteur["recettes_totales"].sum()
) * 100

print("\nAnalyse par secteur")
print(recettes_secteur)

plt.figure(figsize=(8,5))
sns.barplot(
    x=recettes_secteur.index,
    y=recettes_secteur["recettes_totales"]
)
plt.title("Recettes par secteur")
plt.xlabel("Secteur")
plt.ylabel("Recettes totales")
plt.show()

# 5. Analyse par produit

recettes_produit = df.groupby("produit").agg(
    recettes_totales=("montant", "sum"),
    ventes=("id_vente", "count")
)

recettes_produit["part_%"] = (
    recettes_produit["recettes_totales"] /
    recettes_produit["recettes_totales"].sum()
) * 100

print("\nAnalyse par produit")
print(recettes_produit)

plt.figure(figsize=(8,5))
sns.barplot(
    x=recettes_produit.index,
    y=recettes_produit["recettes_totales"]
)
plt.title("Recettes par produit")
plt.xlabel("Produit")
plt.ylabel("Recettes totales")
plt.xticks(rotation=45)
plt.show()

# 6. Analyse par antenne

recettes_antenne = df.groupby("antenne").agg(
    recettes_totales=("montant", "sum"),
    transactions=("id_vente", "count")
)

recettes_antenne["part_%"] = (
    recettes_antenne["recettes_totales"] /
    recettes_antenne["recettes_totales"].sum()
) * 100

print("\nAnalyse par antenne")
print(recettes_antenne)

plt.figure(figsize=(8,5))
sns.barplot(
    x=recettes_antenne.index,
    y=recettes_antenne["recettes_totales"]
)
plt.title("Recettes par antenne")
plt.xlabel("Antenne")
plt.ylabel("Recettes totales")
plt.xticks(rotation=45)
plt.show()