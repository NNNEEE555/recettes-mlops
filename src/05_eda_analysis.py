import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from db_connection import engine

# Style simple
sns.set_theme(style="whitegrid")

# 1. Charger les tables déjà préparées
df_recettes_features = pd.read_sql("SELECT * FROM recettes_features", engine)
df_recettes_mensuelles = pd.read_sql("SELECT * FROM recettes_mensuelles", engine)
df_recettes_produit = pd.read_sql("SELECT * FROM recettes_produit", engine)
df_recettes_secteur = pd.read_sql("SELECT * FROM recettes_secteur", engine)
df_recettes_antenne = pd.read_sql("SELECT * FROM recettes_antenne", engine)
df_recettes_adherent = pd.read_sql("SELECT * FROM recettes_adherent", engine)

# 3. Evolution mensuelle des recettes
plt.figure(figsize=(10, 5))
plt.plot(df_recettes_mensuelles["date"], df_recettes_mensuelles["recettes"], marker="o")
plt.title("Évolution mensuelle des recettes")
plt.xlabel("Date")
plt.ylabel("Recettes")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. Evolution annuelle des recettes
recettes_annuelles = df_recettes_mensuelles.groupby("annee", as_index=False).agg(
    recettes_totales=("recettes", "sum"),
    transactions_totales=("transactions", "sum")
)

recettes_annuelles["croissance_%"] = (
    recettes_annuelles["recettes_totales"].pct_change() * 100
)

print("\nEvolution annuelle des recettes")
print(recettes_annuelles)

plt.figure(figsize=(8, 5))
sns.barplot(data=recettes_annuelles, x="annee", y="recettes_totales")
plt.title("Recettes totales par année")
plt.xlabel("Année")
plt.ylabel("Recettes totales")
plt.tight_layout()
plt.show()

# 5. Saisonnalité mensuelle
saisonnalite = df_recettes_mensuelles.groupby("month", as_index=False).agg(
    recettes_totales=("recettes", "sum"),
    transactions_totales=("transactions", "sum")
)

mois_max = saisonnalite.loc[saisonnalite["recettes_totales"].idxmax(), "month"]
mois_min = saisonnalite.loc[saisonnalite["recettes_totales"].idxmin(), "month"]

print("\nSaisonnalité mensuelle")
print(saisonnalite)
print(f"\nMois avec recettes maximales : {mois_max}")
print(f"Mois avec recettes minimales : {mois_min}")

plt.figure(figsize=(8, 5))
sns.lineplot(data=saisonnalite, x="month", y="recettes_totales", marker="o")
plt.title("Saisonnalité mensuelle des recettes")
plt.xlabel("Mois")
plt.ylabel("Recettes totales")
plt.tight_layout()
plt.show()

# 6. Analyse par produit
analyse_produit = df_recettes_produit.groupby(["produit", "annee"], as_index=False).agg(
    recettes_totales=("recettes", "sum")
)

analyse_produit["part_%"] = (
    analyse_produit["recettes_totales"] /
    analyse_produit["recettes_totales"].sum()
) * 100

analyse_produit = analyse_produit.sort_values("recettes_totales", ascending=False)

print("\nAnalyse par produit")
print(analyse_produit)

plt.figure(figsize=(10,5))
sns.barplot(
    data=analyse_produit,
    x="produit",
    y="recettes_totales",
    hue="annee"
)
plt.title("Recettes par produit et par année")
plt.xticks(rotation=45)
plt.legend(title="Année")
plt.tight_layout()
plt.show()

# 7. Analyse par secteur
analyse_secteur = df_recettes_secteur.groupby(["secteur", "annee"], as_index=False).agg(
    recettes_totales=("recettes", "sum")
)

analyse_secteur["part_%"] = (
    analyse_secteur["recettes_totales"] /
    analyse_secteur["recettes_totales"].sum()
) * 100

analyse_secteur = analyse_secteur.sort_values("recettes_totales", ascending=False)

print("\nAnalyse par secteur")
print(analyse_secteur)

analyse_secteur = df_recettes_secteur.groupby(["secteur", "annee"], as_index=False).agg(
    recettes_totales=("recettes", "sum")
)
plt.figure(figsize=(10,5))

sns.barplot(
    data=analyse_secteur,
    x="secteur",
    y="recettes_totales",
    hue="annee"
)

plt.title("Recettes par secteur et par année")
plt.xticks(rotation=45)
plt.legend(title="Année")
plt.tight_layout()

plt.show()


# 8. Analyse par antenne
analyse_antenne = df_recettes_antenne.groupby(["antenne", "annee"], as_index=False).agg(
    recettes_totales=("recettes", "sum")
)


analyse_antenne["part_%"] = (
    analyse_antenne["recettes_totales"] /
    analyse_antenne["recettes_totales"].sum()
) * 100

analyse_antenne = analyse_antenne.sort_values("recettes_totales", ascending=False)

print("\nAnalyse par antenne")
print(analyse_antenne)

plt.figure(figsize=(10,5))
sns.barplot(
    data=analyse_antenne,
    x="antenne",
    y="recettes_totales",
    hue="annee"
)
plt.title("Recettes par antenne et par année")
plt.xticks(rotation=45)
plt.legend(title="Année")
plt.tight_layout()
plt.show()

# 9. Analyse par type d’adhérent
analyse_adherent = df_recettes_adherent.groupby("personne_physique", as_index=False).agg(
    recettes_totales=("recettes", "sum"),
    transactions_totales=("transactions", "sum")
)

analyse_adherent["part_%"] = (
    analyse_adherent["recettes_totales"] /
    analyse_adherent["recettes_totales"].sum()
) * 100

print("\nAnalyse par type d’adhérent")
print(analyse_adherent)

plt.figure(figsize=(6, 6))
plt.pie(
    analyse_adherent["recettes_totales"],
    labels=analyse_adherent["personne_physique"],
    autopct="%1.1f%%"
)
plt.title("Part des recettes par type d’adhérent")
plt.tight_layout()
plt.show()







print("\nEDA terminé avec succès.")