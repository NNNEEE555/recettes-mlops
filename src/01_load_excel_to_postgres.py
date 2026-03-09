import pandas as pd
from sqlalchemy import create_engine

# 1) Chemin vers le fichier Excel versionné par DVC
EXCEL_PATH = r"data/raw/ccistta_data_unlocked.xlsx"

# 2) Paramètres de connexion PostgreSQL
HOST = "localhost"
PORT = 5432
DB   = "ccistta_mlops"
USER = "postgres"
PWD  = "postgres123"   

# 3) Créer l'engine (connexion)
engine = create_engine(
    f"postgresql+psycopg2://{USER}:{PWD}@{HOST}:{PORT}/{DB}"
)

# 4) Lire l'Excel
df = pd.read_excel(EXCEL_PATH, engine="openpyxl")

# 5) Normaliser les noms de colonnes 
df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

# 6) Importer dans PostgreSQL 
df.to_sql("recettes_raw", engine, schema="public", if_exists="replace", index=False)

print("Import terminé : public.recettes_raw")
print("Lignes:", len(df), "| Colonnes:", len(df.columns))