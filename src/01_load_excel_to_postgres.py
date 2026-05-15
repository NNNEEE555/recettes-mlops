import pandas as pd
from pathlib import Path
from db_connection import engine

# 1) Chemin vers le fichier Excel versionné par DVC
BASE_DIR = Path(__file__).resolve().parent.parent
EXCEL_PATH = BASE_DIR / "data" / "raw" / "ccistta_data_unlocked.xlsx"

# 2) Lire l'Excel
df = pd.read_excel(EXCEL_PATH, engine="openpyxl")

# 3) Normaliser les noms de colonnes 
df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

# 4) Importer dans PostgreSQL 
df.to_sql("recettes_raw", engine, schema="public", if_exists="replace", index=False)

print("Import terminé : public.recettes_raw")
print("Lignes:", len(df), "| Colonnes:", len(df.columns))