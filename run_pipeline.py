import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def run_step(step_name, command):
    print("\n" + "=" * 60)
    print(f"{step_name}")
    print("=" * 60)

    result = subprocess.run(
        command,
        cwd=BASE_DIR,
        shell=True
    )

    if result.returncode != 0:
        print(f"\n Erreur dans l'étape : {step_name}")
        sys.exit(result.returncode)

    print(f"\n Étape terminée : {step_name}")


def main():
    print("\n=== DÉMARRAGE DU PIPELINE MLOPS ===")

    run_step(
        "1. Chargement des données Excel vers PostgreSQL",
        "python src/01_load_excel_to_postgres.py"
    )

    run_step(
        "2. Nettoyage des données",
        "python src/02_clean_data.py"
    )

    run_step(
        "3. Validation des données",
        "python src/03_data_validation.py"
    )

    
    run_step(
        "4. Feature Engineering",
        "python src/04_feature_engineering.py"
    )

    run_step(
        "5. Analyse EDA",
        "python src/05_eda_analysis.py"
    )

    run_step(
        "6. Entraînement du modèle XGBoost",
        "python src/06_train_xgboost.py"
    )

    run_step(
        "7. Évaluation du modèle XGBoost",
        "python src/07_evaluate_xgboost.py"
    )

    run_step(
        "8. Génération des prévisions",
        "python src/08_generate_forecasts.py"
    )

    print("\n" + "=" * 60)
    print(" PIPELINE MLOPS TERMINÉ AVEC SUCCÈS")
    print("=" * 60)


if __name__ == "__main__":
    main()