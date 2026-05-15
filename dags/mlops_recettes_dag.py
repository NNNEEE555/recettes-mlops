from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


default_args = {
    "owner": "pfe",
    "retries": 1,
}


with DAG(
    dag_id="mlops_recettes_pipeline",
    default_args=default_args,
    description="Pipeline MLOps pour la prévision des recettes",
    start_date=datetime(2026, 1, 1),
    schedule="0 8 1 * *",
    catchup=False,
    tags=["mlops", "recettes", "xgboost"],
) as dag:

    load_excel_to_postgres = BashOperator(
        task_id="load_excel_to_postgres",
        bash_command="cd /usr/local/airflow && python src/01_load_excel_to_postgres.py",
    )

    clean_data = BashOperator(
        task_id="clean_data",
        bash_command="cd /usr/local/airflow && python src/02_clean_data.py",
    )

    data_validation = BashOperator(
        task_id="data_validation",
        bash_command="cd /usr/local/airflow && python src/03_data_validation.py",
    )

    eda_analysis = BashOperator(
        task_id="eda_analysis",
        bash_command="cd /usr/local/airflow && python src/04_eda_analysis.py",
    )

    feature_engineering = BashOperator(
        task_id="feature_engineering",
        bash_command="cd /usr/local/airflow && python src/05_feature_engineering.py",
    )

    train_xgboost = BashOperator(
        task_id="train_xgboost",
        bash_command="cd /usr/local/airflow && python src/06_train_xgboost.py",
    )

    evaluate_xgboost = BashOperator(
        task_id="evaluate_xgboost",
        bash_command="cd /usr/local/airflow && python src/07_evaluate_xgboost.py",
    )

    generate_forecasts = BashOperator(
        task_id="generate_forecasts",
        bash_command="cd /usr/local/airflow && python src/08_generate_forecasts.py",
    )

    (
        load_excel_to_postgres
        >> clean_data
        >> data_validation
        >> eda_analysis
        >> feature_engineering
        >> train_xgboost
        >> evaluate_xgboost
        >> generate_forecasts
    )