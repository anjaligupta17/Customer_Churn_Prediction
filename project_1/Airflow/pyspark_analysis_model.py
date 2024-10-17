
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 9, 25, 0, 0),  # Set to today
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'pyspark_analysis_model',
    default_args=default_args,
    description='A DAG to run a PySpark script daily',
    schedule_interval='@daily',
    catchup=False,  # Disable catching up on missed runs
)

run_pyspark_script = BashOperator(
    task_id='run_pyspark',
    bash_command='spark-submit /Users/anjaligupta/Desktop/project/project_1/pyspark/customer_churn_prediction.py',
    dag=dag,
)

run_pyspark_script