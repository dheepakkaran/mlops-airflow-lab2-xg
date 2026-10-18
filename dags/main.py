# File: main.py
from __future__ import annotations

import pendulum
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.smtp.operators.smtp import EmailOperator
from airflow.sdk import task

from src.assets import XG_MODEL
from src.model_development import (
    TRAINING_SETS,
    load_data,
    data_preprocessing,
    separate_data_outputs,
    build_model,
    select_best_model,
)

# ---------- Default args ----------
default_args = {
    "start_date": pendulum.datetime(2024, 1, 1, tz="UTC"),
    "retries": 0,
}

# ---------- DAG ----------
dag = DAG(
    dag_id="Airflow_Lab2",
    default_args=default_args,
    description="Airflow-Lab2 DAG Description",
    schedule="@daily",
    catchup=False,
    tags=["example"],
    owner_links={"Ramin Mohammadi": "https://github.com/raminmohammadi/MLOps/"},
    max_active_runs=1,
)

# ---------- Tasks ----------
owner_task = BashOperator(
    task_id="task_using_linked_owner",
    bash_command="echo 1",
    owner="Ramin Mohammadi",
    dag=dag,
)

# send_email = EmailOperator(
#     task_id="send_email",
#     to="YOUR EMAIL ADDRESS",
#     subject="Notification from Airflow",
#     html_content="<p>This is a notification email sent from Airflow.</p>",
#     dag=dag,
# )

load_data_task = PythonOperator(
    task_id="load_data_task",
    python_callable=load_data,
    dag=dag,
)

data_preprocessing_task = PythonOperator(
    task_id="data_preprocessing_task",
    python_callable=data_preprocessing,
    op_args=[load_data_task.output],
    dag=dag,
)

separate_data_outputs_task = PythonOperator(
    task_id="separate_data_outputs_task",
    python_callable=separate_data_outputs,
    op_args=[data_preprocessing_task.output],
    dag=dag,
)

# Fan out: one mapped task instance per training set, all running in parallel.
# The lab trained a single model in a single task.
@task(task_id="build_save_model_task", dag=dag)
def build_save_model(file_path: str, training_set: str) -> dict:
    return build_model(file_path, training_set)


build_save_model_task = build_save_model.partial(
    file_path=separate_data_outputs_task.output
).expand(training_set=TRAINING_SETS)


# Fan in. The asset is emitted here, so a model reaches the serving DAG only
# after every mapped model has trained and the best of them has been picked.
@task(task_id="select_best_model_task", outlets=[XG_MODEL], dag=dag)
def select_best(results: list) -> str:
    return select_best_model(results, "model.sav")


select_best_model_task = select_best(build_save_model_task)

# ---------- Dependencies ----------
# There is no trigger task. Airflow_Lab2_Flask is scheduled on the XG_MODEL
# asset, so it starts when select_best_model_task succeeds and emits the event.
owner_task >> load_data_task >> data_preprocessing_task >> \
    separate_data_outputs_task >> build_save_model_task >> \
    select_best_model_task
