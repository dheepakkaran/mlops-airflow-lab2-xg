# File: src/assets.py
"""The asset both DAGs agree on: a trained and evaluated xG model.

Airflow only records an asset event when the producing task *succeeds*, so a
failed training run cannot publish a model to the serving DAG.
"""

from airflow.sdk import Asset

XG_MODEL = Asset(name="xg_model", uri="file://xg/model.sav")
