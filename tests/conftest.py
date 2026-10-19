"""Shared setup. Everything here must run before airflow is imported."""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# A scratch AIRFLOW_HOME, so running the tests never touches a real instance.
_TMP = tempfile.mkdtemp(prefix="lab2-tests-")
os.environ.setdefault("AIRFLOW_HOME", _TMP)
os.environ["AIRFLOW__CORE__DAGS_FOLDER"] = str(ROOT / "dags")
os.environ["AIRFLOW__CORE__LOAD_EXAMPLES"] = "False"
os.environ["LAB2_WORKING_DIR"] = os.path.join(_TMP, "working_data")
os.environ["LAB2_MODEL_DIR"] = os.path.join(_TMP, "model")

# The DAG files import `src.…`, which only resolves with dags/ on the path.
sys.path.insert(0, str(ROOT / "dags"))

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def dagbag():
    # airflow.models.dagbag is deprecated in Airflow 3; examples are excluded by
    # AIRFLOW__CORE__LOAD_EXAMPLES above rather than by a constructor argument.
    from airflow.dag_processing.dagbag import DagBag

    return DagBag(dag_folder=str(ROOT / "dags"))


@pytest.fixture(scope="session")
def shots():
    import pandas as pd

    return pd.read_csv(ROOT / "dags" / "data" / "shots.csv")
