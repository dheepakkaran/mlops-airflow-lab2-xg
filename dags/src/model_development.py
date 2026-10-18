# File: src/model_development.py
import os
import pickle

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

# Output lives at the project root, outside dags/ which Airflow scans.
# Overridable so the same code still runs under Docker, where AIRFLOW_HOME is /opt/airflow.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKING_DIR = os.getenv("LAB2_WORKING_DIR", os.path.join(PROJECT_ROOT, "working_data"))
MODEL_DIR = os.getenv("LAB2_MODEL_DIR", os.path.join(PROJECT_ROOT, "model"))
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Shot features. Penalties are already excluded from shots.csv, because their
# chance of scoring does not depend on position or pressure.
FEATURES = [
    "distance",
    "angle",
    "is_header",
    "first_time",
    "under_pressure",
    "one_on_one",
    "from_free_kick",
    "from_counter",
    "from_set_piece",
]
TARGET = "is_goal"
SEED = 42

def load_data() -> str:
    """
    Load CSV and persist raw dataframe to a pickle file.
    Returns path to saved file.
    """
    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "shots.csv",
    )
    df = pd.read_csv(csv_path)

    out_path = os.path.join(WORKING_DIR, "raw.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(df, f)
    return out_path

def data_preprocessing(file_path: str) -> str:
    """
    Load dataframe, split, and save (X_train, X_test, y_train, y_test) to pickle.
    Returns path to saved file.
    """
    with open(file_path, "rb") as f:
        df = pickle.load(f)

    X = df[FEATURES]
    y = df[TARGET]

    # Only 9.1% of shots are goals, so stratify to hold that rate in both splits.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED
    )

    # No scaling: gradient boosting splits on thresholds, so feature scale is irrelevant.
    out_path = os.path.join(WORKING_DIR, "preprocessed.pkl")
    with open(out_path, "wb") as f:
        pickle.dump((X_train, X_test, y_train.values, y_test.values), f)
    return out_path

def separate_data_outputs(file_path: str) -> str:
    """
    Passthrough; kept so the DAG composes cleanly.
    """
    return file_path

def build_model(file_path: str, filename: str) -> str:
    """
    Train the xG model and save to MODEL_DIR/filename. Returns model path.
    """
    with open(file_path, "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)

    # Balanced weights so the 9% goal class is not drowned out by the misses.
    weights = compute_sample_weight(class_weight="balanced", y=y_train)

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=SEED,
    )
    model.fit(X_train, y_train, sample_weight=weights)

    model_path = os.path.join(MODEL_DIR, filename)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return model_path

def load_model(file_path: str, filename: str) -> int:
    """
    Load saved model and test set, print metrics, and return first prediction as int.
    """
    with open(file_path, "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)

    model_path = os.path.join(MODEL_DIR, filename)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Accuracy is meaningless here: always predicting "no goal" already scores 90.9%.
    # ROC-AUC measures ranking, Brier measures whether the probabilities are honest,
    # which is the property an xG model lives or dies by.
    proba = model.predict_proba(X_test)[:, 1]
    print(f"Test ROC-AUC: {roc_auc_score(y_test, proba):.3f}")
    print(f"Test Brier score: {brier_score_loss(y_test, proba):.4f}")

    pred = model.predict(X_test)
    return int(pred[0])
