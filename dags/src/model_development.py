# File: src/model_development.py
import os
import pickle
import shutil

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

# One model is trained per entry, as parallel mapped task instances. "all" is the
# pooled baseline the per-tournament models have to beat.
TRAINING_SETS = ["FIFA World Cup 2022", "UEFA Euro 2024", "all"]

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

    # Only 9.1% of shots are goals, so stratify to hold that rate in both splits.
    # Whole frames are kept, not just FEATURES, because the mapped training tasks
    # filter on the tournament column.
    train_df, test_df = train_test_split(
        df, test_size=0.3, stratify=df[TARGET], random_state=SEED
    )

    # No scaling: gradient boosting splits on thresholds, so feature scale is irrelevant.
    out_path = os.path.join(WORKING_DIR, "preprocessed.pkl")
    with open(out_path, "wb") as f:
        pickle.dump((train_df, test_df), f)
    return out_path

def separate_data_outputs(file_path: str) -> str:
    """
    Passthrough; kept so the DAG composes cleanly.
    """
    return file_path

def build_model(file_path: str, training_set: str) -> dict:
    """
    Train one xG model on a single training set and score it.

    One mapped task instance runs per entry in TRAINING_SETS. Returns the metrics
    the reduce step compares, not the model itself, so the XCom stays small.
    """
    with open(file_path, "rb") as f:
        train_df, test_df = pickle.load(f)

    subset = (
        train_df
        if training_set == "all"
        else train_df[train_df["tournament"] == training_set]
    )

    X, y = subset[FEATURES], subset[TARGET]

    # Balanced weights so the 9% goal class is not drowned out by the misses.
    weights = compute_sample_weight(class_weight="balanced", y=y)

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=SEED,
    )
    model.fit(X, y, sample_weight=weights)

    # Every model is scored on the same held-out test set, which spans both
    # tournaments, so the numbers are comparable across mapped instances.
    proba = model.predict_proba(test_df[FEATURES])[:, 1]

    slug = training_set.lower().replace(" ", "_")
    model_path = os.path.join(MODEL_DIR, f"model_{slug}.sav")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return {
        "training_set": training_set,
        "model_path": model_path,
        "n_shots": int(len(subset)),
        "roc_auc": float(roc_auc_score(test_df[TARGET], proba)),
        "brier": float(brier_score_loss(test_df[TARGET], proba)),
    }

def select_best_model(results: list, filename: str) -> str:
    """
    Reduce step: compare the mapped models and publish the best one as filename.

    Takes the list of dicts the mapped build_model instances returned.
    """
    # Ranked and selected on Brier, not ROC-AUC. ROC-AUC only says the model
    # orders shots correctly; Brier says the probability itself is honest, and an
    # xG number nobody can trust is not worth publishing.
    for r in sorted(results, key=lambda r: r["brier"]):
        print(
            f"{r['training_set']:>20} | {r['n_shots']:>5} shots | "
            f"ROC-AUC {r['roc_auc']:.3f} | Brier {r['brier']:.4f}"
        )

    best = min(results, key=lambda r: r["brier"])
    published = os.path.join(MODEL_DIR, filename)
    shutil.copyfile(best["model_path"], published)
    print(f"Published {best['training_set']} as {filename}")
    return published
