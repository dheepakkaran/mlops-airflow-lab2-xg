"""Unit tests for the pipeline functions the DAG tasks call."""

import pickle

import pytest

from src.model_development import (
    FEATURES,
    TARGET,
    TRAINING_SETS,
    build_model,
    data_preprocessing,
    load_data,
    select_best_model,
)


@pytest.fixture(scope="module")
def prepared():
    """Run the real load and split once; several tests read the result."""
    path = data_preprocessing(load_data())
    with open(path, "rb") as f:
        train_df, test_df = pickle.load(f)
    return path, train_df, test_df


def test_features_exclude_the_label_and_the_reference_model():
    """statsbomb_xg is another model's answer for the same shot; training on it leaks."""
    assert TARGET not in FEATURES
    assert "statsbomb_xg" not in FEATURES
    assert "is_goal" not in FEATURES


def test_preprocessing_keeps_the_column_the_mapping_needs(prepared):
    _, train_df, test_df = prepared
    assert "tournament" in train_df.columns
    assert "tournament" in test_df.columns


def test_split_preserves_the_goal_rate(prepared, shots):
    """Goals are only ~9% of shots, so an unstratified split skews the rate."""
    _, train_df, test_df = prepared
    overall = shots[TARGET].mean()
    assert train_df[TARGET].mean() == pytest.approx(overall, abs=0.01)
    assert test_df[TARGET].mean() == pytest.approx(overall, abs=0.01)


def test_training_set_filters_to_one_tournament(prepared):
    path, train_df, _ = prepared
    result = build_model(path, "UEFA Euro 2024")
    expected = (train_df["tournament"] == "UEFA Euro 2024").sum()
    assert result["n_shots"] == expected
    assert result["n_shots"] < len(train_df)


def test_all_uses_every_training_shot(prepared):
    path, train_df, _ = prepared
    assert build_model(path, "all")["n_shots"] == len(train_df)


def test_build_model_returns_comparable_metrics(prepared):
    path, _, _ = prepared
    result = build_model(path, "all")
    assert set(result) == {"training_set", "model_path", "n_shots", "roc_auc", "brier"}
    assert 0.0 <= result["roc_auc"] <= 1.0
    assert 0.0 <= result["brier"] <= 1.0


def test_selection_prefers_calibration_over_ranking(tmp_path):
    """
    The pooled model ranks shots best but is the worst at being right about the
    probability. An xG value *is* a probability, so the lower Brier has to win
    even when its ROC-AUC is lower.
    """
    best_ranked = tmp_path / "ranked.sav"
    best_calibrated = tmp_path / "calibrated.sav"
    best_ranked.write_bytes(b"ranked")
    best_calibrated.write_bytes(b"calibrated")

    results = [
        {"training_set": "all", "model_path": str(best_ranked),
         "n_shots": 1913, "roc_auc": 0.773, "brier": 0.1377},
        {"training_set": "UEFA Euro 2024", "model_path": str(best_calibrated),
         "n_shots": 909, "roc_auc": 0.757, "brier": 0.1188},
    ]

    published = select_best_model(results, "model.sav")
    assert open(published, "rb").read() == b"calibrated"


def test_every_training_set_is_a_real_tournament_or_the_pool(shots):
    tournaments = set(shots["tournament"].unique())
    for name in TRAINING_SETS:
        assert name == "all" or name in tournaments, f"{name} matches no shots"
