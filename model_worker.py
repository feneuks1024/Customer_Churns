import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

RANDOM_STATE = 42
DISPLAY_NAMES = {
    "logistic": "Логистическая регрессия",
    "random_forest": "Случайный лес",
    "gradient_boosting": "Градиентный бустинг",
}


def get_model(name):
    if name == "logistic":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                C=0.5,
                max_iter=3000,
                solver="liblinear",
                random_state=RANDOM_STATE,
            )),
        ])
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=180,
            max_depth=10,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=1,
            random_state=RANDOM_STATE,
        )
    if name == "gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.07,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=RANDOM_STATE,
        )
    raise ValueError(name)


def fit_model(model, X, y, balanced):
    params = {}
    if balanced:
        weights = compute_sample_weight(class_weight="balanced", y=y)
        if isinstance(model, Pipeline):
            params["model__sample_weight"] = weights
        else:
            params["sample_weight"] = weights
    model.fit(X, y, **params)
    return model


def metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
    }


def load_data(path):
    df = pd.read_csv(path).drop_duplicates().reset_index(drop=True)
    features = [column for column in df.columns if column not in ["Churn", "FN", "FP"]]
    X = df[features]
    y = df["Churn"].astype(int)
    return df, features, X, y


def run_cv(model_name, balanced, data_path):
    _, _, X, y = load_data(data_path)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    probabilities = np.zeros(len(y_train), dtype=float)
    folds = []
    base_model = get_model(model_name)
    for fold, (train_index, valid_index) in enumerate(cv.split(X_train, y_train), start=1):
        model = fit_model(
            clone(base_model),
            X_train.iloc[train_index],
            y_train.iloc[train_index],
            balanced,
        )
        fold_probabilities = model.predict_proba(X_train.iloc[valid_index])[:, 1]
        probabilities[valid_index] = fold_probabilities
        folds.append({"fold": fold, **metrics(y_train.iloc[valid_index], fold_probabilities, 0.5)})
    thresholds = np.linspace(0.05, 0.95, 181)
    f1_scores = np.array([
        f1_score(y_train, probabilities >= threshold, zero_division=0)
        for threshold in thresholds
    ])
    threshold = float(thresholds[int(f1_scores.argmax())])
    return {
        "model_key": model_name,
        "model_name": DISPLAY_NAMES[model_name],
        "balanced": balanced,
        "threshold": threshold,
        "default_metrics": metrics(y_train, probabilities, 0.5),
        "tuned_metrics": metrics(y_train, probabilities, threshold),
        "folds": folds,
    }


def run_final(model_name, balanced, threshold, data_path):
    _, features, X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    model = fit_model(get_model(model_name), X_train, y_train, balanced)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    estimator = model.named_steps["model"] if isinstance(model, Pipeline) else model
    if hasattr(estimator, "feature_importances_"):
        importance_values = estimator.feature_importances_
        importance_method = "Встроенная важность"
    elif hasattr(estimator, "coef_"):
        importance_values = np.abs(estimator.coef_[0])
        importance_method = "Модуль коэффициента"
    else:
        result = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="f1",
            n_repeats=5,
            random_state=RANDOM_STATE,
        )
        importance_values = result.importances_mean
        importance_method = "Permutation importance"
    return {
        "model_key": model_name,
        "model_name": DISPLAY_NAMES[model_name],
        "balanced": balanced,
        "threshold": threshold,
        "metrics": metrics(y_test, probabilities, threshold),
        "features": features,
        "importance": [float(value) for value in importance_values],
        "importance_method": importance_method,
        "y_test": [int(value) for value in y_test],
        "probabilities": [float(value) for value in probabilities],
        "predictions": [int(value) for value in predictions],
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=DISPLAY_NAMES)
    parser.add_argument("--mode", required=True, choices=["cv", "final"])
    parser.add_argument("--balanced", required=True, choices=["0", "1"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--data", default="iranian_churn.csv")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    balanced = args.balanced == "1"
    if args.mode == "cv":
        result = run_cv(args.model, balanced, args.data)
    else:
        result = run_final(args.model, balanced, args.threshold, args.data)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
