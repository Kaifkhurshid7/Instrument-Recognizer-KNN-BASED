"""
Model evaluation script.
Runs stratified k-fold cross-validation on the reference database
and prints accuracy, precision, recall, and F1 score.

Usage:
    python evaluate.py
"""

import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate

from config import DATABASE_FILE, KNN_NEIGHBORS, KNN_METRIC, KNN_WEIGHTS, FEATURE_VECTOR_LENGTH


def evaluate():
    print("[Evaluate] Loading reference database...")

    with open(DATABASE_FILE, "rb") as f:
        database = pickle.load(f)

    X, y = [], []
    class_names = sorted(database.keys())
    class_map = {name: idx for idx, name in enumerate(class_names)}

    for instrument, data in database.items():
        for vector in data["fingerprints"]:
            if len(vector) == FEATURE_VECTOR_LENGTH:
                X.append(vector)
                y.append(class_map[instrument])

    X = np.array(X)
    y = np.array(y)

    print(f"[Evaluate] {len(class_names)} classes | {len(y)} samples | {FEATURE_VECTOR_LENGTH}-D vector")
    print(f"[Evaluate] Model: KNN (K={KNN_NEIGHBORS}, metric={KNN_METRIC}, weights={KNN_WEIGHTS})")
    print()

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KNN model
    model = KNeighborsClassifier(
        n_neighbors=KNN_NEIGHBORS,
        metric=KNN_METRIC,
        weights=KNN_WEIGHTS,
    )

    # 5-fold stratified cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
    results = cross_validate(model, X_scaled, y, cv=cv, scoring=scoring)

    print("=" * 50)
    print("  5-Fold Stratified Cross-Validation Results")
    print("=" * 50)
    print(f"  Accuracy:   {results['test_accuracy'].mean() * 100:.2f}% (+/- {results['test_accuracy'].std() * 100:.2f}%)")
    print(f"  Precision:  {results['test_precision_weighted'].mean() * 100:.2f}%")
    print(f"  Recall:     {results['test_recall_weighted'].mean() * 100:.2f}%")
    print(f"  F1 Score:   {results['test_f1_weighted'].mean() * 100:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    evaluate()
