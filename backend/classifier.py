"""
KNN instrument classifier.

Loads a pre-built reference database at startup, trains a KNN model
with cosine distance, and exposes a predict() method for inference.
"""

import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from config import (
    DATABASE_FILE,
    KNN_NEIGHBORS,
    KNN_METRIC,
    KNN_WEIGHTS,
    FEATURE_VECTOR_LENGTH,
)


class InstrumentClassifier:
    """KNN-based instrument classifier with StandardScaler normalization."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.class_names = None
        self.reference_database = None
        self._is_ready = False

    @property
    def is_ready(self):
        return self._is_ready

    def load_and_train(self):
        """Load reference database and fit the KNN model. Called once at startup."""
        print("[Classifier] Loading reference database...")

        with open(DATABASE_FILE, "rb") as f:
            self.reference_database = pickle.load(f)

        X, y = [], []
        self.class_names = sorted(self.reference_database.keys())
        class_map = {name: idx for idx, name in enumerate(self.class_names)}

        for instrument, data in self.reference_database.items():
            for vector in data["fingerprints"]:
                if len(vector) == FEATURE_VECTOR_LENGTH:
                    X.append(vector)
                    y.append(class_map[instrument])

        X = np.array(X)
        y = np.array(y)

        # Normalize so no single feature dominates the distance calculation
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = KNeighborsClassifier(
            n_neighbors=KNN_NEIGHBORS,
            metric=KNN_METRIC,
            weights=KNN_WEIGHTS,
        )
        self.model.fit(X_scaled, y)
        self._is_ready = True

        print(f"[Classifier] Ready | {len(self.class_names)} classes | {len(y)} samples")

    def predict(self, feature_vector):
        """
        Classify a feature vector and return prediction with probabilities.

        Args:
            feature_vector: np.ndarray of shape (26,)

        Returns:
            dict with instrument, confidence, probabilities, and average_vector.
        """
        scaled = self.scaler.transform(feature_vector.reshape(1, -1))
        proba = self.model.predict_proba(scaled)[0]

        best_idx = np.argmax(proba)
        instrument = self.class_names[best_idx]
        confidence = proba[best_idx] * 100

        probability_table = sorted(
            [
                {"name": self.class_names[i].title(), "score": round(p * 100, 2)}
                for i, p in enumerate(proba)
            ],
            key=lambda x: x["score"],
            reverse=True,
        )

        return {
            "instrument": instrument.title(),
            "confidence": round(confidence, 2),
            "probabilities": probability_table,
            "average_vector": self.reference_database[instrument]["average_vector"].tolist(),
        }
