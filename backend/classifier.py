"""
KNN Classifier Service
----------------------
Manages the lifecycle of the KNN model: loading the reference database,
training the classifier, and running predictions.

Architecture Note:
  The classifier is trained once at server startup from a pre-built
  reference database (pickle file). This avoids re-processing thousands
  of audio files on every restart.
"""

import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from config import DATABASE_FILE, KNN_NEIGHBORS, KNN_METRIC, KNN_WEIGHTS


class InstrumentClassifier:
    """Encapsulates the KNN model, scaler, and reference data."""

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
        """
        Load the reference database and train the KNN classifier.
        Called once during server startup.
        """
        print("[Classifier] Loading reference database...")

        with open(DATABASE_FILE, "rb") as f:
            self.reference_database = pickle.load(f)

        # Build training matrices from stored fingerprints
        X, y = [], []
        self.class_names = sorted(self.reference_database.keys())
        class_map = {name: idx for idx, name in enumerate(self.class_names)}

        for instrument, data in self.reference_database.items():
            for vector in data["fingerprints"]:
                if len(vector) == 10:
                    X.append(vector)
                    y.append(class_map[instrument])

        X = np.array(X)
        y = np.array(y)

        # Normalize features so no single dimension dominates distance calc
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train KNN with cosine distance (works well for spectral fingerprints)
        self.model = KNeighborsClassifier(
            n_neighbors=KNN_NEIGHBORS,
            metric=KNN_METRIC,
            weights=KNN_WEIGHTS,
        )
        self.model.fit(X_scaled, y)
        self._is_ready = True

        print(
            f"[Classifier] Ready | Classes: {len(self.class_names)} | Samples: {len(y)}"
        )

    def predict(self, feature_vector):
        """
        Classify a single feature vector.

        Parameters
        ----------
        feature_vector : np.ndarray of shape (10,)

        Returns
        -------
        dict with keys:
            'instrument'    : str - predicted instrument name (title case)
            'confidence'    : float - confidence percentage (0-100)
            'probabilities' : list[dict] - all classes sorted by score
            'average_vector': list[float] - database average for predicted class
        """
        scaled = self.scaler.transform(feature_vector.reshape(1, -1))
        proba = self.model.predict_proba(scaled)[0]

        best_idx = np.argmax(proba)
        instrument = self.class_names[best_idx]
        confidence = proba[best_idx] * 100

        # Build sorted probability table for all instruments
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
            "average_vector": self.reference_database[instrument][
                "average_vector"
            ].tolist(),
        }
