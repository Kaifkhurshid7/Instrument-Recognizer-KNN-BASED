"""
KNN-based instrument classifier with explainability support.

This module implements a K-Nearest Neighbors classifier for musical instrument
recognition. It includes:
- Model loading and training from pre-built reference database
- Prediction with confidence scores and probability distribution
- Feature normalization using StandardScaler
- Detailed logging and error handling

The classifier operates on 26-dimensional spectral feature vectors and
uses cosine distance with distance-weighted voting for predictions.

Author: Instrument Recognizer Team
Date: 2024
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from config import (
    DATABASE_FILE,
    KNN_NEIGHBORS,
    KNN_METRIC,
    KNN_WEIGHTS,
    FEATURE_VECTOR_LENGTH,
)
from logger import setup_logger
from validators import (
    ModelNotReadyError,
    ClassificationError,
    InstrumentRecognizerException,
)

# Module-level logger
logger = setup_logger(__name__)


# ============================================================================
# PREDICTION RESPONSE TYPES
# ============================================================================


class PredictionResult:
    """
    Container for classification prediction results.
    
    Attributes:
        instrument: Predicted instrument class name
        confidence: Confidence score as percentage (0-100)
        probabilities: List of all class probabilities
        average_vector: Average feature vector for predicted class
    """
    
    def __init__(
        self,
        instrument: str,
        confidence: float,
        probabilities: List[Dict[str, float]],
        average_vector: List[float]
    ):
        """Initialize prediction result."""
        self.instrument = instrument
        self.confidence = confidence
        self.probabilities = probabilities
        self.average_vector = average_vector
    
    def to_dict(self) -> Dict:
        """
        Convert result to dictionary format.
        
        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            "instrument": self.instrument,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "average_vector": self.average_vector,
        }


# ============================================================================
# CLASSIFIER CLASS
# ============================================================================


class InstrumentClassifier:
    """
    KNN-based classifier for musical instrument recognition.
    
    The classifier uses:
    - K=3 nearest neighbors
    - Cosine distance metric
    - Distance-weighted voting
    - StandardScaler normalization
    
    Example:
        >>> classifier = InstrumentClassifier()
        >>> classifier.load_and_train()
        >>> prediction = classifier.predict(feature_vector)
        >>> print(f"{prediction.instrument}: {prediction.confidence}%")
    """
    
    def __init__(self, database_path: Optional[Path] = None):
        """
        Initialize the classifier.
        
        Args:
            database_path: Path to reference database pickle file.
                          Defaults to config.DATABASE_FILE
        """
        self.database_path: Path = Path(database_path or DATABASE_FILE)
        self.model: Optional[KNeighborsClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.class_names: Optional[np.ndarray] = None
        self.reference_database: Optional[Dict] = None
        self._is_ready: bool = False
        
        logger.debug(f"Classifier initialized with database: {self.database_path}")
    
    @property
    def is_ready(self) -> bool:
        """
        Check if classifier is ready for predictions.
        
        Returns:
            True if model is loaded and trained, False otherwise
        """
        return self._is_ready and self.model is not None
    
    def load_and_train(self) -> None:
        """
        Load reference database and train KNN model.
        
        This method:
        1. Loads the pre-built reference database
        2. Extracts feature vectors and labels
        3. Fits StandardScaler for normalization
        4. Trains KNeighborsClassifier
        
        Must be called before any predictions.
        
        Raises:
            FileNotFoundError: If database file not found
            InstrumentRecognizerException: If training fails
            
        Example:
            >>> classifier = InstrumentClassifier()
            >>> classifier.load_and_train()
            >>> assert classifier.is_ready
        """
        
        logger.info("Loading reference database...")
        
        try:
            # Load database
            if not self.database_path.exists():
                raise FileNotFoundError(
                    f"Reference database not found: {self.database_path}. "
                    f"Run 'python build_database.py' to generate it."
                )
            
            with open(self.database_path, 'rb') as f:
                self.reference_database = pickle.load(f)
            
            logger.debug(
                f"Database loaded successfully with "
                f"{len(self.reference_database)} classes"
            )
            
            # Extract training data
            X: List[np.ndarray] = []
            y: List[int] = []
            self.class_names = np.array(sorted(self.reference_database.keys()))
            class_map: Dict[str, int] = {
                name: idx for idx, name in enumerate(self.class_names)
            }
            
            # Build training set from fingerprints
            for instrument, data in self.reference_database.items():
                fingerprints = data.get("fingerprints", [])
                
                for vector in fingerprints:
                    if len(vector) != FEATURE_VECTOR_LENGTH:
                        logger.warning(
                            f"Skipping fingerprint for {instrument}: "
                            f"expected {FEATURE_VECTOR_LENGTH} features, "
                            f"got {len(vector)}"
                        )
                        continue
                    
                    X.append(vector)
                    y.append(class_map[instrument])
            
            if not X:
                raise InstrumentRecognizerException(
                    "No valid training data found in reference database"
                )
            
            X_array: np.ndarray = np.array(X)
            y_array: np.ndarray = np.array(y)
            
            logger.info(
                f"Training data prepared: "
                f"{len(X_array)} samples, {len(self.class_names)} classes"
            )
            
            # Fit StandardScaler
            self.scaler = StandardScaler()
            X_scaled: np.ndarray = self.scaler.fit_transform(X_array)
            
            # Train KNN model
            self.model = KNeighborsClassifier(
                n_neighbors=KNN_NEIGHBORS,
                metric=KNN_METRIC,
                weights=KNN_WEIGHTS,
                n_jobs=-1,  # Use all CPU cores
            )
            self.model.fit(X_scaled, y_array)
            
            self._is_ready = True
            
            logger.info(
                "Classifier training completed",
                extra={
                    'num_classes': len(self.class_names),
                    'num_samples': len(y_array),
                    'k_neighbors': KNN_NEIGHBORS,
                    'metric': KNN_METRIC,
                }
            )
            
        except FileNotFoundError as e:
            logger.error(f"Database file error: {e}")
            raise
        except Exception as e:
            logger.error(
                "Classifier training failed",
                extra={'error': str(e)},
                exc_info=True
            )
            self._is_ready = False
            raise InstrumentRecognizerException(
                f"Failed to train classifier: {e}"
            )
    
    def predict(self, feature_vector: np.ndarray) -> PredictionResult:
        """
        Predict instrument class for a feature vector.
        
        Args:
            feature_vector: np.ndarray of shape (26,) containing
                          26-dimensional spectral features
        
        Returns:
            PredictionResult with instrument, confidence, and probabilities
        
        Raises:
            ModelNotReadyError: If model not yet trained
            ClassificationError: If prediction fails
            ValueError: If feature vector has wrong shape
            
        Example:
            >>> classifier = InstrumentClassifier()
            >>> classifier.load_and_train()
            >>> result = classifier.predict(features)
            >>> print(f"Predicted: {result.instrument}")
            >>> print(f"Confidence: {result.confidence:.1f}%")
        """
        
        if not self.is_ready:
            raise ModelNotReadyError(
                "Classifier not initialized. Call load_and_train() first."
            )
        
        try:
            # Validate input
            if not isinstance(feature_vector, np.ndarray):
                feature_vector = np.array(feature_vector)
            
            if feature_vector.shape[0] != FEATURE_VECTOR_LENGTH:
                raise ValueError(
                    f"Feature vector has wrong shape. "
                    f"Expected ({FEATURE_VECTOR_LENGTH},), "
                    f"got {feature_vector.shape}"
                )
            
            # Normalize using fitted scaler
            scaled: np.ndarray = self.scaler.transform(
                feature_vector.reshape(1, -1)
            )
            
            # Get prediction probabilities
            proba: np.ndarray = self.model.predict_proba(scaled)[0]
            
            # Find best prediction
            best_idx: int = int(np.argmax(proba))
            instrument: str = str(self.class_names[best_idx]).title()
            confidence: float = round(float(proba[best_idx] * 100), 2)
            
            # Build probability distribution
            probability_table: List[Dict[str, float]] = [
                {
                    "name": str(self.class_names[i]).title(),
                    "score": round(float(proba[i] * 100), 2)
                }
                for i in range(len(proba))
            ]
            
            # Sort by score (descending)
            probability_table.sort(key=lambda x: x["score"], reverse=True)
            
            # Get average feature vector for predicted class
            average_vector: List[float] = (
                self.reference_database[self.class_names[best_idx]]
                ["average_vector"]
                .tolist()
            )
            
            logger.debug(
                "prediction_complete",
                extra={
                    'instrument': instrument,
                    'confidence': confidence,
                    'top_3': [p['name'] for p in probability_table[:3]]
                }
            )
            
            return PredictionResult(
                instrument=instrument,
                confidence=confidence,
                probabilities=probability_table,
                average_vector=average_vector,
            )
            
        except ValueError as e:
            logger.error(f"Invalid input for prediction: {e}")
            raise
        except Exception as e:
            logger.error(
                "Prediction failed",
                extra={'error': str(e)},
                exc_info=True
            )
            raise ClassificationError(f"Prediction failed: {e}")
