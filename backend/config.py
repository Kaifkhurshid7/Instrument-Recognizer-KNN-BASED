"""
Application Configuration
-------------------------
Centralized settings for the Flask backend. Values can be overridden
via environment variables for deployment flexibility.
"""

import os

# Base directory (backend/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
DATABASE_FILE = os.path.join(BASE_DIR, "reference_database.pkl")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DATASET_PATH = os.path.join(BASE_DIR, "IRMAS-TrainingData")

# Audio Processing
SAMPLE_RATE = 22050
MAX_DURATION_SECONDS = 30
WAVEFORM_DISPLAY_SAMPLES = 1000

# KNN Classifier
KNN_NEIGHBORS = 7
KNN_METRIC = "cosine"
KNN_WEIGHTS = "distance"

# Feature Extraction
FEATURE_VECTOR_LENGTH = 26  # Enhanced from 10 → 26 dimensions
MFCC_COEFFICIENTS = 13

# Server
PORT = int(os.environ.get("PORT", 5000))
DEBUG = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
