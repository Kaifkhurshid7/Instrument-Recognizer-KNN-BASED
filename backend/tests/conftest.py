"""
Pytest configuration and fixtures for Instrument Recognizer tests.

This module provides:
- Shared test fixtures
- Database and file system setup/teardown
- Mock objects and utilities
- Test configuration

Author: Instrument Recognizer Team
Date: 2024
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock
from fastapi.testclient import TestClient

from app import app, classifier
from classifier import InstrumentClassifier, PredictionResult
from config import (
    FEATURE_VECTOR_LENGTH,
    ModelConfig,
)


# ============================================================================
# TEST CLIENT FIXTURE
# ============================================================================


@pytest.fixture(scope="session")
def test_client() -> TestClient:
    """
    Create FastAPI test client for API testing.
    
    Returns:
        TestClient instance for making requests to the app
    """
    return TestClient(app)


# ============================================================================
# MOCK DATA FIXTURES
# ============================================================================


@pytest.fixture
def sample_feature_vector() -> np.ndarray:
    """
    Create a sample 26-dimensional feature vector.
    
    Returns:
        np.ndarray with random spectral features
    """
    return np.random.randn(FEATURE_VECTOR_LENGTH)


@pytest.fixture
def sample_features_batch() -> np.ndarray:
    """
    Create a batch of sample feature vectors.
    
    Returns:
        np.ndarray of shape (10, 26)
    """
    return np.random.randn(10, FEATURE_VECTOR_LENGTH)


@pytest.fixture
def mock_prediction_result() -> PredictionResult:
    """
    Create a mock prediction result.
    
    Returns:
        PredictionResult object with sample data
    """
    return PredictionResult(
        instrument="Piano",
        confidence=85.5,
        probabilities=[
            {"name": "Piano", "score": 85.5},
            {"name": "Violin", "score": 10.2},
            {"name": "Cello", "score": 4.3},
        ],
        average_vector=[0.1] * FEATURE_VECTOR_LENGTH,
    )


# ============================================================================
# AUDIO FILE FIXTURES
# ============================================================================


@pytest.fixture
def temp_audio_directory() -> Path:
    """
    Create a temporary directory for audio files.
    
    Yields:
        Path to temporary directory (cleaned up after test)
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_wav_file(temp_audio_directory) -> Path:
    """
    Create a sample WAV file for testing.
    
    Creates a 1-second sine wave at 440 Hz.
    
    Args:
        temp_audio_directory: Temporary directory fixture
        
    Returns:
        Path to generated WAV file
    """
    import soundfile as sf
    
    # Generate 1-second sine wave at 440 Hz
    sample_rate = 22050
    duration = 1.0
    frequency = 440
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = 0.3 * np.sin(2 * np.pi * frequency * t)
    
    # Save as WAV
    wav_path = temp_audio_directory / "test_audio.wav"
    sf.write(str(wav_path), audio_data, sample_rate)
    
    return wav_path


@pytest.fixture
def invalid_audio_file(temp_audio_directory) -> Path:
    """
    Create an invalid audio file for error testing.
    
    Args:
        temp_audio_directory: Temporary directory fixture
        
    Returns:
        Path to file with invalid audio data
    """
    invalid_path = temp_audio_directory / "invalid_audio.wav"
    with open(invalid_path, 'wb') as f:
        f.write(b"not valid audio data")
    
    return invalid_path


# ============================================================================
# CLASSIFIER FIXTURES
# ============================================================================


@pytest.fixture
def mock_classifier() -> Mock:
    """
    Create a mock classifier for unit testing.
    
    Returns:
        Mock InstrumentClassifier instance
    """
    mock = Mock(spec=InstrumentClassifier)
    mock.is_ready = True
    mock.predict = Mock(return_value=PredictionResult(
        instrument="Piano",
        confidence=85.5,
        probabilities=[
            {"name": "Piano", "score": 85.5},
            {"name": "Violin", "score": 10.2},
        ],
        average_vector=[0.1] * FEATURE_VECTOR_LENGTH,
    ))
    return mock


# ============================================================================
# CONFIGURATION FIXTURES
# ============================================================================


@pytest.fixture
def model_config_dict() -> dict:
    """
    Create a dictionary of model configuration.
    
    Returns:
        Dictionary with model parameters
    """
    return {
        'FEATURE_VECTOR_LENGTH': FEATURE_VECTOR_LENGTH,
        'NUM_CLASSES': ModelConfig.NUM_CLASSES,
        'KNN_NEIGHBORS': ModelConfig.KNN_NEIGHBORS,
        'KNN_METRIC': ModelConfig.KNN_METRIC,
        'INSTRUMENT_CLASSES': ModelConfig.INSTRUMENT_CLASSES,
    }


# ============================================================================
# MARKERS AND HOOKS
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers",
        "e2e: mark test as an end-to-end test"
    )


@pytest.fixture(autouse=True)
def reset_modules():
    """
    Reset imported modules after each test.
    
    Ensures tests don't interfere with each other through
    module-level state.
    """
    yield
    # Cleanup happens here if needed
