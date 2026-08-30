"""
Tests package for Instrument Recognizer backend.

This package contains all unit and integration tests for the
backend application.

Test Organization:
    - test_feature_extraction.py: Feature extraction tests
    - test_api.py: API endpoint tests
    - test_classifier.py: Classifier tests (future)
    - test_validators.py: Validation tests (future)
    - conftest.py: Shared fixtures and configuration

Running Tests:
    pytest                          # Run all tests
    pytest -v                       # Verbose output
    pytest -m unit                  # Only unit tests
    pytest -m integration           # Only integration tests
    pytest --cov=backend            # With coverage report

Author: Instrument Recognizer Team
Date: 2024
"""
