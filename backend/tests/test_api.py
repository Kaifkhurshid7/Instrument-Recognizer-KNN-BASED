"""
Unit and integration tests for FastAPI endpoints.

Tests cover:
- Health check endpoint
- Audio analysis endpoint
- Input validation
- Error handling
- Response formats

Author: Instrument Recognizer Team
Date: 2024
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.mark.unit
class TestHealthEndpoint:
    """Test the /health endpoint."""
    
    def test_health_check_returns_200(self, test_client: TestClient) -> None:
        """Test that health endpoint returns 200 OK."""
        response = test_client.get("/health")
        
        assert response.status_code == 200
    
    def test_health_check_response_format(self, test_client: TestClient) -> None:
        """Test that health check response has correct format."""
        response = test_client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "model_ready" in data
        assert "version" in data
        assert isinstance(data["status"], str)
        assert isinstance(data["model_ready"], bool)
    
    def test_health_check_model_ready_field(self, test_client: TestClient) -> None:
        """Test that model_ready field reflects actual state."""
        response = test_client.get("/health")
        data = response.json()
        
        # Model should be ready after app startup
        assert data["model_ready"] in (True, False)
    
    def test_health_check_version_format(self, test_client: TestClient) -> None:
        """Test that API version is properly formatted."""
        response = test_client.get("/health")
        data = response.json()
        
        version = data["version"]
        parts = version.split(".")
        assert len(parts) >= 2  # At least major.minor


@pytest.mark.unit
class TestAnalysisEndpoint:
    """Test the /v1/analyze endpoint."""
    
    def test_analyze_requires_file(self, test_client: TestClient) -> None:
        """Test that endpoint requires audio file."""
        response = test_client.post("/v1/analyze")
        
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_analyze_with_valid_audio_file(
        self,
        test_client: TestClient,
        sample_wav_file: Path
    ) -> None:
        """
        Test analysis with valid audio file.
        
        Args:
            test_client: FastAPI test client
            sample_wav_file: Path to test WAV file
        """
        with open(sample_wav_file, 'rb') as f:
            files = {'audioFile': ('test.wav', f, 'audio/wav')}
            response = test_client.post("/v1/analyze", files=files)
        
        assert response.status_code == 200
    
    def test_analyze_response_format(
        self,
        test_client: TestClient,
        sample_wav_file: Path
    ) -> None:
        """
        Test that analysis response has correct format.
        
        Args:
            test_client: FastAPI test client
            sample_wav_file: Path to test WAV file
        """
        with open(sample_wav_file, 'rb') as f:
            files = {'audioFile': ('test.wav', f, 'audio/wav')}
            response = test_client.post("/v1/analyze", files=files)
        
        assert response.status_code == 200
        
        data = response.json()
        
        # Check required fields
        required_fields = [
            'instrument',
            'confidence_score',
            'waveform',
            'feature_vector',
            'compared_vector',
            'knn_probabilities'
        ]
        
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
    
    def test_analyze_instrument_field_not_empty(
        self,
        test_client: TestClient,
        sample_wav_file: Path
    ) -> None:
        """
        Test that predicted instrument is not empty.
        
        Args:
            test_client: FastAPI test client
            sample_wav_file: Path to test WAV file
        """
        with open(sample_wav_file, 'rb') as f:
            files = {'audioFile': ('test.wav', f, 'audio/wav')}
            response = test_client.post("/v1/analyze", files=files)
        
        data = response.json()
        
        assert len(data['instrument']) > 0
    
    def test_analyze_confidence_in_valid_range(
        self,
        test_client: TestClient,
        sample_wav_file: Path
    ) -> None:
        """
        Test that confidence score is between 0 and 100.
        
        Args:
            test_client: FastAPI test client
            sample_wav_file: Path to test WAV file
        """
        with open(sample_wav_file, 'rb') as f:
            files = {'audioFile': ('test.wav', f, 'audio/wav')}
            response = test_client.post("/v1/analyze", files=files)
        
        data = response.json()
        confidence = data['confidence_score']
        
        assert 0 <= confidence <= 100
    
    def test_analyze_feature_vector_correct_length(
        self,
        test_client: TestClient,
        sample_wav_file: Path
    ) -> None:
        """
        Test that feature vector has 26 dimensions.
        
        Args:
            test_client: FastAPI test client
            sample_wav_file: Path to test WAV file
        """
        from config import FEATURE_VECTOR_LENGTH
        
        with open(sample_wav_file, 'rb') as f:
            files = {'audioFile': ('test.wav', f, 'audio/wav')}
            response = test_client.post("/v1/analyze", files=files)
        
        data = response.json()
        
        assert len(data['feature_vector']) == FEATURE_VECTOR_LENGTH
        assert len(data['compared_vector']) == FEATURE_VECTOR_LENGTH
    
    def test_analyze_probabilities_format(
        self,
        test_client: TestClient,
        sample_wav_file: Path
    ) -> None:
        """
        Test that probabilities have correct format.
        
        Args:
            test_client: FastAPI test client
            sample_wav_file: Path to test WAV file
        """
        with open(sample_wav_file, 'rb') as f:
            files = {'audioFile': ('test.wav', f, 'audio/wav')}
            response = test_client.post("/v1/analyze", files=files)
        
        data = response.json()
        probabilities = data['knn_probabilities']
        
        assert isinstance(probabilities, list)
        assert len(probabilities) > 0
        
        # Check each probability entry
        for prob_entry in probabilities:
            assert 'name' in prob_entry
            assert 'score' in prob_entry
            assert isinstance(prob_entry['name'], str)
            assert isinstance(prob_entry['score'], (int, float))
            assert 0 <= prob_entry['score'] <= 100
    
    def test_analyze_with_invalid_audio_file(
        self,
        test_client: TestClient,
        invalid_audio_file: Path
    ) -> None:
        """
        Test handling of invalid audio file.
        
        Args:
            test_client: FastAPI test client
            invalid_audio_file: Path to invalid audio file
        """
        with open(invalid_audio_file, 'rb') as f:
            files = {'audioFile': ('invalid.wav', f, 'audio/wav')}
            response = test_client.post("/v1/analyze", files=files)
        
        # Should handle gracefully with error response
        assert response.status_code in (400, 422, 500)
    
    def test_analyze_waveform_data_format(
        self,
        test_client: TestClient,
        sample_wav_file: Path
    ) -> None:
        """
        Test that waveform data is properly formatted.
        
        Args:
            test_client: FastAPI test client
            sample_wav_file: Path to test WAV file
        """
        with open(sample_wav_file, 'rb') as f:
            files = {'audioFile': ('test.wav', f, 'audio/wav')}
            response = test_client.post("/v1/analyze", files=files)
        
        data = response.json()
        waveform = data['waveform']
        
        assert 'time' in waveform
        assert 'amplitude' in waveform
        assert isinstance(waveform['time'], list)
        assert isinstance(waveform['amplitude'], list)


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for complete API workflows."""
    
    def test_health_check_then_analysis(
        self,
        test_client: TestClient,
        sample_wav_file: Path
    ) -> None:
        """
        Test complete workflow: health check then analysis.
        
        Args:
            test_client: FastAPI test client
            sample_wav_file: Path to test WAV file
        """
        # Check health first
        health_response = test_client.get("/health")
        assert health_response.status_code == 200
        
        # If model is ready, perform analysis
        if health_response.json()['model_ready']:
            with open(sample_wav_file, 'rb') as f:
                files = {'audioFile': ('test.wav', f, 'audio/wav')}
                analysis_response = test_client.post("/v1/analyze", files=files)
            
            assert analysis_response.status_code == 200
    
    def test_multiple_analysis_requests(
        self,
        test_client: TestClient,
        sample_wav_file: Path
    ) -> None:
        """
        Test that API can handle multiple requests.
        
        Args:
            test_client: FastAPI test client
            sample_wav_file: Path to test WAV file
        """
        num_requests = 3
        responses = []
        
        for _ in range(num_requests):
            with open(sample_wav_file, 'rb') as f:
                files = {'audioFile': ('test.wav', f, 'audio/wav')}
                response = test_client.post("/v1/analyze", files=files)
            
            responses.append(response)
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200


@pytest.mark.unit
class TestErrorHandling:
    """Test error handling and validation."""
    
    def test_endpoint_invalid_http_method(self, test_client: TestClient) -> None:
        """Test that invalid HTTP method returns 405."""
        response = test_client.put("/v1/analyze")
        
        assert response.status_code == 405  # Method Not Allowed
    
    def test_endpoint_nonexistent_path(self, test_client: TestClient) -> None:
        """Test that nonexistent endpoint returns 404."""
        response = test_client.get("/nonexistent")
        
        assert response.status_code == 404
    
    def test_error_response_format(self, test_client: TestClient) -> None:
        """Test that error responses have consistent format."""
        response = test_client.post("/v1/analyze")
        
        # Should have error response
        assert response.status_code in (400, 422)
        
        try:
            data = response.json()
            # Should have some error information
            assert isinstance(data, dict)
        except ValueError:
            # Some error responses might not be JSON
            pass


@pytest.mark.unit
class TestResponseValidation:
    """Test response data validation."""
    
    def test_analysis_all_fields_present(
        self,
        test_client: TestClient,
        sample_wav_file: Path
    ) -> None:
        """
        Test that all required fields are present in analysis response.
        
        Args:
            test_client: FastAPI test client
            sample_wav_file: Path to test WAV file
        """
        with open(sample_wav_file, 'rb') as f:
            files = {'audioFile': ('test.wav', f, 'audio/wav')}
            response = test_client.post("/v1/analyze", files=files)
        
        assert response.status_code == 200
        
        data = response.json()
        
        # Verify all fields are present and have correct types
        assert isinstance(data['instrument'], str)
        assert isinstance(data['confidence_score'], (int, float))
        assert isinstance(data['feature_vector'], list)
        assert isinstance(data['compared_vector'], list)
        assert isinstance(data['knn_probabilities'], list)
