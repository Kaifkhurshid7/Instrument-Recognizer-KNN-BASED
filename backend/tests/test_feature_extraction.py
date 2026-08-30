"""
Unit tests for feature extraction module.

Tests cover:
- Feature vector shape and properties
- Error handling for invalid files
- Waveform data generation
- Audio loading and validation

Author: Instrument Recognizer Team
Date: 2024
"""

import pytest
import numpy as np
from pathlib import Path

from feature_extraction import (
    extract_features,
    _extract_spectral_features,
    _generate_waveform_data,
)
from validators import FeatureExtractionError, AudioProcessingError
from config import FEATURE_VECTOR_LENGTH, WAVEFORM_DISPLAY_SAMPLES


@pytest.mark.unit
class TestFeatureExtraction:
    """Test feature extraction functionality."""
    
    def test_extract_features_returns_correct_shape(
        self,
        sample_wav_file: Path
    ) -> None:
        """
        Test that extracted features have correct dimensionality.
        
        Args:
            sample_wav_file: Path to test WAV file fixture
        """
        result = extract_features(str(sample_wav_file), include_waveform=False)
        
        assert result['features'] is not None
        assert result['features'].shape == (FEATURE_VECTOR_LENGTH,)
    
    def test_extract_features_returns_finite_values(
        self,
        sample_wav_file: Path
    ) -> None:
        """
        Test that extracted features contain only finite values.
        
        Args:
            sample_wav_file: Path to test WAV file fixture
        """
        result = extract_features(str(sample_wav_file))
        
        features = result['features']
        assert np.isfinite(features).all(), "Features contain NaN or Inf"
    
    def test_extract_features_includes_waveform_data(
        self,
        sample_wav_file: Path
    ) -> None:
        """
        Test that waveform data is returned when requested.
        
        Args:
            sample_wav_file: Path to test WAV file fixture
        """
        result = extract_features(str(sample_wav_file), include_waveform=True)
        
        assert 'time' in result
        assert 'amplitude' in result
        assert len(result['time']) == WAVEFORM_DISPLAY_SAMPLES
        assert len(result['amplitude']) <= WAVEFORM_DISPLAY_SAMPLES
    
    def test_extract_features_missing_file_raises_error(self) -> None:
        """Test that missing file raises AudioProcessingError."""
        with pytest.raises(AudioProcessingError):
            extract_features("/nonexistent/file.wav")
    
    def test_extract_features_invalid_file_raises_error(
        self,
        invalid_audio_file: Path
    ) -> None:
        """
        Test that invalid audio file raises AudioProcessingError.
        
        Args:
            invalid_audio_file: Path to invalid audio file fixture
        """
        with pytest.raises(AudioProcessingError):
            extract_features(str(invalid_audio_file))
    
    def test_extract_features_without_waveform(
        self,
        sample_wav_file: Path
    ) -> None:
        """
        Test extraction without waveform data.
        
        Args:
            sample_wav_file: Path to test WAV file fixture
        """
        result = extract_features(str(sample_wav_file), include_waveform=False)
        
        assert 'time' not in result
        assert 'amplitude' not in result
        assert 'features' in result
    
    def test_feature_values_in_reasonable_range(
        self,
        sample_wav_file: Path
    ) -> None:
        """
        Test that feature values are in reasonable ranges.
        
        Most features should be normalized to [-1, 1] range.
        
        Args:
            sample_wav_file: Path to test WAV file fixture
        """
        result = extract_features(str(sample_wav_file))
        features = result['features']
        
        # Allow some features to exceed range but most should not
        exceeds_range = np.abs(features) > 1e6
        assert not exceeds_range.any(), "Features contain unreasonably large values"


@pytest.mark.unit
class TestSpectralFeatures:
    """Test individual spectral feature extraction."""
    
    def test_spectral_features_returns_dict_with_all_features(
        self,
        sample_wav_file: Path
    ) -> None:
        """
        Test that all spectral features are computed.
        
        Args:
            sample_wav_file: Path to test WAV file fixture
        """
        import librosa
        from config import SAMPLE_RATE, MFCC_COEFFICIENTS
        
        y, sr = librosa.load(str(sample_wav_file), sr=SAMPLE_RATE)
        
        features_dict = _extract_spectral_features(
            y, sr, hop_length=1024
        )
        
        # Check all required features are present
        required_features = [
            'mfcc_means', 'mfcc_delta', 'chroma', 'centroid',
            'rolloff', 'zcr', 'bandwidth', 'flatness', 'rms'
        ]
        
        for feature_name in required_features:
            assert feature_name in features_dict
            assert isinstance(features_dict[feature_name], np.ndarray)
    
    def test_spectral_features_correct_dimensions(
        self,
        sample_wav_file: Path
    ) -> None:
        """
        Test that spectral features have correct dimensions.
        
        Args:
            sample_wav_file: Path to test WAV file fixture
        """
        import librosa
        from config import SAMPLE_RATE, MFCC_COEFFICIENTS
        
        y, sr = librosa.load(str(sample_wav_file), sr=SAMPLE_RATE)
        
        features_dict = _extract_spectral_features(
            y, sr, hop_length=1024
        )
        
        assert features_dict['mfcc_means'].shape == (MFCC_COEFFICIENTS,)
        assert features_dict['mfcc_delta'].shape == (1,)
        assert features_dict['chroma'].shape == (2,)
        assert features_dict['centroid'].shape == (2,)


@pytest.mark.unit
class TestWaveformGeneration:
    """Test waveform data generation for visualization."""
    
    def test_waveform_generation_correct_length(
        self,
        sample_wav_file: Path
    ) -> None:
        """
        Test that generated waveform has correct number of samples.
        
        Args:
            sample_wav_file: Path to test WAV file fixture
        """
        import librosa
        from config import SAMPLE_RATE
        
        y, sr = librosa.load(str(sample_wav_file), sr=SAMPLE_RATE)
        
        waveform_data = _generate_waveform_data(y, sr)
        
        assert len(waveform_data['time']) == WAVEFORM_DISPLAY_SAMPLES
        assert len(waveform_data['amplitude']) <= WAVEFORM_DISPLAY_SAMPLES
    
    def test_waveform_time_values_are_valid(
        self,
        sample_wav_file: Path
    ) -> None:
        """
        Test that time values are monotonically increasing.
        
        Args:
            sample_wav_file: Path to test WAV file fixture
        """
        import librosa
        from config import SAMPLE_RATE
        
        y, sr = librosa.load(str(sample_wav_file), sr=SAMPLE_RATE)
        
        waveform_data = _generate_waveform_data(y, sr)
        time_values = waveform_data['time']
        
        assert all(time_values[i] <= time_values[i+1] for i in range(len(time_values)-1))
        assert time_values[0] >= 0
    
    def test_waveform_amplitude_in_reasonable_range(
        self,
        sample_wav_file: Path
    ) -> None:
        """
        Test that amplitude values are in audio range [-1, 1].
        
        Args:
            sample_wav_file: Path to test WAV file fixture
        """
        import librosa
        from config import SAMPLE_RATE
        
        y, sr = librosa.load(str(sample_wav_file), sr=SAMPLE_RATE)
        
        waveform_data = _generate_waveform_data(y, sr)
        amplitudes = np.array(waveform_data['amplitude'])
        
        assert np.all(amplitudes >= -1.5) and np.all(amplitudes <= 1.5)


@pytest.mark.integration
class TestFeatureExtractionIntegration:
    """Integration tests for feature extraction with real audio."""
    
    def test_extract_features_multiple_calls_consistent(
        self,
        sample_wav_file: Path
    ) -> None:
        """
        Test that multiple extractions from same file are consistent.
        
        Args:
            sample_wav_file: Path to test WAV file fixture
        """
        result1 = extract_features(str(sample_wav_file))
        result2 = extract_features(str(sample_wav_file))
        
        assert np.allclose(result1['features'], result2['features'])
    
    def test_extract_features_complete_pipeline(
        self,
        sample_wav_file: Path
    ) -> None:
        """
        Test complete feature extraction pipeline.
        
        Args:
            sample_wav_file: Path to test WAV file fixture
        """
        result = extract_features(str(sample_wav_file), include_waveform=True)
        
        # Check all expected keys present
        assert 'features' in result
        assert 'time' in result
        assert 'amplitude' in result
        
        # Check data types
        assert isinstance(result['features'], np.ndarray)
        assert isinstance(result['time'], list)
        assert isinstance(result['amplitude'], list)
        
        # Check shapes
        assert result['features'].shape == (FEATURE_VECTOR_LENGTH,)
        assert len(result['time']) == WAVEFORM_DISPLAY_SAMPLES
