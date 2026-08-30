"""
Spectral feature extraction for audio classification.

Produces a 26-dimensional fingerprint vector per audio file:
  [0-12]   MFCC 1-13 means (per-coefficient timbre profile)
  [13]     MFCC delta mean (temporal dynamics)
  [14-15]  Chroma mean/std (harmonic content)
  [16-17]  Spectral centroid mean/std (brightness)
  [18-19]  Spectral rolloff mean/std (high-freq energy boundary)
  [20-21]  Zero crossing rate mean/std (noisiness)
  [22-23]  Spectral bandwidth mean/std (richness)
  [24]     Spectral flatness (tonal vs noisy)
  [25]     RMS energy std (dynamic range)

This module handles:
- Audio loading with librosa
- Feature extraction using standard MIR techniques
- Error handling and validation
- Waveform visualization data generation

Author: Instrument Recognizer Team
Date: 2024
"""

import numpy as np
import librosa
from pathlib import Path
from typing import Dict, Optional, Tuple

from config import (
    SAMPLE_RATE,
    MAX_DURATION_SECONDS,
    MFCC_COEFFICIENTS,
    FEATURE_VECTOR_LENGTH,
    WAVEFORM_DISPLAY_SAMPLES,
)
from logger import setup_logger
from validators import FeatureExtractionError, AudioProcessingError

# Module-level logger
logger = setup_logger(__name__)


# ============================================================================
# FEATURE EXTRACTION CONFIGURATION
# ============================================================================


class FeatureExtractionConfig:
    """Configuration constants for feature extraction."""
    
    # FFT window size
    N_FFT: int = 2048
    
    # Hop length (number of samples between successive frames)
    HOP_LENGTH: int = 1024
    
    # Chroma feature n_fft
    CHROMA_N_FFT: int = 2048


# ============================================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================================


def extract_features(
    file_path: str,
    include_waveform: bool = False
) -> Dict[str, Optional[np.ndarray]]:
    """
    Extract a 26-dimensional spectral feature vector from an audio file.
    
    Performs comprehensive audio analysis including:
    - MFCC (Mel-Frequency Cepstral Coefficients)
    - Temporal dynamics (MFCC delta)
    - Harmonic content (Chroma)
    - Spectral characteristics (centroid, rolloff, bandwidth, flatness)
    - Energy dynamics (RMS)
    
    Args:
        file_path: Path to audio file (.mp3, .wav, .flac, etc.)
        include_waveform: If True, also returns waveform visualization data
        
    Returns:
        Dictionary containing:
            - 'features': np.ndarray of shape (26,) or None if extraction fails
            - 'time': list of time points (if include_waveform=True)
            - 'amplitude': list of amplitude values (if include_waveform=True)
            
    Raises:
        AudioProcessingError: If audio file cannot be loaded
        FeatureExtractionError: If feature extraction fails
        
    Example:
        >>> result = extract_features("song.mp3", include_waveform=True)
        >>> if result['features'] is not None:
        ...     print(f"Extracted {len(result['features'])} features")
        ...     print(f"Waveform: {len(result['time'])} samples")
    """
    
    file_path_obj = Path(file_path)
    
    try:
        # Validate file exists
        if not file_path_obj.exists():
            raise AudioProcessingError(f"File not found: {file_path}")
        
        # Log extraction start
        logger.info(
            "feature_extraction_started",
            extra={'file': str(file_path_obj), 'file_size_kb': file_path_obj.stat().st_size / 1024}
        )
        
        # Load audio file with librosa
        try:
            y: np.ndarray
            sr: int
            y, sr = librosa.load(
                file_path,
                sr=SAMPLE_RATE,
                duration=MAX_DURATION_SECONDS
            )
        except Exception as e:
            raise AudioProcessingError(
                f"Failed to load audio file: {e}. "
                f"Ensure file is valid audio format and not corrupted."
            )
        
        # Validate loaded audio
        if len(y) == 0:
            raise AudioProcessingError("Audio file is empty or corrupted")
        
        # Check audio duration
        duration = len(y) / sr
        if duration < 0.1:
            raise AudioProcessingError(
                f"Audio too short ({duration:.2f}s). Minimum 0.1s required."
            )
        
        logger.debug(
            "audio_loaded",
            extra={'sample_rate': sr, 'duration_seconds': duration}
        )
        
        # Extract spectral features
        features_dict = _extract_spectral_features(
            y,
            sr,
            FeatureExtractionConfig.HOP_LENGTH
        )
        
        # Concatenate all features into single vector
        feature_vector: np.ndarray = np.concatenate(list(features_dict.values()))
        
        # Validate feature vector
        if feature_vector.shape[0] != FEATURE_VECTOR_LENGTH:
            raise FeatureExtractionError(
                f"Feature vector dimension mismatch. "
                f"Expected {FEATURE_VECTOR_LENGTH}, got {feature_vector.shape[0]}"
            )
        
        # Check for NaN or infinite values
        if not np.isfinite(feature_vector).all():
            raise FeatureExtractionError(
                "Feature vector contains NaN or infinite values. "
                "Audio may be corrupted or contains silence."
            )
        
        result: Dict[str, Optional[np.ndarray]] = {"features": feature_vector}
        
        # Generate waveform visualization if requested
        if include_waveform:
            logger.debug("generating_waveform_data")
            waveform_data = _generate_waveform_data(y, sr)
            result.update(waveform_data)
        
        logger.info(
            "feature_extraction_completed",
            extra={'feature_dimension': feature_vector.shape[0]}
        )
        
        return result
        
    except (AudioProcessingError, FeatureExtractionError):
        # Re-raise our custom exceptions
        raise
    except Exception as e:
        # Catch unexpected errors and wrap them
        logger.error(
            "feature_extraction_failed",
            extra={'file': str(file_path), 'error': str(e)},
            exc_info=True
        )
        raise FeatureExtractionError(
            f"Unexpected error during feature extraction: {e}"
        )


def _extract_spectral_features(
    y: np.ndarray,
    sr: int,
    hop_length: int
) -> Dict[str, np.ndarray]:
    """
    Extract individual spectral features from audio waveform.
    
    Args:
        y: Audio time series
        sr: Sample rate
        hop_length: Number of samples between successive frames
        
    Returns:
        Dictionary of feature arrays
        
    Raises:
        FeatureExtractionError: If extraction fails
    """
    
    try:
        config = FeatureExtractionConfig
        
        # MFCC - Mel-Frequency Cepstral Coefficients (13 coefficients)
        mfccs: np.ndarray = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=MFCC_COEFFICIENTS,
            hop_length=hop_length
        )
        mfcc_means: np.ndarray = np.mean(mfccs, axis=1)
        
        # MFCC Delta - Temporal dynamics
        mfcc_delta: np.ndarray = librosa.feature.delta(mfccs)
        mfcc_delta_mean: np.ndarray = np.array([np.mean(mfcc_delta)])
        
        # Chroma - Harmonic content
        chroma: np.ndarray = librosa.feature.chroma_stft(
            y=y,
            sr=sr,
            hop_length=hop_length,
            n_fft=config.N_FFT
        )
        chroma_features: np.ndarray = np.array([np.mean(chroma), np.std(chroma)])
        
        # Spectral Centroid - Brightness
        centroid: np.ndarray = librosa.feature.spectral_centroid(
            y=y,
            sr=sr,
            hop_length=hop_length
        )[0]
        centroid_features: np.ndarray = np.array([np.mean(centroid), np.std(centroid)])
        
        # Spectral Rolloff - High-frequency energy boundary
        rolloff: np.ndarray = librosa.feature.spectral_rolloff(
            y=y,
            sr=sr,
            hop_length=hop_length
        )[0]
        rolloff_features: np.ndarray = np.array([np.mean(rolloff), np.std(rolloff)])
        
        # Zero Crossing Rate - Noisiness indicator
        zcr: np.ndarray = librosa.feature.zero_crossing_rate(
            y=y,
            hop_length=hop_length
        )[0]
        zcr_features: np.ndarray = np.array([np.mean(zcr), np.std(zcr)])
        
        # Spectral Bandwidth - Richness of spectrum
        bandwidth: np.ndarray = librosa.feature.spectral_bandwidth(
            y=y,
            sr=sr,
            hop_length=hop_length
        )[0]
        bandwidth_features: np.ndarray = np.array([np.mean(bandwidth), np.std(bandwidth)])
        
        # Spectral Flatness - Tonal vs Noisy indicator
        flatness: np.ndarray = librosa.feature.spectral_flatness(
            y=y,
            hop_length=hop_length
        )[0]
        flatness_features: np.ndarray = np.array([np.mean(flatness)])
        
        # RMS Energy - Dynamic range
        rms: np.ndarray = librosa.feature.rms(
            y=y,
            hop_length=hop_length
        )[0]
        rms_features: np.ndarray = np.array([np.std(rms)])
        
        return {
            'mfcc_means': mfcc_means,
            'mfcc_delta': mfcc_delta_mean,
            'chroma': chroma_features,
            'centroid': centroid_features,
            'rolloff': rolloff_features,
            'zcr': zcr_features,
            'bandwidth': bandwidth_features,
            'flatness': flatness_features,
            'rms': rms_features,
        }
        
    except Exception as e:
        raise FeatureExtractionError(
            f"Failed to extract spectral features: {e}"
        )


def _generate_waveform_data(y: np.ndarray, sr: int) -> Dict[str, list]:
    """
    Generate downsampled waveform data for visualization.
    
    Args:
        y: Audio time series
        sr: Sample rate
        
    Returns:
        Dictionary with 'time' and 'amplitude' lists
        
    Raises:
        FeatureExtractionError: If generation fails
    """
    
    try:
        num_samples: int = WAVEFORM_DISPLAY_SAMPLES
        duration: float = len(y) / sr
        
        # Generate time array
        time_array: list = np.linspace(
            0, duration, num=num_samples
        ).tolist()
        
        # Downsample amplitude
        step: int = max(1, len(y) // num_samples)
        amplitude_array: list = y[::step][:num_samples].tolist()
        
        return {
            "time": time_array,
            "amplitude": amplitude_array,
        }
        
    except Exception as e:
        logger.warning(f"Failed to generate waveform data: {e}")
        return {
            "time": [],
            "amplitude": [],
        }
