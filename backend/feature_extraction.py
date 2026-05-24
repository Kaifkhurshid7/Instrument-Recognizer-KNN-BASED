"""
Feature Extraction Module
-------------------------
Extracts a 10-dimensional spectral fingerprint from audio files.
This module is shared between the API server and the database builder
to ensure consistent feature computation.

Feature Vector Layout:
  [0] MFCC Mean         - Average timbre / texture
  [1] MFCC Std          - Timbre variation
  [2] Chroma Mean       - Average harmonic content
  [3] Chroma Std        - Harmony variation
  [4] Spectral Centroid - Brightness of sound
  [5] Centroid Std      - Brightness variation
  [6] Spectral Rolloff  - Spectral power shape
  [7] Rolloff Std       - Rolloff variation
  [8] Zero Crossing Rate- Noisiness measure
  [9] Spectral Bandwidth- Richness of sound
"""

import numpy as np
import librosa

from config import (
    SAMPLE_RATE,
    MAX_DURATION_SECONDS,
    MFCC_COEFFICIENTS,
    FEATURE_VECTOR_LENGTH,
    WAVEFORM_DISPLAY_SAMPLES,
)


def extract_features(file_path, include_waveform=False):
    """
    Extract a normalized 10-dimensional feature vector from an audio file.

    Parameters
    ----------
    file_path : str
        Path to the audio file (WAV format preferred).
    include_waveform : bool
        If True, also returns downsampled time/amplitude arrays for visualization.

    Returns
    -------
    dict with keys:
        'features'  : np.ndarray of shape (10,) or None on failure
        'time'      : list[float] (only if include_waveform=True)
        'amplitude' : list[float] (only if include_waveform=True)
    """
    try:
        # Load audio with consistent sample rate and duration cap
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=MAX_DURATION_SECONDS)

        # Use a larger hop_length to reduce memory allocation for spectrograms.
        # Default hop=512 on long files creates huge matrices; 1024 halves memory usage.
        hop = 1024

        # Compute spectral features with memory-safe parameters
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_COEFFICIENTS, hop_length=hop)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop, n_fft=2048)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop)
        zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop)

        # Build the 10-dimensional fingerprint
        feature_vector = np.array([
            np.mean(mfccs), np.std(mfccs),
            np.mean(chroma), np.std(chroma),
            np.mean(centroid), np.std(centroid),
            np.mean(rolloff), np.std(rolloff),
            np.mean(zcr),
            np.mean(bandwidth),
        ])

        if feature_vector.shape[0] != FEATURE_VECTOR_LENGTH:
            raise ValueError(
                f"Expected {FEATURE_VECTOR_LENGTH} features, got {feature_vector.shape[0]}"
            )

        result = {"features": feature_vector}

        # Optional waveform data for frontend visualization
        if include_waveform:
            num_samples = WAVEFORM_DISPLAY_SAMPLES
            result["time"] = np.linspace(0, len(y) / sr, num=num_samples).tolist()
            step = max(1, len(y) // num_samples)
            result["amplitude"] = y[::step][:num_samples].tolist()

        return result

    except Exception as e:
        print(f"[Feature Extraction] Error processing {file_path}: {e}")
        return {"features": None}
