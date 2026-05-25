"""
Feature Extraction Module (Enhanced)
--------------------------------------
Extracts a 26-dimensional spectral fingerprint from audio files.
This module is shared between the API server and the database builder
to ensure consistent feature computation.

Key improvement over the original 10-D vector:
  - Individual MFCC coefficients (1-13 mean) capture distinct timbre bands
  - Delta MFCCs capture temporal dynamics (attack, decay patterns)
  - Additional spectral shape descriptors improve class separation

Feature Vector Layout (26 dimensions):
  [0-12]  MFCC 1-13 Mean       - Per-coefficient timbre profile
  [13]    MFCC Delta Mean       - Temporal dynamics (attack/sustain)
  [14]    Chroma Mean           - Harmonic content
  [15]    Chroma Std            - Harmony variation
  [16]    Spectral Centroid     - Brightness
  [17]    Centroid Std          - Brightness variation
  [18]    Spectral Rolloff      - High-frequency energy boundary
  [19]    Rolloff Std           - Rolloff variation
  [20]    Zero Crossing Rate    - Noisiness / percussiveness
  [21]    ZCR Std               - Noisiness variation
  [22]    Spectral Bandwidth    - Richness of sound
  [23]    Bandwidth Std         - Richness variation
  [24]    Spectral Flatness     - Tonal vs noisy (key for wind instruments)
  [25]    RMS Energy Std        - Dynamic range / envelope shape
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
    Extract an enhanced 26-dimensional feature vector from an audio file.

    The vector captures:
      - Timbre (individual MFCCs + delta)
      - Harmony (chroma)
      - Spectral shape (centroid, rolloff, bandwidth, flatness)
      - Temporal dynamics (ZCR std, RMS std, MFCC delta)

    Parameters
    ----------
    file_path : str
        Path to the audio file (WAV format preferred).
    include_waveform : bool
        If True, also returns downsampled time/amplitude arrays for visualization.

    Returns
    -------
    dict with keys:
        'features'  : np.ndarray of shape (FEATURE_VECTOR_LENGTH,) or None on failure
        'time'      : list[float] (only if include_waveform=True)
        'amplitude' : list[float] (only if include_waveform=True)
    """
    try:
        # Load audio with consistent sample rate and duration cap
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=MAX_DURATION_SECONDS)

        # Larger hop_length to reduce memory usage on long files
        hop = 1024

        # --- Core spectral features ---
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_COEFFICIENTS, hop_length=hop)
        mfcc_delta = librosa.feature.delta(mfccs)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop, n_fft=2048)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop)
        zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop)
        flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop)
        rms = librosa.feature.rms(y=y, hop_length=hop)

        # --- Build the 26-dimensional fingerprint ---
        feature_vector = np.concatenate([
            # Individual MFCC means (13 values) - captures distinct timbre bands
            np.mean(mfccs, axis=1),

            # MFCC delta mean - captures temporal dynamics
            [np.mean(mfcc_delta)],

            # Chroma (harmonic content)
            [np.mean(chroma), np.std(chroma)],

            # Spectral centroid (brightness)
            [np.mean(centroid), np.std(centroid)],

            # Spectral rolloff (high-freq energy boundary)
            [np.mean(rolloff), np.std(rolloff)],

            # Zero crossing rate (noisiness / percussiveness)
            [np.mean(zcr), np.std(zcr)],

            # Spectral bandwidth (richness)
            [np.mean(bandwidth), np.std(bandwidth)],

            # Spectral flatness (tonal vs noisy - great for wind vs string)
            [np.mean(flatness)],

            # RMS energy std (dynamic range / envelope shape)
            [np.std(rms)],
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
