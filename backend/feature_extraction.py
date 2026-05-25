"""
Spectral feature extraction for audio classification.

Produces a 26-dimensional fingerprint vector per audio file:
  [0-12]  MFCC 1-13 means (per-coefficient timbre profile)
  [13]    MFCC delta mean (temporal dynamics)
  [14-15] Chroma mean/std (harmonic content)
  [16-17] Spectral centroid mean/std (brightness)
  [18-19] Spectral rolloff mean/std (high-freq energy boundary)
  [20-21] Zero crossing rate mean/std (noisiness)
  [22-23] Spectral bandwidth mean/std (richness)
  [24]    Spectral flatness (tonal vs noisy)
  [25]    RMS energy std (dynamic range)
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
    Extract a 26-dimensional spectral feature vector from an audio file.

    Args:
        file_path: Path to the audio file (WAV preferred).
        include_waveform: If True, returns downsampled waveform for visualization.

    Returns:
        dict with 'features' (np.ndarray or None), and optionally 'time'/'amplitude'.
    """
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=MAX_DURATION_SECONDS)

        # hop=1024 reduces spectrogram matrix size (memory optimization)
        hop = 1024

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_COEFFICIENTS, hop_length=hop)
        mfcc_delta = librosa.feature.delta(mfccs)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop, n_fft=2048)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop)
        zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop)
        flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop)
        rms = librosa.feature.rms(y=y, hop_length=hop)

        feature_vector = np.concatenate([
            np.mean(mfccs, axis=1),              # 13 MFCC means
            [np.mean(mfcc_delta)],               # temporal dynamics
            [np.mean(chroma), np.std(chroma)],
            [np.mean(centroid), np.std(centroid)],
            [np.mean(rolloff), np.std(rolloff)],
            [np.mean(zcr), np.std(zcr)],
            [np.mean(bandwidth), np.std(bandwidth)],
            [np.mean(flatness)],                 # tonal vs noisy
            [np.std(rms)],                       # dynamic range
        ])

        if feature_vector.shape[0] != FEATURE_VECTOR_LENGTH:
            raise ValueError(
                f"Expected {FEATURE_VECTOR_LENGTH} features, got {feature_vector.shape[0]}"
            )

        result = {"features": feature_vector}

        if include_waveform:
            num_samples = WAVEFORM_DISPLAY_SAMPLES
            result["time"] = np.linspace(0, len(y) / sr, num=num_samples).tolist()
            step = max(1, len(y) // num_samples)
            result["amplitude"] = y[::step][:num_samples].tolist()

        return result

    except Exception as e:
        print(f"[Feature Extraction] Error: {file_path} — {e}")
        return {"features": None}
