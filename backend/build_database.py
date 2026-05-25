"""
Build the reference fingerprint database from IRMAS training data.

Usage:
    python build_database.py

Scans IRMAS-TrainingData/ for instrument folders, extracts features
from each WAV file, and saves the result to reference_database.pkl.
"""

import os
import time
import pickle
import numpy as np

from config import DATASET_PATH, DATABASE_FILE, FEATURE_VECTOR_LENGTH
from feature_extraction import extract_features


def build_database():
    """Process all instrument folders and compile the reference database."""
    database = {}
    start_time = time.time()

    print("[Database Builder] Scanning IRMAS dataset...")

    if not os.path.exists(DATASET_PATH):
        print(f"[Error] Dataset not found: {DATASET_PATH}")
        return

    instrument_folders = sorted(
        d for d in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, d))
    )

    for instrument in instrument_folders:
        folder_path = os.path.join(DATASET_PATH, instrument)
        wav_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".wav")]

        print(f"\n  Processing: {instrument.upper()} ({len(wav_files)} files)")

        fingerprints = []
        for idx, filename in enumerate(wav_files, 1):
            file_path = os.path.join(folder_path, filename)
            result = extract_features(file_path)

            if result["features"] is not None:
                fingerprints.append(result["features"])

            if idx % 50 == 0 or idx == len(wav_files):
                print(f"    {idx}/{len(wav_files)} processed", end="\r")

        print(f"    Collected {len(fingerprints)} fingerprints")

        if fingerprints:
            database[instrument] = {
                "fingerprints": fingerprints,
                "average_vector": np.mean(fingerprints, axis=0),
            }

    if not database:
        print("\n[Error] No data collected. Check dataset structure.")
        return

    with open(DATABASE_FILE, "wb") as f:
        pickle.dump(database, f)

    elapsed = time.time() - start_time
    total_samples = sum(len(d["fingerprints"]) for d in database.values())

    print(f"\n[Done] {len(database)} classes | {total_samples} samples | {elapsed:.1f}s")
    print(f"  Saved: {DATABASE_FILE}")


if __name__ == "__main__":
    build_database()
