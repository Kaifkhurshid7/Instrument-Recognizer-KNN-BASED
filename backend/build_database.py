"""
Reference Database Builder
---------------------------
Processes the IRMAS training dataset to extract spectral fingerprints
for each instrument class. Outputs a pickle file used by the classifier.

Usage:
    python build_database.py

The script scans IRMAS-TrainingData/ for instrument folders, extracts
features from each WAV file, and stores them in reference_database.pkl.
"""

import os
import time
import pickle
import numpy as np

from config import DATASET_PATH, DATABASE_FILE, FEATURE_VECTOR_LENGTH
from feature_extraction import extract_features


def build_database():
    """
    Iterate through all instrument folders, extract features from each
    audio sample, and save the compiled database to disk.
    """
    database = {}
    start_time = time.time()

    print("[Database Builder] Scanning IRMAS dataset...")

    if not os.path.exists(DATASET_PATH):
        print(f"[Error] Dataset not found at: {DATASET_PATH}")
        print("  Download IRMAS-TrainingData and place it in the backend/ folder.")
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

            # Progress indicator
            if idx % 50 == 0 or idx == len(wav_files):
                print(f"    {idx}/{len(wav_files)} processed", end="\r")

        print(f"    Collected {len(fingerprints)} valid fingerprints")

        if fingerprints:
            database[instrument] = {
                "fingerprints": fingerprints,
                "average_vector": np.mean(fingerprints, axis=0),
            }

    # --- Save ---
    if not database:
        print("\n[Error] No data collected. Check dataset structure.")
        return

    with open(DATABASE_FILE, "wb") as f:
        pickle.dump(database, f)

    elapsed = time.time() - start_time
    total_samples = sum(len(d["fingerprints"]) for d in database.values())

    print(f"\n[Database Builder] Complete!")
    print(f"  Classes: {len(database)}")
    print(f"  Total samples: {total_samples}")
    print(f"  Saved to: {DATABASE_FILE}")
    print(f"  Time: {elapsed:.1f}s")


if __name__ == "__main__":
    build_database()
