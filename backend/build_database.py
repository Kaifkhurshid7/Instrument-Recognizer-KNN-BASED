import os
import librosa
import numpy as np
import pickle
import time

# =========================================================
# PATH-SAFE CONFIGURATION
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'IRMAS-TrainingData')
DATABASE_FILE = os.path.join(BASE_DIR, 'reference_database.pkl')
FEATURE_LENGTH = 10

# =========================================================
# FEATURE EXTRACTION 
# =========================================================
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050, duration=30)

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)

        feature_vector = np.array([
            np.mean(mfccs), np.std(mfccs),
            np.mean(chroma), np.std(chroma),
            np.mean(spec_centroid), np.std(spec_centroid),
            np.mean(spec_rolloff), np.std(spec_rolloff),
            np.mean(zcr),
            np.mean(spec_bw)
        ])

        if feature_vector.shape[0] != FEATURE_LENGTH:
            return None

        return feature_vector

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# =========================================================
# BUILD DATABASE
# =========================================================
def build_database():
    database = {}
    start_time = time.time()

    print(" Building reference database from IRMAS dataset...")

    if not os.path.exists(DATASET_PATH):
        print(f" Dataset path not found: {DATASET_PATH}")
        return

    for instrument_folder in sorted(os.listdir(DATASET_PATH)):
        folder_path = os.path.join(DATASET_PATH, instrument_folder)

        if not os.path.isdir(folder_path):
            continue

        print(f"\n Instrument: {instrument_folder.upper()}")

        fingerprints = []
        wav_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.wav')]

        for idx, filename in enumerate(wav_files):
            file_path = os.path.join(folder_path, filename)
            features = extract_features(file_path)

            if features is not None:
                fingerprints.append(features)

            print(f"  Processing {idx + 1}/{len(wav_files)}", end='\r')

        print(f"\n  Collected {len(fingerprints)} fingerprints")

        if fingerprints:
            database[instrument_folder] = {
                'fingerprints': fingerprints,
                'average_vector': np.mean(fingerprints, axis=0)
            }

    if not database:
        print("\nDatabase empty. Check dataset structure.")
        return

    with open(DATABASE_FILE, 'wb') as f:
        pickle.dump(database, f)

    elapsed = time.time() - start_time
    print("\n Database build complete!")
    print(f"Saved to: {DATABASE_FILE}")
    print(f"Time taken: {elapsed:.2f} seconds")

# =========================================================
# RUN SCRIPT
# =========================================================
if __name__ == '__main__':
    build_database()
