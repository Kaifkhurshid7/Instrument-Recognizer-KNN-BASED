from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import pickle
import os
from pydub import AudioSegment
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# =========================================================
# PATH-SAFE CONFIGURATION
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_FILE = os.path.join(BASE_DIR, 'reference_database.pkl')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)

# =========================================================
# GLOBAL MODEL OBJECTS
# =========================================================
knn_classifier = None
scaler = None
class_names = None
reference_database = None
BEST_K_VALUE = 7

# =========================================================
# FEATURE EXTRACTION (10-DIMENSION FINGERPRINT)
# =========================================================
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050, duration=30)

        num_samples = 1000
        time = np.linspace(0, len(y) / sr, num=num_samples)
        amplitude = y[::max(1, len(y) // num_samples)][:num_samples]

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

        if feature_vector.shape[0] != 10:
            raise ValueError("Feature vector size mismatch")

        return feature_vector, time.tolist(), amplitude.tolist()

    except Exception as e:
        print("Feature extraction error:", e)
        return None, None, None

# =========================================================
# LOAD DATABASE + TRAIN KNN (SERVER STARTUP)
# =========================================================
def load_and_train_knn():
    global knn_classifier, scaler, class_names, reference_database

    print("Loading reference database...")

    with open(DATABASE_FILE, 'rb') as f:
        reference_database = pickle.load(f)

    X, y = [], []
    class_names = sorted(reference_database.keys())
    class_map = {name: i for i, name in enumerate(class_names)}

    for instrument, data in reference_database.items():
        for vec in data['fingerprints']:
            if len(vec) == 10:
                X.append(vec)
                y.append(class_map[instrument])

    X = np.array(X)
    y = np.array(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    knn_classifier = KNeighborsClassifier(
        n_neighbors=BEST_K_VALUE,
        metric='cosine',
        weights='distance'
    )
    knn_classifier.fit(X_scaled, y)

    print(f" Model ready | Classes: {len(class_names)} | Samples: {len(y)}")

# =========================================================
# ANALYZE ENDPOINT
# =========================================================
@app.route('/analyze', methods=['POST'])
def analyze_audio():
    if knn_classifier is None or reference_database is None:
        return jsonify({'error': 'Server not initialized'}), 500

    if 'audioFile' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['audioFile']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    raw_path = os.path.join(UPLOAD_FOLDER, file.filename)
    wav_path = os.path.join(UPLOAD_FOLDER, 'temp.wav')

    file.save(raw_path)

    try:
        audio = AudioSegment.from_file(raw_path)
        audio.set_channels(1).set_frame_rate(22050).export(wav_path, format='wav')
        os.remove(raw_path)
    except Exception as e:
        return jsonify({'error': f'Audio conversion failed: {e}'}), 500

    features, time_data, amp_data = extract_features(wav_path)
    os.remove(wav_path)

    if features is None:
        return jsonify({'error': 'Feature extraction failed'}), 500

    features_scaled = scaler.transform(features.reshape(1, -1))
    probabilities = knn_classifier.predict_proba(features_scaled)[0]

    best_idx = np.argmax(probabilities)
    instrument = class_names[best_idx]
    confidence = probabilities[best_idx] * 100

    probability_table = [
        {'name': class_names[i].title(), 'score': round(p * 100, 2)}
        for i, p in enumerate(probabilities)
    ]
    probability_table.sort(key=lambda x: x['score'], reverse=True)

    result = {
        'instrument': instrument.title(),
        'confidence_score': round(confidence, 2),
        'waveform': {
            'time': time_data,
            'amplitude': amp_data
        },
        'feature_vector': features.tolist(),
        'compared_vector': reference_database[instrument]['average_vector'].tolist(),
        'knn_probabilities': probability_table
    }

    print(f"🎵 Prediction: {instrument} ({confidence:.2f}%)")
    return jsonify(result)

# =========================================================
# START SERVER
# =========================================================
if __name__ == '__main__':
    load_and_train_knn()
    app.run(debug=True, port=5000)
