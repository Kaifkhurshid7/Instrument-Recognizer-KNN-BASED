"""
Instrument Recognizer - Flask API Server
=========================================
REST API that accepts audio file uploads, extracts spectral features,
and returns KNN-based instrument classification with explainability data.

Endpoints:
    POST /analyze  - Upload audio file, receive classification results
    GET  /health   - Server health check
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydub import AudioSegment

from config import UPLOAD_FOLDER, SAMPLE_RATE, PORT, DEBUG
from classifier import InstrumentClassifier
from feature_extraction import extract_features

# =============================================================
# Application Factory
# =============================================================

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)

# Initialize the classifier (trained once at startup)
classifier = InstrumentClassifier()


# =============================================================
# Routes
# =============================================================


@app.route("/health", methods=["GET"])
def health_check():
    """Simple health check for monitoring and frontend connectivity tests."""
    return jsonify({"status": "ok", "model_ready": classifier.is_ready})


@app.route("/analyze", methods=["POST"])
def analyze_audio():
    """
    Analyze an uploaded audio file and return instrument classification.

    Expects: multipart/form-data with field 'audioFile'
    Returns: JSON with instrument prediction, confidence, waveform,
             feature vectors, and probability distribution.
    """
    # --- Validation ---
    if not classifier.is_ready:
        return jsonify({"error": "Model not initialized. Server is starting up."}), 503

    if "audioFile" not in request.files:
        return jsonify({"error": "No audio file provided."}), 400

    file = request.files["audioFile"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    # --- Audio Conversion ---
    # Convert any supported format to mono WAV at target sample rate
    raw_path = os.path.join(UPLOAD_FOLDER, file.filename)
    wav_path = os.path.join(UPLOAD_FOLDER, "temp_analysis.wav")

    file.save(raw_path)

    try:
        audio = AudioSegment.from_file(raw_path)
        audio = audio.set_channels(1).set_frame_rate(SAMPLE_RATE)
        audio.export(wav_path, format="wav")
    except Exception as e:
        _cleanup(raw_path, wav_path)
        return jsonify({"error": f"Audio conversion failed: {e}"}), 500
    finally:
        # Always remove the raw upload
        if os.path.exists(raw_path):
            os.remove(raw_path)

    # --- Feature Extraction ---
    extraction = extract_features(wav_path, include_waveform=True)
    _cleanup(wav_path)

    if extraction["features"] is None:
        return jsonify({"error": "Feature extraction failed. File may be corrupted."}), 500

    # --- Classification ---
    prediction = classifier.predict(extraction["features"])

    # --- Build Response ---
    response = {
        "instrument": prediction["instrument"],
        "confidence_score": prediction["confidence"],
        "waveform": {
            "time": extraction.get("time", []),
            "amplitude": extraction.get("amplitude", []),
        },
        "feature_vector": extraction["features"].tolist(),
        "compared_vector": prediction["average_vector"],
        "knn_probabilities": prediction["probabilities"],
    }

    print(f"[Analysis] {prediction['instrument']} ({prediction['confidence']:.1f}%)")
    return jsonify(response)


# =============================================================
# Helpers
# =============================================================


def _cleanup(*paths):
    """Remove temporary files, ignoring missing ones."""
    for path in paths:
        if path and os.path.exists(path):
            os.remove(path)


# =============================================================
# Entry Point
# =============================================================

if __name__ == "__main__":
    classifier.load_and_train()
    app.run(debug=DEBUG, port=PORT)
