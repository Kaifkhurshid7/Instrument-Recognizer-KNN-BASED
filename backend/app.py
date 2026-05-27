"""
Flask API server for instrument recognition.

Endpoints:
    POST /analyze  — Upload audio, get classification + explainability data
    GET  /health   — Server status check
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydub import AudioSegment

from config import UPLOAD_FOLDER, SAMPLE_RATE, PORT, DEBUG
from classifier import InstrumentClassifier
from feature_extraction import extract_features

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)

classifier = InstrumentClassifier()


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "model_ready": classifier.is_ready})


@app.route("/analyze", methods=["POST"])
def analyze_audio():
    """Accept an audio file upload, extract features, classify, and return results."""
    if not classifier.is_ready:
        return jsonify({"error": "Model not initialized."}), 503

    if "audioFile" not in request.files:
        return jsonify({"error": "No audio file provided."}), 400

    file = request.files["audioFile"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    # Convert uploaded file to mono WAV at target sample rate
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
        if os.path.exists(raw_path):
            os.remove(raw_path)

    # Extract spectral features
    extraction = extract_features(wav_path, include_waveform=True)
    _cleanup(wav_path)

    if extraction["features"] is None:
        return jsonify({"error": "Feature extraction failed."}), 500

    # Run KNN prediction
    prediction = classifier.predict(extraction["features"])

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


def _cleanup(*paths):
    """Remove temporary files silently."""
    for path in paths:
        if path and os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    classifier.load_and_train()
    app.run(debug=DEBUG, port=PORT)
else:
    classifier.load_and_train()
