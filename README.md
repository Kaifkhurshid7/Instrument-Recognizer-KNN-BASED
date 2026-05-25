# Instrument Recognizer

An explainable audio intelligence platform that identifies musical instruments from audio files using spectral feature analysis and K-Nearest Neighbors (KNN) classification.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![React](https://img.shields.io/badge/React-19-61dafb)
![Flask](https://img.shields.io/badge/Flask-3.1-green)
![License](https://img.shields.io/badge/License-Educational-orange)

---

## Overview

Unlike black-box deep learning models, this project focuses on **explainable AI**. Users receive a full visual breakdown of the spectral features (the "fingerprint") that led to a classification, making the decision process transparent and interpretable.

### Key Features

- **26-Dimensional Spectral Analysis** — Extracts 13 MFCCs, Delta, Chroma, Spectral Centroid, Rolloff, ZCR, Bandwidth, Flatness, and RMS
- **Explainable Results** — Radar chart comparing input fingerprint vs. database average
- **Probability Distribution** — See how the model scored all instrument classes
- **Waveform Visualization** — Raw time-domain signal display
- **CSV Reports** — Download detailed analysis for offline review
- **Drag & Drop Upload** — Modern, responsive dark-themed UI

### Supported Instruments (11 Classes)

Acoustic Guitar · Cello · Clarinet · Electric Guitar · Flute · Human Voice · Organ · Piano · Saxophone · Trumpet · Violin

---

## Model Performance

Evaluated using **5-Fold Stratified Cross-Validation** on 3724 samples across 11 classes.

### Final Model (K=3, Cosine Distance, Distance-Weighted)

| Metric | Score |
|--------|-------|
| **Accuracy** | 61.90% (±0.92%) |
| **Precision** | 62.05% |
| **Recall** | 61.90% |
| **F1 Score** | 61.73% |

### K-Value Tuning Results

| K | Accuracy |
|---|----------|
| 3 | **61.90%** |
| 5 | 60.69% |
| 7 | 59.18% |
| 9 | 57.04% |
| 11 | 56.34% |
| 15 | 55.10% |
| 21 | 53.36% |

K=3 was selected as the optimal value — smaller K captures local structure better for spectral fingerprints where similar instruments cluster tightly.

### Reproduce Results

```bash
cd backend
python evaluate.py
```

---

## Architecture

```
┌─────────────────┐         POST /analyze         ┌──────────────────────┐
│                 │  ──────────────────────────▶   │                      │
│   React UI      │                               │   Flask API Server   │
│   (Port 3000)   │  ◀──────────────────────────  │   (Port 5000)        │
│                 │         JSON Response          │                      │
└─────────────────┘                               └──────────────────────┘
                                                           │
                                                           ▼
                                                  ┌──────────────────────┐
                                                  │  KNN Classifier      │
                                                  │  (Cosine Distance)   │
                                                  │  K=3, Weighted       │
                                                  └──────────────────────┘
                                                           │
                                                           ▼
                                                  ┌──────────────────────┐
                                                  │  Reference Database  │
                                                  │  (3724 fingerprints) │
                                                  └──────────────────────┘
```

### How Classification Works

1. **Feature Extraction** — Audio is loaded at 22050 Hz and analyzed to produce a 26-dimensional spectral vector
2. **Normalization** — The vector is scaled using a pre-trained `StandardScaler` to prevent feature dominance
3. **KNN Classification** — Cosine distance is computed against 3724 reference fingerprints (K=3, distance-weighted)
4. **Result** — The instrument with highest aggregate similarity is returned with full probability distribution

---

## Project Structure

```
├── backend/
│   ├── app.py                  # Flask server & API routes
│   ├── classifier.py           # KNN model training & prediction
│   ├── feature_extraction.py   # 26-D spectral feature extraction
│   ├── config.py               # Centralized configuration
│   ├── build_database.py       # Script to build reference_database.pkl
│   ├── evaluate.py             # Cross-validation & metrics
│   └── IRMAS-TrainingData/     # Training audio dataset (not in git)
│
├── frontend/
│   ├── src/
│   │   ├── App.js              # Root component
│   │   ├── config/
│   │   │   ├── theme.js        # MUI dark theme
│   │   │   └── constants.js    # Feature labels, colors, API URL
│   │   ├── services/
│   │   │   └── api.js          # Backend communication layer
│   │   ├── components/
│   │   │   ├── Header.js       # App title & navigation
│   │   │   ├── FileUpload.js   # Drag-and-drop upload + settings panel
│   │   │   ├── ResultCard.js   # Instrument + confidence display
│   │   │   ├── FeatureTable.js # Numerical feature comparison
│   │   │   └── charts/
│   │   │       ├── WaveformChart.js    # Time-domain signal
│   │   │       ├── ProbabilityChart.js # Class probabilities
│   │   │       └── RadarChart.js       # Feature fingerprint overlay
│   │   └── utils/
│   │       └── reportGenerator.js  # CSV export logic
│   └── .env                    # Environment config
│
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Setup & Installation

### Prerequisites

- Python 3.8+ (tested with 3.10)
- Node.js 18+ & npm
- FFmpeg (required by pydub for audio conversion)

### Backend

```bash
cd backend

# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r ../requirements.txt

# Build the reference database (first time only, ~2-3 min)
python build_database.py

# Start the API server
python app.py
```

The server starts at `http://127.0.0.1:5000`

### Frontend

```bash
cd frontend

# Install packages
npm install

# Start development server
npm start
```

The app opens at `http://localhost:3000`

---

## API Reference

### `POST /analyze`

Upload an audio file for classification.

**Request:** `multipart/form-data` with field `audioFile`

**Response:**
```json
{
  "instrument": "Piano",
  "confidence_score": 87.34,
  "waveform": { "time": [...], "amplitude": [...] },
  "feature_vector": [26 floats],
  "compared_vector": [26 floats],
  "knn_probabilities": [{ "name": "Piano", "score": 87.34 }, ...]
}
```

### `GET /health`

Server health check.

**Response:** `{ "status": "ok", "model_ready": true }`

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `5000` | Backend server port |
| `FLASK_DEBUG` | `true` | Enable Flask debug mode |
| `REACT_APP_API_URL` | `http://127.0.0.1:5000` | Backend URL for frontend |

---

## Technical Decisions

- **KNN over Deep Learning** — Chosen for explainability. Users can see exactly which features drove the classification.
- **Cosine Distance** — Works better than Euclidean for spectral fingerprints because it measures directional similarity regardless of magnitude.
- **K=3** — Optimal value found via grid search. Smaller K captures local structure better for tightly-clustered spectral fingerprints.
- **26-D Vector** — Individual MFCC coefficients (instead of just mean/std) preserve per-band timbre information critical for distinguishing similar instruments.
- **StandardScaler** — Prevents high-magnitude features (spectral centroid ~2000 Hz) from dominating low-magnitude ones (ZCR ~0.05).

---

## Future Improvements

- [ ] Add audio recording directly in browser
- [ ] Support multi-instrument detection in polyphonic audio
- [ ] Add confusion matrix visualization per class
- [ ] Implement segment-based voting (split audio into chunks, majority vote)
- [ ] Deploy with Docker (backend + frontend in one compose file)
- [ ] Add audio augmentation for improved generalization

---

## License

Developed for educational and research purposes in Musical Information Retrieval (MIR).
Dataset: [IRMAS](https://www.upf.edu/web/mtg/irmas) (Instrument Recognition in Musical Audio Signals).
