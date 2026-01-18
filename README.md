# 🎵 Instrument Recognizer (KNN-BASED)

An explainable audio intelligence platform that identifies musical instruments using spectral feature analysis and K-Nearest Neighbors (KNN) classification.

![Banner](https://img.shields.io/badge/Status-Active-brightgreen)
![Tech](https://img.shields.io/badge/Stack-Flask%20%7C%20React%20%7C%20Librosa-blue)

## 🚀 Overview
The Instrument Recognizer is a full-stack web application designed to classify musical instruments from audio files (.wav, .mp3). Unlike "black-box" AI models, this project focuses on **explainable AI**, providing users with a visual breakdown of the spectral features (the "fingerprint") that led to a specific classification.

### Key Features
- **Real-time Spectral Analysis:** Extracts 10 unique dimensions of audio, including MFCCs, Chroma, and Spectral Centroid.
- **Explainable Results:**
    - **Waveform Visualization:** View the raw time-domain signal.
    - **Feature Fingerprint Comparison:** A radar chart comparing the input audio's features against the database average for the identified instrument.
    - **Probability Distribution:** See how the KNN model scored other potential instrument matches.
- **Detailed CSV Reports:** Download a full spectral analysis report for offline use.
- **Modern UI:** Built with Material UI and Chart.js for a premium, dark-themed analytical experience.

---

## 🛠️ Technical Stack

### Backend (AI & API)
- **Python / Flask:** Lightweight REST API.
- **Librosa:** Robust audio processing and feature extraction.
- **Scikit-learn:** KNN classifier with cosine similarity metrics for high-accuracy matching.
- **StandardScaler:** Ensures feature normalization for unbiased classification.

### Frontend (UI/UX)
- **React:** Modern component-based architecture.
- **MUI (Material UI):** Professional-grade dark theme and layout.
- **Chart.js:** Dynamic rendering of waveforms, radar charts, and bar graphs.
- **Axios:** Asynchronous communication with the Flask backend.

---

## 🧠 How It Works (KNN Spectral Analysis)
The system uses a **Fingerprinting Approach** to classify audio:
1. **Feature Extraction:** The audio is analyzed to extract a 10-dimensional vector:
    - **MFCC (Mean & Std):** Captures timbre and texture.
    - **Chroma (Mean & Std):** Captures harmonic content.
    - **Spectral Centroid:** Measures "brightness."
    - **Spectral Rolloff:** Captures the shape of the spectral power distribution.
    - **Zero Crossing Rate:** Measure of noisiness.
    - **Spectral Bandwidth:** Measures the "richness" of the sound.
2. **Normalization:** The input vector is scaled using a pre-trained `StandardScaler`.
3. **Classification:** A **K-Nearest Neighbors (K=7)** classifier calculates the **Cosine Distance** between the input vector and thousands of reference fingerprints in the `reference_database`.
4. **Result:** The instrument with the highest aggregate similarity (weighted by distance) is returned.

---

## 📥 Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js & npm

### Backend Setup
1. Navigate to the `backend` folder:
   ```bash
   cd backend
   ```
2. Install dependencies:
   ```bash
   pip install -r ../requirements.txt
   ```
3. Run the Flask server:
   ```bash
   python app.py
   ```
   *The server will start on `http://127.0.0.1:5000`*

### Frontend Setup
1. Navigate to the `frontend` folder:
   ```bash
   cd frontend
   ```
2. Install packages:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm start
   ```
   *The application will open on `http://localhost:3000`*

---

## 📂 Project Structure
```text
.
├── backend/
│   ├── app.py          # Flask Server & Logic
│   ├── reference_database.pkl # Pre-extracted spectral data
│   └── uploads/        # Temporary audio storage
├── frontend/
│   ├── src/
│   │   └── App.js      # Main UI Logic
│   └── package.json    # Frontend dependencies
├── requirements.txt    # Backend dependencies
└── README.md           # You are here!
```

---

## 📜 License
Developed for educational and research purposes in musical information retrieval (MIR).
