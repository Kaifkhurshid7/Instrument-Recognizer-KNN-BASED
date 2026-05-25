import React from "react";
import { Box, Typography, Grid, Card, Divider } from "@mui/material";

const capabilities = [
  {
    title: "Spectral Feature Extraction",
    desc: "Extracts a 26-dimensional spectral fingerprint from audio using Librosa. Includes 13 individual MFCC coefficients, MFCC delta for temporal dynamics, chroma for harmonic content, spectral centroid and rolloff for brightness, zero crossing rate for percussiveness, bandwidth for richness, flatness for tonal vs noisy distinction, and RMS energy for dynamic range.",
    details: [
      "Sample rate: 22050 Hz",
      "Max duration: 30 seconds",
      "Hop length: 1024 (memory-optimized)",
      "MFCC coefficients: 13",
    ],
  },
  {
    title: "KNN Classification with Cosine Distance",
    desc: "Uses a K-Nearest Neighbors classifier with cosine similarity metric and distance-weighted voting. Cosine distance measures directional similarity between spectral fingerprints regardless of magnitude, making it ideal for audio features where absolute values vary but relative proportions are consistent within instrument classes.",
    details: [
      "K = 3 (optimal via grid search)",
      "Metric: Cosine distance",
      "Weights: Distance-weighted",
      "Normalization: StandardScaler",
    ],
  },
  {
    title: "Explainable Classification Results",
    desc: "Every prediction comes with full transparency. The radar chart overlays your audio's spectral profile against the database average for the predicted instrument, showing exactly which features matched and which diverged. The probability distribution reveals how the model scored all 11 classes, not just the top prediction.",
    details: [
      "Radar chart: input vs database average",
      "Bar chart: all class probabilities",
      "Waveform: time-domain signal",
      "Feature table: all 26 dimensions",
    ],
  },
  {
    title: "Audio Format Support",
    desc: "Accepts multiple audio formats through automatic conversion. Uploaded files are converted to mono WAV at 22050 Hz before analysis, ensuring consistent feature extraction regardless of the original format, channel count, or sample rate.",
    details: [
      "WAV (uncompressed)",
      "MP3 (compressed)",
      "OGG, FLAC, M4A",
      "Auto mono conversion",
    ],
  },
  {
    title: "Downloadable Analysis Reports",
    desc: "Generate a CSV report containing the full analysis: predicted instrument, confidence score, probability distribution across all 11 classes, and all 26 feature values compared against the database average. Useful for offline review, documentation, or further analysis in Excel or Python.",
    details: [
      "Instrument prediction + confidence",
      "All 11 class probabilities",
      "26 feature values (input vs DB)",
      "Timestamped CSV download",
    ],
  },
  {
    title: "REST API for Integration",
    desc: "The Flask backend exposes a clean REST API that can be consumed by any frontend or integrated into other applications. Upload audio via multipart form data and receive structured JSON with all classification data, waveform arrays, feature vectors, and probability tables.",
    details: [
      "POST /analyze — classify audio",
      "GET /health — server status",
      "JSON response with full data",
      "CORS enabled for any origin",
    ],
  },
];

export default function FeaturesSection() {
  return (
    <Box mb={8} sx={{ position: "relative", zIndex: 1 }}>
      <Typography
        variant="h6"
        sx={{ color: "#3b82f6", mb: 1, fontSize: "0.75rem", letterSpacing: "0.08em" }}
      >
        CAPABILITIES
      </Typography>
      <Typography variant="h4" sx={{ color: "#f5f5f5", mb: 1 }}>
        What this platform can do
      </Typography>
      <Typography variant="body1" sx={{ mb: 5, maxWidth: 600 }}>
        A complete audio intelligence toolkit for musical instrument recognition.
        Each capability is designed to provide transparency and usability for
        researchers, students, and developers working with audio classification.
      </Typography>

      <Grid container spacing={3}>
        {capabilities.map((cap) => (
          <Grid item xs={12} md={6} key={cap.title}>
            <Card
              sx={{
                p: 4,
                height: "100%",
                display: "flex",
                flexDirection: "column",
                transition: "border-color 0.2s",
                "&:hover": { borderColor: "rgba(255,255,255,0.12)" },
              }}
            >
              <Typography
                variant="body1"
                sx={{ color: "#f5f5f5", fontWeight: 600, mb: 1.5, fontSize: "1.05rem" }}
              >
                {cap.title}
              </Typography>

              <Typography variant="body2" sx={{ lineHeight: 1.7, mb: 2, color: "#a3a3a3" }}>
                {cap.desc}
              </Typography>

              <Divider sx={{ mb: 2 }} />

              <Box component="ul" sx={{ m: 0, pl: 2.5 }}>
                {cap.details.map((d) => (
                  <Box
                    component="li"
                    key={d}
                    sx={{ color: "#737373", fontSize: "0.8rem", mb: 0.5, lineHeight: 1.6 }}
                  >
                    <Typography variant="caption" sx={{ color: "#737373", fontSize: "0.8rem" }}>
                      {d}
                    </Typography>
                  </Box>
                ))}
              </Box>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
}
