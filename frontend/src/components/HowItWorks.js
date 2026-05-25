import React from "react";
import { Box, Typography, Grid, Card, Divider } from "@mui/material";

const steps = [
  {
    step: "01",
    title: "Upload Audio",
    desc: "Select or drag-and-drop an audio file in any supported format. The system accepts WAV, MP3, OGG, FLAC, and M4A files up to 30 seconds in duration. Files are automatically converted to mono 22050 Hz WAV for consistent processing.",
  },
  {
    step: "02",
    title: "Feature Extraction",
    desc: "The audio signal is analyzed using Librosa to compute a 26-dimensional spectral fingerprint. This includes 13 individual MFCC means, temporal delta, chroma statistics, spectral centroid, rolloff, zero crossing rate, bandwidth, flatness, and RMS energy variation.",
  },
  {
    step: "03",
    title: "Normalization and Classification",
    desc: "The feature vector is normalized using StandardScaler (fitted on the training set) to prevent any single dimension from dominating. The KNN classifier then computes cosine distance against 3724 reference fingerprints and returns a distance-weighted vote across K=3 nearest neighbors.",
  },
  {
    step: "04",
    title: "Explainable Results",
    desc: "The response includes the predicted instrument with confidence score, a probability distribution across all 11 classes, the raw waveform for visualization, and a radar chart comparing your audio's fingerprint against the database average for the predicted class.",
  },
];

export default function HowItWorks() {
  return (
    <Box mb={8} sx={{ position: "relative", zIndex: 1 }}>
      <Typography
        variant="h6"
        sx={{ color: "#10b981", mb: 1, fontSize: "0.75rem", letterSpacing: "0.08em" }}
      >
        PIPELINE
      </Typography>
      <Typography variant="h4" sx={{ color: "#f5f5f5", mb: 1 }}>
        How it works
      </Typography>
      <Typography variant="body1" sx={{ mb: 5, maxWidth: 550 }}>
        The classification pipeline processes audio through four stages,
        from raw signal to explainable prediction.
      </Typography>

      <Grid container spacing={3}>
        {steps.map((s) => (
          <Grid item xs={12} sm={6} key={s.step}>
            <Card
              sx={{
                p: 4,
                height: "100%",
                position: "relative",
                transition: "border-color 0.2s",
                "&:hover": { borderColor: "rgba(255,255,255,0.12)" },
              }}
            >
              {/* Step number watermark */}
              <Typography
                sx={{
                  position: "absolute",
                  top: 16,
                  right: 20,
                  fontSize: "3rem",
                  fontWeight: 800,
                  color: "rgba(255,255,255,0.03)",
                  lineHeight: 1,
                }}
              >
                {s.step}
              </Typography>

              <Box
                sx={{
                  display: "inline-block",
                  px: 1.5,
                  py: 0.3,
                  borderRadius: 1,
                  bgcolor: "rgba(16,185,129,0.08)",
                  border: "1px solid rgba(16,185,129,0.15)",
                  mb: 2,
                }}
              >
                <Typography variant="caption" sx={{ color: "#10b981", fontWeight: 600 }}>
                  Step {s.step}
                </Typography>
              </Box>

              <Typography
                variant="body1"
                sx={{ color: "#f5f5f5", fontWeight: 600, mb: 1.5, fontSize: "1.05rem" }}
              >
                {s.title}
              </Typography>

              <Typography variant="body2" sx={{ lineHeight: 1.7, color: "#a3a3a3" }}>
                {s.desc}
              </Typography>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Divider sx={{ mt: 6 }} />
    </Box>
  );
}
