import React from "react";
import { Box, Typography, Card, Divider } from "@mui/material";

export default function ApiDocs() {
  return (
    <Box mb={8} sx={{ position: "relative", zIndex: 1 }}>
      <Typography
        variant="h6"
        sx={{ color: "#f59e0b", mb: 1, fontSize: "0.75rem", letterSpacing: "0.08em" }}
      >
        API REFERENCE
      </Typography>
      <Typography variant="h4" sx={{ color: "#f5f5f5", mb: 1 }}>
        REST API Documentation
      </Typography>
      <Typography variant="body1" sx={{ mb: 5, maxWidth: 550 }}>
        Integrate instrument recognition into your own applications using the Flask REST API.
      </Typography>

      {/* POST /analyze */}
      <Card sx={{ p: 4, mb: 3 }}>
        <Box display="flex" alignItems="center" gap={1.5} mb={2}>
          <Box
            sx={{
              px: 1.5, py: 0.3, borderRadius: 1,
              bgcolor: "rgba(16,185,129,0.1)",
              border: "1px solid rgba(16,185,129,0.2)",
            }}
          >
            <Typography variant="caption" sx={{ color: "#10b981", fontWeight: 700 }}>
              POST
            </Typography>
          </Box>
          <Typography variant="body1" sx={{ color: "#f5f5f5", fontWeight: 600, fontFamily: "monospace" }}>
            /analyze
          </Typography>
        </Box>

        <Typography variant="body2" sx={{ color: "#a3a3a3", mb: 3, lineHeight: 1.7 }}>
          Upload an audio file for instrument classification. Accepts multipart/form-data
          with a field named <code style={{ color: "#3b82f6" }}>audioFile</code>. Returns
          the predicted instrument, confidence score, probability distribution, waveform data,
          and feature vectors for visualization.
        </Typography>

        <Divider sx={{ mb: 2 }} />

        <Typography variant="caption" sx={{ color: "#525252", textTransform: "uppercase", letterSpacing: "0.05em" }}>
          Request
        </Typography>
        <Box
          sx={{
            mt: 1, mb: 3, p: 2, borderRadius: 1,
            bgcolor: "#0a0a0a",
            border: "1px solid rgba(255,255,255,0.06)",
            fontFamily: "monospace",
            fontSize: "0.8rem",
            color: "#a3a3a3",
            overflowX: "auto",
          }}
        >
          <Box>Content-Type: multipart/form-data</Box>
          <Box sx={{ mt: 0.5 }}>Body: audioFile (File) — WAV, MP3, OGG, FLAC, M4A</Box>
        </Box>

        <Typography variant="caption" sx={{ color: "#525252", textTransform: "uppercase", letterSpacing: "0.05em" }}>
          Response (200 OK)
        </Typography>
        <Box
          sx={{
            mt: 1, p: 2, borderRadius: 1,
            bgcolor: "#0a0a0a",
            border: "1px solid rgba(255,255,255,0.06)",
            fontFamily: "monospace",
            fontSize: "0.75rem",
            color: "#a3a3a3",
            overflowX: "auto",
            whiteSpace: "pre",
            lineHeight: 1.6,
          }}
        >
{`{
  "instrument": "Piano",
  "confidence_score": 87.34,
  "waveform": {
    "time": [0.0, 0.003, ...],
    "amplitude": [0.012, -0.008, ...]
  },
  "feature_vector": [26 float values],
  "compared_vector": [26 float values],
  "knn_probabilities": [
    { "name": "Piano", "score": 87.34 },
    { "name": "Organ", "score": 5.21 },
    ...
  ]
}`}
        </Box>
      </Card>

      {/* GET /health */}
      <Card sx={{ p: 4 }}>
        <Box display="flex" alignItems="center" gap={1.5} mb={2}>
          <Box
            sx={{
              px: 1.5, py: 0.3, borderRadius: 1,
              bgcolor: "rgba(59,130,246,0.1)",
              border: "1px solid rgba(59,130,246,0.2)",
            }}
          >
            <Typography variant="caption" sx={{ color: "#3b82f6", fontWeight: 700 }}>
              GET
            </Typography>
          </Box>
          <Typography variant="body1" sx={{ color: "#f5f5f5", fontWeight: 600, fontFamily: "monospace" }}>
            /health
          </Typography>
        </Box>

        <Typography variant="body2" sx={{ color: "#a3a3a3", mb: 3, lineHeight: 1.7 }}>
          Health check endpoint. Returns server status and whether the KNN model
          has been loaded and is ready for inference.
        </Typography>

        <Divider sx={{ mb: 2 }} />

        <Typography variant="caption" sx={{ color: "#525252", textTransform: "uppercase", letterSpacing: "0.05em" }}>
          Response (200 OK)
        </Typography>
        <Box
          sx={{
            mt: 1, p: 2, borderRadius: 1,
            bgcolor: "#0a0a0a",
            border: "1px solid rgba(255,255,255,0.06)",
            fontFamily: "monospace",
            fontSize: "0.75rem",
            color: "#a3a3a3",
            overflowX: "auto",
            whiteSpace: "pre",
            lineHeight: 1.6,
          }}
        >
{`{
  "status": "ok",
  "model_ready": true
}`}
        </Box>
      </Card>
    </Box>
  );
}
