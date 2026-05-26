import React from "react";
import { Box, Typography, Card, Grid, Divider } from "@mui/material";

export default function ApiDocs() {
  return (
    <Box mb={8} sx={{ position: "relative", zIndex: 1 }}>
      <Typography
        variant="h6"
        sx={{ color: "#8b5cf6", mb: 1, fontSize: "0.75rem", letterSpacing: "0.08em" }}
      >
        API DOCUMENTATION
      </Typography>
      <Typography variant="h4" sx={{ color: "#f5f5f5", mb: 1 }}>
        REST API Reference
      </Typography>
      <Typography variant="body1" sx={{ mb: 5, maxWidth: 600 }}>
        Integrate instrument recognition into your own applications using the Flask REST API.
        All endpoints return JSON responses.
      </Typography>

      <Grid container spacing={3}>
        {/* POST /analyze */}
        <Grid item xs={12} md={6}>
          <Card sx={{ p: 4, height: "100%" }}>
            <Box display="flex" alignItems="center" gap={1.5} mb={2}>
              <Box
                sx={{
                  px: 1.2, py: 0.3, borderRadius: 1,
                  bgcolor: "rgba(16,185,129,0.1)",
                  border: "1px solid rgba(16,185,129,0.2)",
                }}
              >
                <Typography variant="caption" sx={{ color: "#10b981", fontWeight: 700, fontFamily: "monospace" }}>
                  POST
                </Typography>
              </Box>
              <Typography variant="body1" sx={{ color: "#f5f5f5", fontWeight: 600, fontFamily: "monospace" }}>
                /analyze
              </Typography>
            </Box>

            <Typography variant="body2" sx={{ color: "#a3a3a3", mb: 2, lineHeight: 1.7 }}>
              Upload an audio file for instrument classification. Returns the predicted instrument,
              confidence score, probability distribution across all 11 classes, waveform data for
              visualization, and the 26-dimensional feature vector compared against the database average.
            </Typography>

            <Divider sx={{ mb: 2 }} />

            <Typography variant="caption" sx={{ color: "#525252", textTransform: "uppercase", letterSpacing: "0.05em" }}>
              Request
            </Typography>
            <Box
              sx={{
                mt: 1, mb: 2, p: 2, borderRadius: 1,
                bgcolor: "#0a0a0a",
                border: "1px solid rgba(255,255,255,0.06)",
                fontFamily: "monospace",
                fontSize: "0.8rem",
                color: "#a3a3a3",
                lineHeight: 1.8,
              }}
            >
              Content-Type: multipart/form-data<br />
              Field: audioFile (File)
            </Box>

            <Typography variant="caption" sx={{ color: "#525252", textTransform: "uppercase", letterSpacing: "0.05em" }}>
              Response (200)
            </Typography>
            <Box
              sx={{
                mt: 1, p: 2, borderRadius: 1,
                bgcolor: "#0a0a0a",
                border: "1px solid rgba(255,255,255,0.06)",
                fontFamily: "monospace",
                fontSize: "0.75rem",
                color: "#a3a3a3",
                lineHeight: 1.8,
                whiteSpace: "pre-wrap",
              }}
            >
{`{
  "instrument": "Piano",
  "confidence_score": 87.34,
  "knn_probabilities": [
    {"name": "Piano", "score": 87.34},
    {"name": "Organ", "score": 5.21},
    ...
  ],
  "feature_vector": [26 floats],
  "compared_vector": [26 floats],
  "waveform": {
    "time": [1000 floats],
    "amplitude": [1000 floats]
  }
}`}
            </Box>
          </Card>
        </Grid>

        {/* GET /health */}
        <Grid item xs={12} md={6}>
          <Card sx={{ p: 4, height: "100%" }}>
            <Box display="flex" alignItems="center" gap={1.5} mb={2}>
              <Box
                sx={{
                  px: 1.2, py: 0.3, borderRadius: 1,
                  bgcolor: "rgba(59,130,246,0.1)",
                  border: "1px solid rgba(59,130,246,0.2)",
                }}
              >
                <Typography variant="caption" sx={{ color: "#3b82f6", fontWeight: 700, fontFamily: "monospace" }}>
                  GET
                </Typography>
              </Box>
              <Typography variant="body1" sx={{ color: "#f5f5f5", fontWeight: 600, fontFamily: "monospace" }}>
                /health
              </Typography>
            </Box>

            <Typography variant="body2" sx={{ color: "#a3a3a3", mb: 2, lineHeight: 1.7 }}>
              Check if the backend server is running and the KNN model has been loaded and trained.
              Use this endpoint for connectivity checks before sending analysis requests.
            </Typography>

            <Divider sx={{ mb: 2 }} />

            <Typography variant="caption" sx={{ color: "#525252", textTransform: "uppercase", letterSpacing: "0.05em" }}>
              Response (200)
            </Typography>
            <Box
              sx={{
                mt: 1, mb: 3, p: 2, borderRadius: 1,
                bgcolor: "#0a0a0a",
                border: "1px solid rgba(255,255,255,0.06)",
                fontFamily: "monospace",
                fontSize: "0.75rem",
                color: "#a3a3a3",
                lineHeight: 1.8,
                whiteSpace: "pre-wrap",
              }}
            >
{`{
  "status": "ok",
  "model_ready": true
}`}
            </Box>

            <Divider sx={{ mb: 2 }} />

            <Typography variant="caption" sx={{ color: "#525252", textTransform: "uppercase", letterSpacing: "0.05em" }}>
              Error Responses
            </Typography>
            <Box sx={{ mt: 1.5 }}>
              {[
                { code: "400", msg: "No audio file provided / Empty filename" },
                { code: "500", msg: "Audio conversion failed / Feature extraction failed" },
                { code: "503", msg: "Model not initialized (server starting up)" },
              ].map((err) => (
                <Box key={err.code} display="flex" alignItems="center" gap={1.5} mb={1}>
                  <Box
                    sx={{
                      px: 1, py: 0.2, borderRadius: 0.5,
                      bgcolor: "rgba(239,68,68,0.08)",
                      border: "1px solid rgba(239,68,68,0.15)",
                      minWidth: 36,
                      textAlign: "center",
                    }}
                  >
                    <Typography variant="caption" sx={{ color: "#ef4444", fontFamily: "monospace", fontWeight: 600 }}>
                      {err.code}
                    </Typography>
                  </Box>
                  <Typography variant="body2" sx={{ color: "#737373", fontSize: "0.8rem" }}>
                    {err.msg}
                  </Typography>
                </Box>
              ))}
            </Box>

            <Divider sx={{ my: 2 }} />

            <Typography variant="caption" sx={{ color: "#525252", textTransform: "uppercase", letterSpacing: "0.05em" }}>
              Configuration
            </Typography>
            <Box sx={{ mt: 1.5 }}>
              {[
                { key: "PORT", val: "5000", desc: "Server port" },
                { key: "FLASK_DEBUG", val: "true", desc: "Debug mode" },
                { key: "REACT_APP_API_URL", val: "http://127.0.0.1:5000", desc: "Frontend API target" },
              ].map((env) => (
                <Box key={env.key} display="flex" alignItems="center" gap={1.5} mb={1}>
                  <Typography variant="caption" sx={{ color: "#3b82f6", fontFamily: "monospace", fontWeight: 600, minWidth: 160 }}>
                    {env.key}
                  </Typography>
                  <Typography variant="caption" sx={{ color: "#737373" }}>
                    {env.desc} (default: {env.val})
                  </Typography>
                </Box>
              ))}
            </Box>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}
