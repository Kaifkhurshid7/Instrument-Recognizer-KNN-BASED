/**
 * Header Component
 * ----------------
 * Clean documentation-style header inspired by Music.AI.
 * Large title, subtitle description, and navigation-like layout.
 */

import React from "react";
import { Box, Typography, Button, Chip, Divider } from "@mui/material";
import ArrowForwardIcon from "@mui/icons-material/ArrowForward";

export default function Header() {
  return (
    <Box mb={6}>
      {/* Top bar */}
      <Box
        display="flex"
        alignItems="center"
        justifyContent="space-between"
        mb={4}
        pb={2}
        sx={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}
      >
        <Box display="flex" alignItems="center" gap={1.5}>
          <Box
            sx={{
              width: 8,
              height: 8,
              borderRadius: "50%",
              bgcolor: "#10b981",
            }}
          />
          <Typography
            variant="body2"
            sx={{ color: "#a3a3a3", fontWeight: 500 }}
          >
            Classification
          </Typography>
          <Typography variant="body2" sx={{ color: "#525252" }}>
            /
          </Typography>
          <Typography
            variant="body2"
            sx={{ color: "#f5f5f5", fontWeight: 600 }}
          >
            Instruments Detection
          </Typography>
        </Box>
      </Box>

      {/* Main title */}
      <Typography variant="h2" sx={{ color: "#f5f5f5", mb: 2 }}>
        Instruments Detection
      </Typography>

      {/* Description */}
      <Typography
        variant="body1"
        sx={{ maxWidth: 600, mb: 4, lineHeight: 1.8 }}
      >
        Identify instruments present in music. Upload an audio file and get
        real-time classification with explainable spectral analysis using
        KNN and cosine similarity.
      </Typography>

      {/* CTA buttons */}
      <Box display="flex" gap={2} alignItems="center" mb={4}>
        <Button
          variant="contained"
          size="large"
          sx={{ px: 4 }}
        >
          Upload & Analyze
        </Button>
        <Button
          variant="text"
          endIcon={<ArrowForwardIcon />}
          sx={{ color: "#a3a3a3", "&:hover": { color: "#f5f5f5" } }}
        >
          View Documentation
        </Button>
      </Box>

      {/* Tags */}
      <Box display="flex" gap={1} flexWrap="wrap">
        <Chip label="KNN (K=7)" size="small" variant="outlined" />
        <Chip label="26-D Feature Vector" size="small" variant="outlined" />
        <Chip label="Cosine Distance" size="small" variant="outlined" />
        <Chip label="11 Instruments" size="small" variant="outlined" />
      </Box>

      <Divider sx={{ mt: 4 }} />
    </Box>
  );
}
