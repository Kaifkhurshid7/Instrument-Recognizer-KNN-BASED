/**
 * Header Component
 * ----------------
 * Application title and subtitle with branding.
 */

import React from "react";
import { Box, Typography, Chip } from "@mui/material";
import GraphicEqIcon from "@mui/icons-material/GraphicEq";

export default function Header() {
  return (
    <Box textAlign="center" mb={5}>
      <Box display="flex" justifyContent="center" alignItems="center" gap={1.5} mb={1}>
        <GraphicEqIcon sx={{ fontSize: 40, color: "primary.main" }} />
        <Typography variant="h3" component="h1">
          Instrument Recognizer
        </Typography>
      </Box>
      <Typography variant="body1" color="text.secondary" mb={2}>
        Explainable audio intelligence powered by spectral fingerprint analysis
      </Typography>
      <Box display="flex" justifyContent="center" gap={1} flexWrap="wrap">
        <Chip label="KNN Classifier" size="small" variant="outlined" />
        <Chip label="10-D Feature Vector" size="small" variant="outlined" />
        <Chip label="Cosine Similarity" size="small" variant="outlined" />
      </Box>
    </Box>
  );
}
