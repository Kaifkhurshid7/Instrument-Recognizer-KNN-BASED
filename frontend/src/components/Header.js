import React from "react";
import { Box, Typography, Chip } from "@mui/material";
import GraphicEqIcon from "@mui/icons-material/GraphicEq";

export default function Header() {
  return (
    <Box mb={5} textAlign="center">
      {/* Logo / brand */}
      <Box display="flex" justifyContent="center" alignItems="center" gap={1.5} mb={2}>
        <GraphicEqIcon sx={{ fontSize: 36, color: "#3b82f6" }} />
        <Typography
          variant="h4"
          sx={{ color: "#f5f5f5", fontWeight: 700, letterSpacing: "-0.02em" }}
        >
          Instrument Recognizer
        </Typography>
      </Box>

      <Typography variant="body1" sx={{ color: "#737373", maxWidth: 520, mx: "auto", mb: 3 }}>
        Upload an audio file to identify instruments using spectral fingerprint
        analysis and KNN classification
      </Typography>

      {/* Tags */}
      <Box display="flex" justifyContent="center" gap={1} flexWrap="wrap">
        <Chip label="KNN (K=7)" size="small" variant="outlined" />
        <Chip label="26-D Features" size="small" variant="outlined" />
        <Chip label="Cosine Distance" size="small" variant="outlined" />
        <Chip label="11 Classes" size="small" variant="outlined" />
      </Box>
    </Box>
  );
}
