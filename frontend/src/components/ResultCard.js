/**
 * Result Card Component
 * ---------------------
 * Displays the primary classification result: instrument name
 * and confidence score with visual emphasis.
 */

import React from "react";
import { Card, Typography, Box, LinearProgress } from "@mui/material";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";

export default function ResultCard({ instrument, confidence }) {
  // Color the confidence bar based on score
  const getConfidenceColor = (score) => {
    if (score >= 80) return "primary";
    if (score >= 50) return "warning";
    return "error";
  };

  return (
    <Card sx={{ p: 4, mb: 4, textAlign: "center" }}>
      <Box display="flex" alignItems="center" justifyContent="center" gap={1} mb={1}>
        <CheckCircleIcon sx={{ color: "primary.main" }} />
        <Typography variant="h6" color="text.secondary">
          Identified Instrument
        </Typography>
      </Box>

      <Typography
        variant="h3"
        color="primary.main"
        sx={{ my: 2, fontWeight: 700 }}
      >
        {instrument}
      </Typography>

      <Box sx={{ maxWidth: 300, mx: "auto", mt: 2 }}>
        <Box display="flex" justifyContent="space-between" mb={0.5}>
          <Typography variant="caption" color="text.secondary">
            Confidence
          </Typography>
          <Typography variant="caption" fontWeight={600}>
            {confidence}%
          </Typography>
        </Box>
        <LinearProgress
          variant="determinate"
          value={confidence}
          color={getConfidenceColor(confidence)}
          sx={{ height: 8, borderRadius: 4 }}
        />
      </Box>
    </Card>
  );
}
