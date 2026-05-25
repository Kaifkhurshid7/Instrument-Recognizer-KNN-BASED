/**
 * Result Card Component
 * ---------------------
 * Displays the primary classification result in a clean,
 * documentation-style output section.
 */

import React from "react";
import { Card, Typography, Box, LinearProgress, Grid } from "@mui/material";

export default function ResultCard({ instrument, confidence }) {
  const getConfidenceColor = (score) => {
    if (score >= 80) return "success";
    if (score >= 50) return "warning";
    return "error";
  };

  return (
    <Box mb={4}>
      <Typography variant="h6" sx={{ color: "#10b981", mb: 1.5, fontSize: "0.75rem" }}>
        OUTPUT
      </Typography>

      <Grid container spacing={2}>
        {/* Instrument result */}
        <Grid item xs={12} md={6}>
          <Card sx={{ p: 3, height: "100%" }}>
            <Typography variant="caption" sx={{ color: "#737373" }}>
              Identified Instrument
            </Typography>
            <Typography
              variant="h4"
              sx={{ color: "#f5f5f5", mt: 1, fontWeight: 700 }}
            >
              {instrument}
            </Typography>
          </Card>
        </Grid>

        {/* Confidence */}
        <Grid item xs={12} md={6}>
          <Card sx={{ p: 3, height: "100%" }}>
            <Typography variant="caption" sx={{ color: "#737373" }}>
              Confidence Score
            </Typography>
            <Typography
              variant="h4"
              sx={{ color: "#f5f5f5", mt: 1, fontWeight: 700 }}
            >
              {confidence}%
            </Typography>
            <LinearProgress
              variant="determinate"
              value={confidence}
              color={getConfidenceColor(confidence)}
              sx={{ mt: 1.5, height: 6, borderRadius: 3 }}
            />
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}
