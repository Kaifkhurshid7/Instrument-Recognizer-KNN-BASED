/**
 * Radar Chart Component
 * ----------------------
 * Compares the input audio's feature fingerprint against the
 * database average for the predicted instrument class.
 * This is the core "explainability" visualization.
 */

import React from "react";
import { Card, Typography, Box } from "@mui/material";
import { Radar } from "react-chartjs-2";
import { FEATURE_LABELS, COLORS } from "../../config/constants";

export default function RadarChart({ featureVector, comparedVector, instrument }) {
  const data = {
    labels: FEATURE_LABELS.map((f) => f.name),
    datasets: [
      {
        label: "Your Audio",
        data: featureVector,
        borderColor: COLORS.primary,
        backgroundColor: COLORS.primaryAlpha,
        borderWidth: 2,
        pointBackgroundColor: COLORS.primary,
        pointRadius: 3,
      },
      {
        label: `${instrument} Average`,
        data: comparedVector,
        borderColor: COLORS.accent,
        backgroundColor: COLORS.accentAlpha,
        borderWidth: 2,
        pointBackgroundColor: COLORS.accent,
        pointRadius: 3,
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: "bottom",
        labels: { color: COLORS.text, padding: 16 },
      },
    },
    scales: {
      r: {
        ticks: { color: COLORS.text, backdropColor: "transparent" },
        grid: { color: COLORS.grid },
        angleLines: { color: COLORS.grid },
        pointLabels: { color: COLORS.text, font: { size: 10 } },
      },
    },
  };

  return (
    <Card sx={{ p: 3, mb: 4 }}>
      <Typography variant="h6" mb={1}>
        Feature Fingerprint Comparison
      </Typography>
      <Typography variant="caption" color="text.secondary" display="block" mb={2}>
        How your audio's spectral profile compares to the database average
      </Typography>
      <Box sx={{ maxWidth: 500, mx: "auto" }}>
        <Radar data={data} options={options} />
      </Box>
    </Card>
  );
}
