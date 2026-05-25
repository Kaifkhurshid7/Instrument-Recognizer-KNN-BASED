/**
 * Radar Chart Component
 * ----------------------
 * Compares the input audio's feature fingerprint against the
 * database average for the predicted instrument class.
 *
 * Uses a curated subset of 12 features (from the full 26-D vector)
 * to keep the radar chart readable while showing key dimensions.
 */

import React from "react";
import { Card, Typography, Box } from "@mui/material";
import { Radar } from "react-chartjs-2";
import { RADAR_FEATURES, COLORS } from "../../config/constants";

export default function RadarChart({ featureVector, comparedVector, instrument }) {
  const data = {
    labels: RADAR_FEATURES.map((f) => f.name),
    datasets: [
      {
        label: "Your Audio",
        data: RADAR_FEATURES.map((f) => featureVector[f.index]),
        borderColor: COLORS.primary,
        backgroundColor: COLORS.primaryAlpha,
        borderWidth: 2,
        pointBackgroundColor: COLORS.primary,
        pointRadius: 3,
      },
      {
        label: `${instrument} Average`,
        data: RADAR_FEATURES.map((f) => comparedVector[f.index]),
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
        pointLabels: { color: COLORS.text, font: { size: 9 } },
      },
    },
  };

  return (
    <Card sx={{ p: 3, mb: 4 }}>
      <Typography variant="h6" mb={1}>
        Feature Fingerprint Comparison
      </Typography>
      <Typography variant="caption" color="text.secondary" display="block" mb={2}>
        Key spectral dimensions — your audio vs. the database average for {instrument}
      </Typography>
      <Box sx={{ maxWidth: 480, mx: "auto" }}>
        <Radar data={data} options={options} />
      </Box>
    </Card>
  );
}
