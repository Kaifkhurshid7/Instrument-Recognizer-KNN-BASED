/**
 * Radar Chart Component
 * ----------------------
 * Feature fingerprint comparison with clean dark styling.
 */

import React from "react";
import { Card, Typography, Box } from "@mui/material";
import { Radar } from "react-chartjs-2";
import { RADAR_FEATURES } from "../../config/constants";

export default function RadarChart({ featureVector, comparedVector, instrument }) {
  const data = {
    labels: RADAR_FEATURES.map((f) => f.name),
    datasets: [
      {
        label: "Your Audio",
        data: RADAR_FEATURES.map((f) => featureVector[f.index]),
        borderColor: "#3b82f6",
        backgroundColor: "rgba(59, 130, 246, 0.12)",
        borderWidth: 2,
        pointBackgroundColor: "#3b82f6",
        pointRadius: 3,
      },
      {
        label: `${instrument} Average`,
        data: RADAR_FEATURES.map((f) => comparedVector[f.index]),
        borderColor: "#f59e0b",
        backgroundColor: "rgba(245, 158, 11, 0.08)",
        borderWidth: 2,
        pointBackgroundColor: "#f59e0b",
        pointRadius: 3,
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: "bottom",
        labels: {
          color: "#a3a3a3",
          padding: 20,
          font: { size: 12 },
        },
      },
    },
    scales: {
      r: {
        ticks: {
          color: "#525252",
          backdropColor: "transparent",
          font: { size: 9 },
        },
        grid: { color: "rgba(255,255,255,0.04)" },
        angleLines: { color: "rgba(255,255,255,0.04)" },
        pointLabels: { color: "#737373", font: { size: 9 } },
      },
    },
  };

  return (
    <Card sx={{ p: 3, mb: 3 }}>
      <Box mb={2}>
        <Typography variant="h5" sx={{ color: "#f5f5f5" }}>
          Feature Fingerprint
        </Typography>
        <Typography variant="body2" sx={{ mt: 0.5 }}>
          Spectral profile comparison — input vs. database average for{" "}
          {instrument}
        </Typography>
      </Box>
      <Box sx={{ maxWidth: 460, mx: "auto" }}>
        <Radar data={data} options={options} />
      </Box>
    </Card>
  );
}
