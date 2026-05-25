import React from "react";
import { Card, Typography, Box } from "@mui/material";
import { Radar } from "react-chartjs-2";
import { RADAR_FEATURES } from "../../config/constants";

export default function RadarChart({ featureVector, comparedVector, instrument }) {
  const data = {
    labels: RADAR_FEATURES.map((f) => f.name),
    datasets: [
      {
        label: "Input Audio",
        data: RADAR_FEATURES.map((f) => featureVector[f.index]),
        borderColor: "#3b82f6",
        backgroundColor: "rgba(59, 130, 246, 0.1)",
        borderWidth: 2,
        pointBackgroundColor: "#3b82f6",
        pointRadius: 3,
      },
      {
        label: `${instrument} Avg`,
        data: RADAR_FEATURES.map((f) => comparedVector[f.index]),
        borderColor: "#f59e0b",
        backgroundColor: "rgba(245, 158, 11, 0.06)",
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
        labels: { color: "#737373", padding: 16, font: { size: 11 } },
      },
    },
    scales: {
      r: {
        ticks: { color: "#404040", backdropColor: "transparent", font: { size: 8 } },
        grid: { color: "rgba(255,255,255,0.03)" },
        angleLines: { color: "rgba(255,255,255,0.03)" },
        pointLabels: { color: "#525252", font: { size: 8 } },
      },
    },
  };

  return (
    <Card sx={{ p: 3, bgcolor: "#111111", height: "100%" }}>
      <Typography variant="h6" sx={{ color: "#f5f5f5", fontSize: "0.9rem", fontWeight: 600, textTransform: "none", letterSpacing: 0, mb: 0.5 }}>
        Feature Fingerprint
      </Typography>
      <Typography variant="caption" sx={{ color: "#525252" }}>
        Input vs. {instrument} average
      </Typography>
      <Box sx={{ maxWidth: 380, mx: "auto", mt: 1 }}>
        <Radar data={data} options={options} />
      </Box>
    </Card>
  );
}
