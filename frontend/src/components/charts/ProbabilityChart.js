/**
 * Probability Chart Component
 * ----------------------------
 * Horizontal bar chart showing KNN classification probabilities.
 * Clean dark style matching the documentation aesthetic.
 */

import React from "react";
import { Card, Typography, Box } from "@mui/material";
import { Bar } from "react-chartjs-2";

export default function ProbabilityChart({ probabilities }) {
  const sorted = [...probabilities].sort((a, b) => b.score - a.score);

  const data = {
    labels: sorted.map((p) => p.name),
    datasets: [
      {
        label: "Probability (%)",
        data: sorted.map((p) => p.score),
        backgroundColor: sorted.map((_, i) =>
          i === 0 ? "#3b82f6" : "rgba(255,255,255,0.08)"
        ),
        borderRadius: 4,
        barThickness: 24,
      },
    ],
  };

  const options = {
    indexAxis: "y",
    responsive: true,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: "#1f1f1f",
        borderColor: "rgba(255,255,255,0.1)",
        borderWidth: 1,
        titleColor: "#f5f5f5",
        bodyColor: "#a3a3a3",
        callbacks: {
          label: (ctx) => ` ${ctx.parsed.x.toFixed(1)}%`,
        },
      },
    },
    scales: {
      x: {
        max: 100,
        ticks: {
          color: "#525252",
          font: { size: 11 },
          callback: (v) => `${v}%`,
        },
        grid: { color: "rgba(255,255,255,0.03)" },
        border: { display: false },
      },
      y: {
        ticks: { color: "#a3a3a3", font: { size: 12 } },
        grid: { display: false },
        border: { display: false },
      },
    },
  };

  return (
    <Card sx={{ p: 3, mb: 3 }}>
      <Box mb={2}>
        <Typography variant="h5" sx={{ color: "#f5f5f5" }}>
          Probability Distribution
        </Typography>
        <Typography variant="body2" sx={{ mt: 0.5 }}>
          KNN weighted vote across all instrument classes
        </Typography>
      </Box>
      <Bar data={data} options={options} />
    </Card>
  );
}
