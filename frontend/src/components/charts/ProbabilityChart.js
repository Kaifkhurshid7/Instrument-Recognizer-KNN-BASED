/**
 * Probability Chart Component
 * ----------------------------
 * Horizontal bar chart showing KNN classification probabilities
 * for all instrument classes, sorted by score.
 */

import React from "react";
import { Card, Typography } from "@mui/material";
import { Bar } from "react-chartjs-2";
import { COLORS } from "../../config/constants";

export default function ProbabilityChart({ probabilities }) {
  const sorted = [...probabilities].sort((a, b) => b.score - a.score);

  const data = {
    labels: sorted.map((p) => p.name),
    datasets: [
      {
        label: "Probability (%)",
        data: sorted.map((p) => p.score),
        backgroundColor: sorted.map((_, i) =>
          i === 0 ? COLORS.primary : COLORS.secondary
        ),
        borderRadius: 6,
        barThickness: 20,
      },
    ],
  };

  const options = {
    indexAxis: "y",
    responsive: true,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: (ctx) => `${ctx.parsed.x.toFixed(1)}%`,
        },
      },
    },
    scales: {
      x: {
        max: 100,
        ticks: { color: COLORS.text, callback: (v) => `${v}%` },
        grid: { color: COLORS.grid },
        border: { display: false },
      },
      y: {
        ticks: { color: COLORS.text, font: { size: 11 } },
        grid: { display: false },
        border: { display: false },
      },
    },
  };

  return (
    <Card sx={{ p: 3, mb: 4 }}>
      <Typography variant="h6" mb={2}>
        Classification Probabilities
      </Typography>
      <Bar data={data} options={options} />
    </Card>
  );
}
