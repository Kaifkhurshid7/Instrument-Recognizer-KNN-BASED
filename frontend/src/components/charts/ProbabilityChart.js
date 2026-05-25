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
          i === 0 ? "#3b82f6" : "rgba(255,255,255,0.06)"
        ),
        borderRadius: 4,
        barThickness: 22,
      },
    ],
  };

  const options = {
    indexAxis: "y",
    responsive: true,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: "#1a1a1a",
        borderColor: "rgba(255,255,255,0.08)",
        borderWidth: 1,
        titleColor: "#f5f5f5",
        bodyColor: "#a3a3a3",
        callbacks: { label: (ctx) => ` ${ctx.parsed.x.toFixed(1)}%` },
      },
    },
    scales: {
      x: {
        max: 100,
        ticks: { color: "#404040", font: { size: 10 }, callback: (v) => `${v}%` },
        grid: { color: "rgba(255,255,255,0.02)" },
        border: { display: false },
      },
      y: {
        ticks: { color: "#a3a3a3", font: { size: 11 } },
        grid: { display: false },
        border: { display: false },
      },
    },
  };

  return (
    <Card sx={{ p: 3, bgcolor: "#111111" }}>
      <Typography variant="h6" sx={{ color: "#f5f5f5", fontSize: "0.9rem", fontWeight: 600, textTransform: "none", letterSpacing: 0, mb: 0.5 }}>
        Classification Probabilities
      </Typography>
      <Typography variant="caption" sx={{ color: "#525252" }}>
        KNN weighted vote across all classes
      </Typography>
      <Box mt={2}>
        <Bar data={data} options={options} />
      </Box>
    </Card>
  );
}
