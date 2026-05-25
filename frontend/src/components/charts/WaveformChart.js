import React from "react";
import { Card, Typography, Box } from "@mui/material";
import { Line } from "react-chartjs-2";

export default function WaveformChart({ time, amplitude }) {
  const data = {
    labels: time,
    datasets: [
      {
        data: amplitude,
        borderColor: "#3b82f6",
        borderWidth: 1.2,
        pointRadius: 0,
        fill: true,
        backgroundColor: "rgba(59, 130, 246, 0.08)",
        tension: 0.2,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { display: false },
      y: {
        ticks: { color: "#404040", font: { size: 9 } },
        grid: { color: "rgba(255,255,255,0.02)" },
        border: { display: false },
      },
    },
    interaction: { intersect: false, mode: "index" },
  };

  return (
    <Card sx={{ p: 3, bgcolor: "#111111", height: "100%" }}>
      <Typography variant="h6" sx={{ color: "#f5f5f5", fontSize: "0.9rem", fontWeight: 600, textTransform: "none", letterSpacing: 0, mb: 0.5 }}>
        Waveform
      </Typography>
      <Typography variant="caption" sx={{ color: "#525252" }}>
        Time-domain signal
      </Typography>
      <Box sx={{ height: 160, mt: 2 }}>
        <Line data={data} options={options} />
      </Box>
    </Card>
  );
}
