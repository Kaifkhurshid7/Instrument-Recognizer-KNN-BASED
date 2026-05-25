/**
 * Waveform Chart Component
 * -------------------------
 * Renders the time-domain audio signal. Clean minimal style.
 */

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
        borderWidth: 1,
        pointRadius: 0,
        fill: true,
        backgroundColor: "rgba(59, 130, 246, 0.06)",
        tension: 0.1,
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
        ticks: { color: "#525252", font: { size: 10 } },
        grid: { color: "rgba(255,255,255,0.03)" },
        border: { display: false },
      },
    },
    interaction: { intersect: false, mode: "index" },
  };

  return (
    <Card sx={{ p: 3, mb: 3 }}>
      <Box mb={2}>
        <Typography variant="h5" sx={{ color: "#f5f5f5" }}>
          Audio Waveform
        </Typography>
        <Typography variant="body2" sx={{ mt: 0.5 }}>
          Time-domain signal of the input audio
        </Typography>
      </Box>
      <Box sx={{ height: 180 }}>
        <Line data={data} options={options} />
      </Box>
    </Card>
  );
}
