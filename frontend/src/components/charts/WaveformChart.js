/**
 * Waveform Chart Component
 * -------------------------
 * Renders the time-domain audio signal as a line chart.
 * Shows the raw amplitude over time for visual inspection.
 */

import React from "react";
import { Card, Typography, Box } from "@mui/material";
import { Line } from "react-chartjs-2";
import { COLORS } from "../../config/constants";

export default function WaveformChart({ time, amplitude }) {
  const data = {
    labels: time,
    datasets: [
      {
        data: amplitude,
        borderColor: COLORS.primary,
        borderWidth: 1.5,
        pointRadius: 0,
        fill: true,
        backgroundColor: COLORS.primaryAlpha,
        tension: 0.1,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: {
        display: false,
      },
      y: {
        ticks: { color: COLORS.text, font: { size: 10 } },
        grid: { color: COLORS.grid },
        border: { display: false },
      },
    },
    interaction: { intersect: false, mode: "index" },
  };

  return (
    <Card sx={{ p: 3, mb: 4 }}>
      <Typography variant="h6" mb={2}>
        Audio Waveform
      </Typography>
      <Box sx={{ height: 200 }}>
        <Line data={data} options={options} />
      </Box>
      <Typography variant="caption" color="text.secondary" mt={1} display="block">
        Time-domain signal representation of the input audio
      </Typography>
    </Card>
  );
}
