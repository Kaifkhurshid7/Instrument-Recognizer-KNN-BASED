import React from "react";
import { Card, Typography, Box, LinearProgress, Grid } from "@mui/material";
import LibraryMusicIcon from "@mui/icons-material/LibraryMusic";
import SpeedIcon from "@mui/icons-material/Speed";

export default function ResultCard({ instrument, confidence }) {
  const getColor = (score) => {
    if (score >= 80) return "#10b981";
    if (score >= 50) return "#f59e0b";
    return "#ef4444";
  };

  return (
    <Grid container spacing={3} sx={{ mb: 4 }}>
      {/* Instrument */}
      <Grid item xs={12} md={7}>
        <Card
          sx={{
            p: 4,
            height: "100%",
            bgcolor: "#111111",
            display: "flex",
            alignItems: "center",
            gap: 3,
          }}
        >
          <Box
            sx={{
              width: 56,
              height: 56,
              borderRadius: 3,
              bgcolor: "rgba(59,130,246,0.1)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexShrink: 0,
            }}
          >
            <LibraryMusicIcon sx={{ color: "#3b82f6", fontSize: 28 }} />
          </Box>
          <Box>
            <Typography variant="caption" sx={{ color: "#525252", textTransform: "uppercase", letterSpacing: "0.05em" }}>
              Identified Instrument
            </Typography>
            <Typography variant="h3" sx={{ color: "#f5f5f5", fontWeight: 700, mt: 0.5, fontSize: "2rem" }}>
              {instrument}
            </Typography>
          </Box>
        </Card>
      </Grid>

      {/* Confidence */}
      <Grid item xs={12} md={5}>
        <Card
          sx={{
            p: 4,
            height: "100%",
            bgcolor: "#111111",
            display: "flex",
            alignItems: "center",
            gap: 3,
          }}
        >
          <Box
            sx={{
              width: 56,
              height: 56,
              borderRadius: 3,
              bgcolor: `${getColor(confidence)}15`,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexShrink: 0,
            }}
          >
            <SpeedIcon sx={{ color: getColor(confidence), fontSize: 28 }} />
          </Box>
          <Box sx={{ flex: 1 }}>
            <Typography variant="caption" sx={{ color: "#525252", textTransform: "uppercase", letterSpacing: "0.05em" }}>
              Confidence
            </Typography>
            <Typography variant="h3" sx={{ color: "#f5f5f5", fontWeight: 700, mt: 0.5, fontSize: "2rem" }}>
              {confidence}%
            </Typography>
            <LinearProgress
              variant="determinate"
              value={confidence}
              sx={{
                mt: 1,
                height: 5,
                borderRadius: 3,
                bgcolor: "rgba(255,255,255,0.04)",
                "& .MuiLinearProgress-bar": { bgcolor: getColor(confidence), borderRadius: 3 },
              }}
            />
          </Box>
        </Card>
      </Grid>
    </Grid>
  );
}
