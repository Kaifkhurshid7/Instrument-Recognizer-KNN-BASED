import React from "react";
import { Box, Typography, Divider } from "@mui/material";
import GraphicEqIcon from "@mui/icons-material/GraphicEq";

export default function Footer() {
  return (
    <Box sx={{ position: "relative", zIndex: 1, mt: 8 }}>
      <Divider sx={{ mb: 4 }} />
      <Box
        display="flex"
        justifyContent="space-between"
        alignItems="center"
        flexWrap="wrap"
        gap={2}
        pb={4}
      >
        <Box display="flex" alignItems="center" gap={1}>
          <GraphicEqIcon sx={{ color: "#3b82f6", fontSize: 20 }} />
          <Typography variant="body2" sx={{ color: "#525252" }}>
            InstruSense — Musical Instrument Recognition
          </Typography>
        </Box>
        <Box display="flex" gap={3}>
          <Typography variant="caption" sx={{ color: "#525252" }}>
            Built with Flask + React
          </Typography>
          <Typography variant="caption" sx={{ color: "#525252" }}>
            Dataset: IRMAS (UPF-MTG)
          </Typography>
          <Typography variant="caption" sx={{ color: "#525252" }}>
            Educational & Research Use
          </Typography>
        </Box>
      </Box>
    </Box>
  );
}
