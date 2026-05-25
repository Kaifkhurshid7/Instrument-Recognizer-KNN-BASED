import React from "react";
import { Box, Typography, Button, Chip } from "@mui/material";
import GraphicEqIcon from "@mui/icons-material/GraphicEq";
import GitHubIcon from "@mui/icons-material/GitHub";

export default function Header({ onScrollTo }) {
  return (
    <Box mb={6} sx={{ position: "relative", zIndex: 1 }}>
      {/* Top nav bar */}
      <Box
        display="flex"
        alignItems="center"
        justifyContent="space-between"
        mb={5}
        pb={2}
        flexWrap="wrap"
        gap={2}
        sx={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}
      >
        <Box display="flex" alignItems="center" gap={1.5}>
          <GraphicEqIcon sx={{ color: "#3b82f6", fontSize: 28 }} />
          <Typography variant="body1" sx={{ color: "#f5f5f5", fontWeight: 700, letterSpacing: "-0.02em" }}>
            InstruSense
          </Typography>
        </Box>
        <Box display="flex" alignItems="center" gap={{ xs: 2, md: 3 }} flexWrap="wrap">
          <Typography
            variant="body2"
            onClick={() => onScrollTo("howItWorks")}
            sx={{ color: "#737373", cursor: "pointer", "&:hover": { color: "#f5f5f5" } }}
          >
            How it Works
          </Typography>
          <Typography
            variant="body2"
            onClick={() => onScrollTo("capabilities")}
            sx={{ color: "#737373", cursor: "pointer", "&:hover": { color: "#f5f5f5" } }}
          >
            Features
          </Typography>
          <Typography
            variant="body2"
            onClick={() => onScrollTo("apiDocs")}
            sx={{ color: "#737373", cursor: "pointer", "&:hover": { color: "#f5f5f5" } }}
          >
            API Docs
          </Typography>
          <Button
            variant="outlined"
            size="small"
            startIcon={<GitHubIcon />}
            href="https://github.com/Kaifkhurshid7/Instrument-Recognizer-KNN-BASED"
            target="_blank"
            sx={{ borderColor: "rgba(255,255,255,0.15)", color: "#a3a3a3", fontSize: "0.75rem" }}
          >
            GitHub
          </Button>
        </Box>
      </Box>

      {/* Hero section */}
      <Box sx={{ maxWidth: 700 }}>
        <Chip
          label="AI-Powered Audio Analysis"
          size="small"
          sx={{
            mb: 2,
            bgcolor: "rgba(59,130,246,0.1)",
            color: "#3b82f6",
            border: "1px solid rgba(59,130,246,0.2)",
            fontWeight: 600,
          }}
        />

        <Typography
          variant="h2"
          sx={{
            color: "#f5f5f5",
            mb: 2,
            fontSize: { xs: "1.8rem", sm: "2.2rem", md: "3rem" },
            lineHeight: 1.1,
          }}
        >
          Identify instruments{" "}
          <Box component="span" sx={{ color: "#3b82f6" }}>
            in any audio.
          </Box>
        </Typography>

        <Typography variant="body1" sx={{ mb: 4, lineHeight: 1.8, maxWidth: 550 }}>
          Upload a music file and get instant classification across 11 instrument classes.
          See exactly why the model made its decision through spectral fingerprint analysis,
          waveform visualization, and probability breakdowns.
        </Typography>

        {/* Stats row */}
        <Box display="flex" gap={{ xs: 2, md: 4 }} mb={4} flexWrap="wrap">
          <Box>
            <Typography variant="h4" sx={{ color: "#f5f5f5", fontWeight: 700 }}>11</Typography>
            <Typography variant="caption" sx={{ color: "#737373" }}>Instrument Classes</Typography>
          </Box>
          <Box>
            <Typography variant="h4" sx={{ color: "#f5f5f5", fontWeight: 700 }}>26-D</Typography>
            <Typography variant="caption" sx={{ color: "#737373" }}>Feature Vector</Typography>
          </Box>
          <Box>
            <Typography variant="h4" sx={{ color: "#f5f5f5", fontWeight: 700 }}>61.9%</Typography>
            <Typography variant="caption" sx={{ color: "#737373" }}>CV Accuracy</Typography>
          </Box>
          <Box>
            <Typography variant="h4" sx={{ color: "#f5f5f5", fontWeight: 700 }}>3724</Typography>
            <Typography variant="caption" sx={{ color: "#737373" }}>Training Samples</Typography>
          </Box>
        </Box>

        {/* Tech tags */}
        <Box display="flex" gap={1} flexWrap="wrap">
          <Chip label="KNN (K=3)" size="small" variant="outlined" />
          <Chip label="Cosine Similarity" size="small" variant="outlined" />
          <Chip label="Librosa DSP" size="small" variant="outlined" />
          <Chip label="Flask API" size="small" variant="outlined" />
          <Chip label="React + MUI" size="small" variant="outlined" />
        </Box>
      </Box>
    </Box>
  );
}
