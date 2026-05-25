import React from "react";
import { Box } from "@mui/material";

// SVG music notes and waveform decorations rendered as absolute-positioned background elements
export default function BackgroundDecor() {
  return (
    <Box
      sx={{
        position: "fixed",
        top: 0,
        left: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
        zIndex: 0,
        overflow: "hidden",
      }}
    >
      {/* Gradient mesh background */}
      <Box
        sx={{
          position: "absolute",
          top: "-20%",
          right: "-10%",
          width: 600,
          height: 600,
          borderRadius: "50%",
          background: "radial-gradient(circle, rgba(59,130,246,0.06) 0%, transparent 70%)",
        }}
      />
      <Box
        sx={{
          position: "absolute",
          bottom: "-10%",
          left: "-5%",
          width: 500,
          height: 500,
          borderRadius: "50%",
          background: "radial-gradient(circle, rgba(139,92,246,0.04) 0%, transparent 70%)",
        }}
      />

      {/* Ruler / grid lines */}
      <svg
        width="100%"
        height="100%"
        style={{ position: "absolute", top: 0, left: 0, opacity: 0.03 }}
      >
        {Array.from({ length: 20 }).map((_, i) => (
          <line
            key={`h-${i}`}
            x1="0"
            y1={`${i * 5}%`}
            x2="100%"
            y2={`${i * 5}%`}
            stroke="#fff"
            strokeWidth="0.5"
          />
        ))}
        {Array.from({ length: 30 }).map((_, i) => (
          <line
            key={`v-${i}`}
            x1={`${i * 3.33}%`}
            y1="0"
            x2={`${i * 3.33}%`}
            y2="100%"
            stroke="#fff"
            strokeWidth="0.5"
          />
        ))}
      </svg>

      {/* Floating music notes */}
      <svg
        width="100%"
        height="100%"
        style={{ position: "absolute", top: 0, left: 0, opacity: 0.04 }}
      >
        {/* Treble clef shape (simplified) */}
        <text x="5%" y="15%" fontSize="80" fill="#3b82f6" fontFamily="serif">♪</text>
        <text x="88%" y="25%" fontSize="60" fill="#8b5cf6" fontFamily="serif">♫</text>
        <text x="75%" y="70%" fontSize="90" fill="#3b82f6" fontFamily="serif">♩</text>
        <text x="12%" y="80%" fontSize="50" fill="#8b5cf6" fontFamily="serif">♬</text>
        <text x="92%" y="85%" fontSize="70" fill="#3b82f6" fontFamily="serif">♪</text>
        <text x="45%" y="5%" fontSize="40" fill="#6366f1" fontFamily="serif">♫</text>

        {/* Waveform sine wave decoration */}
        <path
          d="M0,300 Q50,250 100,300 T200,300 T300,300 T400,300 T500,300 T600,300 T700,300 T800,300 T900,300 T1000,300 T1100,300 T1200,300 T1300,300 T1400,300"
          fill="none"
          stroke="#3b82f6"
          strokeWidth="1"
          opacity="0.3"
        />
        <path
          d="M0,600 Q80,550 160,600 T320,600 T480,600 T640,600 T800,600 T960,600 T1120,600 T1280,600 T1440,600"
          fill="none"
          stroke="#8b5cf6"
          strokeWidth="0.8"
          opacity="0.2"
        />
      </svg>

      {/* Frequency spectrum bars (left edge) */}
      <Box
        sx={{
          position: "absolute",
          left: 0,
          top: "30%",
          display: "flex",
          flexDirection: "column",
          gap: "3px",
          opacity: 0.06,
        }}
      >
        {[40, 65, 30, 80, 55, 45, 70, 35, 60, 50, 75, 25, 55, 40, 68].map((w, i) => (
          <Box
            key={i}
            sx={{
              width: w,
              height: 3,
              bgcolor: "#3b82f6",
              borderRadius: 1,
            }}
          />
        ))}
      </Box>

      {/* Frequency spectrum bars (right edge) */}
      <Box
        sx={{
          position: "absolute",
          right: 0,
          top: "50%",
          display: "flex",
          flexDirection: "column",
          gap: "3px",
          opacity: 0.05,
          alignItems: "flex-end",
        }}
      >
        {[50, 35, 70, 45, 60, 30, 75, 55, 40, 65, 50, 80, 35].map((w, i) => (
          <Box
            key={i}
            sx={{
              width: w,
              height: 3,
              bgcolor: "#8b5cf6",
              borderRadius: 1,
            }}
          />
        ))}
      </Box>
    </Box>
  );
}
