import React from "react";
import { Box } from "@mui/material";

// Floating music-themed SVG icons rendered as background decoration
const musicIcons = [
  // Musical notes
  { icon: "♪", top: "5%", left: "3%", size: 48, opacity: 0.04, rotate: -15 },
  { icon: "♫", top: "12%", right: "5%", size: 64, opacity: 0.05, rotate: 10 },
  { icon: "♩", top: "25%", left: "8%", size: 36, opacity: 0.03, rotate: 20 },
  { icon: "♬", top: "40%", right: "3%", size: 52, opacity: 0.04, rotate: -8 },
  { icon: "♪", top: "55%", left: "2%", size: 44, opacity: 0.03, rotate: 12 },
  { icon: "♫", top: "70%", right: "7%", size: 56, opacity: 0.04, rotate: -20 },
  { icon: "♩", top: "82%", left: "6%", size: 40, opacity: 0.03, rotate: 5 },
  { icon: "♬", top: "90%", right: "4%", size: 48, opacity: 0.04, rotate: -12 },
  // Extra scattered
  { icon: "♪", top: "18%", left: "85%", size: 32, opacity: 0.03, rotate: 25 },
  { icon: "♫", top: "35%", left: "92%", size: 42, opacity: 0.03, rotate: -5 },
  { icon: "♩", top: "48%", left: "5%", size: 28, opacity: 0.02, rotate: 30 },
  { icon: "♬", top: "65%", left: "90%", size: 38, opacity: 0.03, rotate: -18 },
];

// Waveform-like decorative lines
function WaveformLine({ top, opacity = 0.04 }) {
  return (
    <Box
      sx={{
        position: "absolute",
        top,
        left: 0,
        right: 0,
        height: "60px",
        opacity,
        overflow: "hidden",
        pointerEvents: "none",
      }}
    >
      <svg width="100%" height="60" preserveAspectRatio="none" viewBox="0 0 1200 60">
        <path
          d="M0,30 Q50,10 100,30 T200,30 T300,30 T400,30 T500,30 T600,30 T700,30 T800,30 T900,30 T1000,30 T1100,30 T1200,30"
          fill="none"
          stroke="#3b82f6"
          strokeWidth="1.5"
        />
        <path
          d="M0,30 Q75,5 150,30 T300,30 T450,30 T600,30 T750,30 T900,30 T1050,30 T1200,30"
          fill="none"
          stroke="#6366f1"
          strokeWidth="1"
        />
      </svg>
    </Box>
  );
}

// Circular equalizer-style rings
function EqRing({ top, left, size = 120, opacity = 0.03 }) {
  return (
    <Box
      sx={{
        position: "absolute",
        top,
        left,
        width: size,
        height: size,
        borderRadius: "50%",
        border: "1px solid #3b82f6",
        opacity,
        pointerEvents: "none",
      }}
    />
  );
}

export default function BackgroundDecor() {
  return (
    <Box
      sx={{
        position: "fixed",
        inset: 0,
        overflow: "hidden",
        pointerEvents: "none",
        zIndex: 0,
      }}
    >
      {/* Gradient orbs */}
      <Box
        sx={{
          position: "absolute",
          top: "-10%",
          right: "-5%",
          width: 500,
          height: 500,
          borderRadius: "50%",
          background: "radial-gradient(circle, rgba(59,130,246,0.06) 0%, transparent 70%)",
        }}
      />
      <Box
        sx={{
          position: "absolute",
          bottom: "-10%",
          left: "-5%",
          width: 600,
          height: 600,
          borderRadius: "50%",
          background: "radial-gradient(circle, rgba(99,102,241,0.05) 0%, transparent 70%)",
        }}
      />
      <Box
        sx={{
          position: "absolute",
          top: "40%",
          left: "50%",
          transform: "translateX(-50%)",
          width: 800,
          height: 400,
          borderRadius: "50%",
          background: "radial-gradient(ellipse, rgba(59,130,246,0.03) 0%, transparent 60%)",
        }}
      />

      {/* Floating music icons */}
      {musicIcons.map((item, i) => (
        <Box
          key={i}
          sx={{
            position: "absolute",
            top: item.top,
            left: item.left,
            right: item.right,
            fontSize: item.size,
            opacity: item.opacity,
            transform: `rotate(${item.rotate}deg)`,
            color: "#3b82f6",
            fontFamily: "serif",
            userSelect: "none",
          }}
        >
          {item.icon}
        </Box>
      ))}

      {/* Waveform lines */}
      <WaveformLine top="15%" opacity={0.03} />
      <WaveformLine top="75%" opacity={0.025} />

      {/* EQ rings */}
      <EqRing top="20%" left="80%" size={160} opacity={0.025} />
      <EqRing top="60%" left="5%" size={120} opacity={0.02} />
      <EqRing top="45%" left="70%" size={80} opacity={0.03} />
    </Box>
  );
}
