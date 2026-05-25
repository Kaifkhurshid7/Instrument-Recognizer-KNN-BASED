// Backend API URL
export const API_BASE_URL =
  process.env.REACT_APP_API_URL || "http://127.0.0.1:5000";

// Feature labels matching the 26-D backend vector
export const FEATURE_LABELS = [
  { name: "MFCC 1", description: "Energy / loudness" },
  { name: "MFCC 2", description: "Spectral slope" },
  { name: "MFCC 3", description: "Spectral shape" },
  { name: "MFCC 4", description: "Low-frequency detail" },
  { name: "MFCC 5", description: "Mid-low frequency" },
  { name: "MFCC 6", description: "Mid frequency" },
  { name: "MFCC 7", description: "Mid frequency detail" },
  { name: "MFCC 8", description: "Mid-high frequency" },
  { name: "MFCC 9", description: "High frequency" },
  { name: "MFCC 10", description: "High frequency detail" },
  { name: "MFCC 11", description: "Upper harmonics" },
  { name: "MFCC 12", description: "Fine spectral detail" },
  { name: "MFCC 13", description: "Finest spectral detail" },
  { name: "MFCC Delta", description: "Temporal dynamics" },
  { name: "Chroma Mean", description: "Harmonic content" },
  { name: "Chroma Std", description: "Harmony variation" },
  { name: "Spectral Centroid", description: "Brightness" },
  { name: "Centroid Std", description: "Brightness variation" },
  { name: "Spectral Rolloff", description: "High-freq energy boundary" },
  { name: "Rolloff Std", description: "Rolloff variation" },
  { name: "Zero Crossing Rate", description: "Noisiness" },
  { name: "ZCR Std", description: "Noisiness variation" },
  { name: "Spectral Bandwidth", description: "Richness" },
  { name: "Bandwidth Std", description: "Richness variation" },
  { name: "Spectral Flatness", description: "Tonal vs noisy" },
  { name: "RMS Energy Std", description: "Dynamic range" },
];

// Subset for radar chart (full 26 labels would be unreadable)
export const RADAR_FEATURES = [
  { index: 0, name: "MFCC 1 (Energy)" },
  { index: 1, name: "MFCC 2 (Slope)" },
  { index: 4, name: "MFCC 5 (Mid-Low)" },
  { index: 8, name: "MFCC 9 (High)" },
  { index: 13, name: "Temporal Dynamics" },
  { index: 14, name: "Chroma" },
  { index: 16, name: "Centroid" },
  { index: 18, name: "Rolloff" },
  { index: 20, name: "ZCR" },
  { index: 22, name: "Bandwidth" },
  { index: 24, name: "Flatness" },
  { index: 25, name: "Dynamic Range" },
];

// Chart colors
export const COLORS = {
  primary: "#3b82f6",
  primaryAlpha: "rgba(59, 130, 246, 0.12)",
  accent: "#f59e0b",
  accentAlpha: "rgba(245, 158, 11, 0.08)",
  grid: "rgba(255, 255, 255, 0.03)",
  text: "#a3a3a3",
};

// Accepted upload formats
export const ACCEPTED_FORMATS = ".wav,.mp3,.ogg,.flac,.m4a";
