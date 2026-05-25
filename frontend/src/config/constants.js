/**
 * Application Constants
 * ---------------------
 * Centralized configuration values used across the frontend.
 */

// Backend API base URL - change for production deployment
export const API_BASE_URL =
  process.env.REACT_APP_API_URL || "http://127.0.0.1:5000";

// Feature labels matching the 26-D backend vector
// Grouped for display: MFCCs shown as a summary, then individual spectral features
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

// Subset of features for the radar chart (too many makes it unreadable)
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

// Chart color palette
export const COLORS = {
  primary: "#6ee7b7",
  primaryAlpha: "rgba(110, 231, 183, 0.2)",
  secondary: "#818cf8",
  secondaryAlpha: "rgba(129, 140, 248, 0.15)",
  accent: "#fbbf24",
  accentAlpha: "rgba(251, 191, 36, 0.15)",
  grid: "rgba(148, 163, 184, 0.06)",
  text: "#94a3b8",
};

// Accepted audio file formats
export const ACCEPTED_FORMATS = ".wav,.mp3,.ogg,.flac,.m4a";
