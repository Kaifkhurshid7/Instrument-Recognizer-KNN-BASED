/**
 * Application Constants
 * ---------------------
 * Centralized configuration values used across the frontend.
 */

// Backend API base URL - change for production deployment
export const API_BASE_URL =
  process.env.REACT_APP_API_URL || "http://127.0.0.1:5000";

// Feature labels used in radar charts and comparison tables
export const FEATURE_LABELS = [
  { name: "MFCC Mean", description: "Average timbre / texture" },
  { name: "MFCC Std", description: "Timbre variation" },
  { name: "Chroma Mean", description: "Average harmonic content" },
  { name: "Chroma Std", description: "Harmony variation" },
  { name: "Spectral Centroid", description: "Brightness of sound" },
  { name: "Centroid Std", description: "Brightness variation" },
  { name: "Spectral Rolloff", description: "Spectral power shape" },
  { name: "Rolloff Std", description: "Rolloff variation" },
  { name: "Zero Crossing Rate", description: "Noisiness measure" },
  { name: "Spectral Bandwidth", description: "Richness of sound" },
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
