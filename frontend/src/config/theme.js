/**
 * Application Theme Configuration
 * --------------------------------
 * Centralized MUI dark theme with custom palette, typography,
 * and component overrides for a premium analytical UI.
 */

import { createTheme } from "@mui/material";

const theme = createTheme({
  palette: {
    mode: "dark",
    primary: { main: "#6ee7b7" },
    secondary: { main: "#818cf8" },
    error: { main: "#f87171" },
    warning: { main: "#fbbf24" },
    background: {
      default: "#0a0e1a",
      paper: "#111827",
    },
    text: {
      primary: "#f1f5f9",
      secondary: "#94a3b8",
    },
    divider: "rgba(148, 163, 184, 0.08)",
  },
  typography: {
    fontFamily: '"Inter", "Segoe UI", system-ui, sans-serif',
    h3: { fontWeight: 700, letterSpacing: "-0.02em" },
    h5: { fontWeight: 600, letterSpacing: "-0.01em" },
    h6: { fontWeight: 600 },
    body2: { color: "#94a3b8" },
  },
  shape: { borderRadius: 12 },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 16,
          border: "1px solid rgba(148, 163, 184, 0.08)",
          backgroundImage: "none",
          boxShadow: "0 4px 24px rgba(0, 0, 0, 0.4)",
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: "none",
          fontWeight: 600,
          borderRadius: 10,
          padding: "10px 20px",
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: "none",
        },
      },
    },
  },
});

export default theme;
