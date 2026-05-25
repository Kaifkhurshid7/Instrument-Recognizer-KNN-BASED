import { createTheme } from "@mui/material";

const theme = createTheme({
  palette: {
    mode: "dark",
    primary: { main: "#3b82f6" },
    secondary: { main: "#6366f1" },
    error: { main: "#ef4444" },
    warning: { main: "#f59e0b" },
    success: { main: "#10b981" },
    background: {
      default: "#0a0a0a",
      paper: "#141414",
    },
    text: {
      primary: "#f5f5f5",
      secondary: "#a3a3a3",
    },
    divider: "rgba(255, 255, 255, 0.06)",
  },
  typography: {
    fontFamily: '"Inter", -apple-system, "Segoe UI", sans-serif',
    h2: {
      fontWeight: 700,
      fontSize: "2.5rem",
      letterSpacing: "-0.03em",
      lineHeight: 1.2,
    },
    h4: {
      fontWeight: 600,
      fontSize: "1.5rem",
      letterSpacing: "-0.02em",
    },
    h5: {
      fontWeight: 600,
      fontSize: "1.25rem",
      letterSpacing: "-0.01em",
    },
    h6: {
      fontWeight: 600,
      fontSize: "1rem",
      textTransform: "uppercase",
      letterSpacing: "0.08em",
      color: "#ef4444",
    },
    body1: {
      fontSize: "1rem",
      lineHeight: 1.7,
      color: "#a3a3a3",
    },
    body2: {
      fontSize: "0.875rem",
      color: "#737373",
    },
    caption: {
      fontSize: "0.8rem",
      color: "#525252",
    },
  },
  shape: { borderRadius: 8 },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          backgroundColor: "#0a0a0a",
          scrollbarWidth: "thin",
          scrollbarColor: "#333 #0a0a0a",
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          border: "1px solid rgba(255, 255, 255, 0.06)",
          backgroundImage: "none",
          backgroundColor: "#141414",
          boxShadow: "none",
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: "none",
          fontWeight: 600,
          borderRadius: 8,
          padding: "10px 24px",
          fontSize: "0.9rem",
        },
        contained: {
          backgroundColor: "#3b82f6",
          "&:hover": { backgroundColor: "#2563eb" },
          boxShadow: "none",
        },
        outlined: {
          borderColor: "rgba(255,255,255,0.15)",
          color: "#f5f5f5",
          "&:hover": {
            borderColor: "rgba(255,255,255,0.3)",
            backgroundColor: "rgba(255,255,255,0.03)",
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: "none",
          backgroundColor: "#141414",
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderColor: "rgba(255,255,255,0.1)",
          fontSize: "0.75rem",
        },
      },
    },
    MuiLinearProgress: {
      styleOverrides: {
        root: {
          backgroundColor: "rgba(255,255,255,0.05)",
        },
      },
    },
  },
});

export default theme;
