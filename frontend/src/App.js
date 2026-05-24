/**
 * Instrument Recognizer - Main Application
 * ==========================================
 * Root component that orchestrates the analysis workflow:
 * 1. User uploads an audio file
 * 2. File is sent to Flask backend for spectral analysis
 * 3. Results are displayed with multiple explainability views
 *
 * Architecture: Thin orchestrator that delegates rendering
 * to focused child components.
 */

import React, { useState } from "react";
import {
  ThemeProvider,
  CssBaseline,
  Container,
  Alert,
  Fade,
  CircularProgress,
  Box,
} from "@mui/material";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  RadialLinearScale,
  PointElement,
  LineElement,
  BarElement,
  Tooltip,
  Legend,
  Filler,
} from "chart.js";

import theme from "./config/theme";
import { analyzeAudio } from "./services/api";
import { downloadCSVReport } from "./utils/reportGenerator";
import Header from "./components/Header";
import FileUpload from "./components/FileUpload";
import ResultCard from "./components/ResultCard";
import WaveformChart from "./components/charts/WaveformChart";
import ProbabilityChart from "./components/charts/ProbabilityChart";
import RadarChart from "./components/charts/RadarChart";
import FeatureTable from "./components/FeatureTable";

// Chart.js registration (required once at app level)
ChartJS.register(
  CategoryScale,
  LinearScale,
  RadialLinearScale,
  PointElement,
  LineElement,
  BarElement,
  Tooltip,
  Legend,
  Filler
);

export default function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  /**
   * Send the selected audio file to the backend for classification.
   * Manages loading state and error handling.
   */
  const handleAnalyze = async () => {
    if (!file) {
      setError("Please select an audio file first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const data = await analyzeAudio(file);
      setResult(data);
    } catch (err) {
      const message =
        err.response?.data?.error ||
        "Could not connect to the backend. Make sure the server is running.";
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth="md" sx={{ py: 6 }}>
        <Header />

        <FileUpload
          file={file}
          onFileSelect={setFile}
          onAnalyze={handleAnalyze}
          onDownload={() => downloadCSVReport(result)}
          loading={loading}
          hasResult={!!result}
        />

        {/* Loading indicator */}
        {loading && (
          <Box display="flex" justifyContent="center" my={4}>
            <CircularProgress color="primary" />
          </Box>
        )}

        {/* Error display */}
        {error && (
          <Alert severity="error" sx={{ mb: 3, borderRadius: 2 }}>
            {error}
          </Alert>
        )}

        {/* Results section - renders only after successful analysis */}
        {result && (
          <Fade in timeout={500}>
            <div>
              <ResultCard
                instrument={result.instrument}
                confidence={result.confidence_score}
              />

              <ProbabilityChart probabilities={result.knn_probabilities} />

              <WaveformChart
                time={result.waveform.time}
                amplitude={result.waveform.amplitude}
              />

              <RadarChart
                featureVector={result.feature_vector}
                comparedVector={result.compared_vector}
                instrument={result.instrument}
              />

              <FeatureTable
                featureVector={result.feature_vector}
                comparedVector={result.compared_vector}
              />
            </div>
          </Fade>
        )}
      </Container>
    </ThemeProvider>
  );
}
