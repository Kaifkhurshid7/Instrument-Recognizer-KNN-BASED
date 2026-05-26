import React, { useState } from "react";
import {
  ThemeProvider,
  CssBaseline,
  Container,
  Alert,
  Fade,
  CircularProgress,
  Box,
  Divider,
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

import BackgroundDecor from "./components/BackgroundDecor";
import Header from "./components/Header";
import FileUpload from "./components/FileUpload";
import ResultCard from "./components/ResultCard";
import WaveformChart from "./components/charts/WaveformChart";
import ProbabilityChart from "./components/charts/ProbabilityChart";
import RadarChart from "./components/charts/RadarChart";
import FeatureTable from "./components/FeatureTable";
import FeaturesSection from "./components/FeaturesSection";
import HowItWorks from "./components/HowItWorks";
import ApiDocs from "./components/ApiDocs";
import Footer from "./components/Footer";

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
      <BackgroundDecor />

      <Container maxWidth="lg" sx={{ py: 5, position: "relative", zIndex: 1 }}>
        {/* Hero + Navigation */}
        <Header />

        {/* Upload + Analysis (top priority section) */}
        <FileUpload
          file={file}
          onFileSelect={setFile}
          onAnalyze={handleAnalyze}
          onDownload={() => downloadCSVReport(result)}
          loading={loading}
          hasResult={!!result}
        />

        {loading && (
          <Box display="flex" justifyContent="center" my={5}>
            <CircularProgress size={32} sx={{ color: "#3b82f6" }} />
          </Box>
        )}

        {error && (
          <Alert
            severity="error"
            sx={{
              mb: 3,
              bgcolor: "rgba(239,68,68,0.08)",
              border: "1px solid rgba(239,68,68,0.2)",
              color: "#fca5a5",
              borderRadius: 2,
            }}
          >
            {error}
          </Alert>
        )}

        {/* Results (shown after analysis) */}
        {result && (
          <Fade in timeout={400}>
            <Box>
              <Divider sx={{ mb: 4 }} />

              <ResultCard
                instrument={result.instrument}
                confidence={result.confidence_score}
                probabilities={result.knn_probabilities}
              />

              <Box
                sx={{
                  display: "grid",
                  gridTemplateColumns: { xs: "1fr", md: "1fr 1fr" },
                  gap: 3,
                  mb: 3,
                }}
              >
                <WaveformChart
                  time={result.waveform.time}
                  amplitude={result.waveform.amplitude}
                />
                <RadarChart
                  featureVector={result.feature_vector}
                  comparedVector={result.compared_vector}
                  instrument={result.instrument}
                />
              </Box>

              <ProbabilityChart probabilities={result.knn_probabilities} />

              <Box mt={3}>
                <FeatureTable
                  featureVector={result.feature_vector}
                  comparedVector={result.compared_vector}
                />
              </Box>

              <Divider sx={{ my: 5 }} />
            </Box>
          </Fade>
        )}

        {/* Informational sections below */}
        <FeaturesSection />

        <HowItWorks />

        <ApiDocs />

        <Footer />
      </Container>
    </ThemeProvider>
  );
}
