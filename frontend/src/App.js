import React, { useState, useRef } from "react";
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

  // Section refs for nav scroll
  const howItWorksRef = useRef(null);
  const capabilitiesRef = useRef(null);
  const apiDocsRef = useRef(null);

  const handleScrollTo = (section) => {
    const refs = {
      howItWorks: howItWorksRef,
      capabilities: capabilitiesRef,
      apiDocs: apiDocsRef,
    };
    refs[section]?.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

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

      <Container maxWidth="lg" sx={{ py: { xs: 3, md: 5 }, position: "relative", zIndex: 1 }}>
        <Header onScrollTo={handleScrollTo} />

        {/* Upload section — first after hero */}
        <FileUpload
          file={file}
          onFileSelect={setFile}
          onAnalyze={handleAnalyze}
          onDownload={() => downloadCSVReport(result)}
          loading={loading}
          hasResult={!!result}
        />

        {/* Loading */}
        {loading && (
          <Box display="flex" justifyContent="center" my={5}>
            <CircularProgress size={32} sx={{ color: "#3b82f6" }} />
          </Box>
        )}

        {/* Error */}
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

        {/* Results */}
        {result && (
          <Fade in timeout={400}>
            <Box sx={{ mb: 6 }}>
              <Divider sx={{ mb: 4 }} />

              <ResultCard
                instrument={result.instrument}
                confidence={result.confidence_score}
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
            </Box>
          </Fade>
        )}

        {/* How it Works */}
        <Box ref={howItWorksRef} sx={{ scrollMarginTop: "2rem" }}>
          <HowItWorks />
        </Box>

        {/* Capabilities */}
        <Box ref={capabilitiesRef} sx={{ scrollMarginTop: "2rem" }}>
          <FeaturesSection />
        </Box>

        {/* API Docs */}
        <Box ref={apiDocsRef} sx={{ scrollMarginTop: "2rem" }}>
          <ApiDocs />
        </Box>

        <Footer />
      </Container>
    </ThemeProvider>
  );
}
