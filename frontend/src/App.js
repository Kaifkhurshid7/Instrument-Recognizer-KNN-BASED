import React, { useState, useRef, useMemo } from 'react';
import axios from 'axios';

/* =======================
   Chart.js
======================= */
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
  Filler
} from 'chart.js';

import { Radar, Bar, Line } from 'react-chartjs-2';

/* =======================
   MUI
======================= */
import {
  ThemeProvider,
  createTheme,
  CssBaseline,
  Container,
  Box,
  Typography,
  Button,
  Card,
  CircularProgress,
  Alert,
  Paper
} from '@mui/material';

/* =======================
   Table
======================= */
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow
} from '@mui/material';

/* =======================
   Icons
======================= */
import FileUploadIcon from '@mui/icons-material/FileUpload';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import DescriptionIcon from '@mui/icons-material/Description';

/* =======================
   Chart Register
======================= */
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

/* =======================
   ULTRA DARK PROFESSIONAL THEME
======================= */
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: { main: '#5fcad8' },
    background: {
      default: '#070a10',
      paper: '#111827'
    },
    text: {
      primary: '#eaf0f6',
      secondary: '#9aa4b2'
    }
  },
  typography: {
    fontFamily: '"Inter", system-ui, sans-serif',
    h3: { fontWeight: 600 },
    h4: { fontWeight: 600, letterSpacing: '0.04em' }
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 16,
          border: '1px solid #1f2633',
          boxShadow: '0 12px 36px rgba(0,0,0,0.6)'
        }
      }
    }
  }
});

/* =======================
   FEATURES
======================= */
const features = [
  { name: 'MFCC Mean', desc: 'Average timbre / texture' },
  { name: 'MFCC Std', desc: 'Timbre variation' },
  { name: 'Chroma Mean', desc: 'Average harmonic content' },
  { name: 'Chroma Std', desc: 'Harmony variation' },
  { name: 'Spectral Centroid', desc: 'Brightness of sound' },
  { name: 'Spectral Rolloff', desc: 'Spectral shape' },
  { name: 'Zero Crossing Rate', desc: 'Noisiness' },
  { name: 'Spectral Bandwidth', desc: 'Spectral richness' }
];

export default function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const fileRef = useRef(null);

  /* =======================
     API CALL
  ======================= */
  const analyze = async () => {
    if (!file) return setError('Please select an audio file.');
    setLoading(true);
    setError('');
    setResult(null);

    const fd = new FormData();
    fd.append('audioFile', file);

    try {
      const res = await axios.post('http://127.0.0.1:5000/analyze', fd);
      setResult(res.data);
    } catch {
      setError('Backend error. Please check server.');
    } finally {
      setLoading(false);
    }
  };

  /* =======================
     DOWNLOAD REPORT (CSV)
  ======================= */
  const downloadReport = () => {
    if (!result) return;

    let csv = `Identified Instrument,${result.instrument}\n\n`;
    csv += `Instrument,Probability (%)\n`;

    result.knn_probabilities.forEach(p => {
      csv += `${p.name},${p.score}\n`;
    });

    csv += `\nFeature,Input Value,Database Average\n`;

    features.forEach((f, i) => {
      csv += `${f.name},${result.feature_vector[i]},${result.compared_vector[i]}\n`;
    });

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'instrument_analysis_report.csv';
    link.click();
  };

  /* =======================
     DATA
  ======================= */
  const probabilities = useMemo(() => {
    if (!result) return [];
    return [...(result.knn_probabilities || [])]
      .sort((a, b) => b.score - a.score);
  }, [result]);

  const barData = {
    labels: probabilities.map(p => p.name),
    datasets: [{
      data: probabilities.map(p => p.score),
      backgroundColor: '#5fcad8',
      borderRadius: 6,
      barThickness: 18
    }]
  };

  const radarData = {
    labels: features.map(f => f.name),
    datasets: [
      {
        label: 'Input Audio',
        data: result?.feature_vector || [],
        borderColor: '#5fcad8',
        backgroundColor: 'rgba(95,202,216,0.25)'
      },
      {
        label: 'Database Avg',
        data: result?.compared_vector || [],
        borderColor: '#f1b25b',
        backgroundColor: 'rgba(241,178,91,0.18)'
      }
    ]
  };

  const waveformData = {
    labels: result?.waveform?.time || [],
    datasets: [{
      data: result?.waveform?.amplitude || [],
      borderColor: '#5fcad8',
      borderWidth: 1.5,
      pointRadius: 0,
      fill: true,
      backgroundColor: 'rgba(95,202,216,0.08)'
    }]
  };

  const waveformOptions = {
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { display: false },
      y: {
        ticks: { color: '#9aa4b2' },
        grid: { color: 'rgba(255,255,255,0.06)' }
      }
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth="md" sx={{ py: 6 }}>

        {/* HEADER */}
        <Box textAlign="center" mb={5}>
          <Typography variant="h3">Instrument Recognition</Typography>
          <Typography color="text.secondary">
            Explainable audio intelligence using spectral feature analysis
          </Typography>
        </Box>

        {/* UPLOAD */}
        <Card sx={{ p: 3, mb: 4, textAlign: 'center' }}>
          <input hidden ref={fileRef} type="file" accept=".wav,.mp3"
            onChange={e => setFile(e.target.files[0])} />
          <Box display="flex" justifyContent="center" gap={2} flexWrap="wrap">
            <Button startIcon={<FileUploadIcon />}
              onClick={() => fileRef.current.click()}>
              Select Audio
            </Button>
            <Button variant="contained" startIcon={<AnalyticsIcon />}
              disabled={!file || loading} onClick={analyze}>
              Analyze
            </Button>
            {result && (
              <Button variant="outlined" startIcon={<DescriptionIcon />}
                onClick={downloadReport}>
                Download Report
              </Button>
            )}
          </Box>
          <Typography color="text.secondary" mt={1}>
            {file?.name || 'Supported formats: WAV, MP3'}
          </Typography>
        </Card>

        {loading && <CircularProgress sx={{ display: 'block', mx: 'auto', my: 3 }} />}
        {error && <Alert severity="error">{error}</Alert>}

        {result && (
          <>
            {/* IDENTIFIED */}
            <Card sx={{ p: 4, mb: 4, textAlign: 'center' }}>
              <Typography variant="h4">Identified Instrument</Typography>
              <Typography variant="h2" color="primary.main" mt={2}>
                {result.instrument}
              </Typography>
            </Card>

            {/* PROBABILITY */}
            <Card sx={{ p: 3, mb: 4 }}>
              <Typography variant="h4" mb={2}>
                Other Instrument Probability
              </Typography>
              <Bar data={barData} />
            </Card>

            {/* WAVEFORM */}
            <Card sx={{ p: 3, mb: 4 }}>
              <Typography variant="h4" mb={2}>
                Audio Waveform
              </Typography>
              <Box sx={{ height: 220 }}>
                <Line data={waveformData} options={waveformOptions} />
              </Box>
              <Typography color="text.secondary" mt={1}>
                Raw time-domain signal representation of the input audio.
              </Typography>
            </Card>

            {/* RADAR */}
            <Card sx={{ p: 3, mb: 4 }}>
              <Typography variant="h4" mb={2}>
                Feature Fingerprint Comparison
              </Typography>
              <Radar data={radarData} />
            </Card>

            {/* TABLE */}
            <Card sx={{ p: 3 }}>
              <Typography variant="h4" mb={2}>
                Feature-Level Comparison
              </Typography>
              <TableContainer component={Paper}>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Feature</TableCell>
                      <TableCell>Description</TableCell>
                      <TableCell align="right">Input</TableCell>
                      <TableCell align="right">Database Avg</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {features.map((f, i) => (
                      <TableRow key={f.name}>
                        <TableCell>{f.name}</TableCell>
                        <TableCell>{f.desc}</TableCell>
                        <TableCell align="right" sx={{ color: 'primary.main' }}>
                          {result.feature_vector[i].toFixed(4)}
                        </TableCell>
                        <TableCell align="right" sx={{ color: '#f1b25b' }}>
                          {result.compared_vector[i].toFixed(4)}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Card>
          </>
        )}
      </Container>
    </ThemeProvider>
  );
}
