import React, { useRef, useState } from "react";
import {
  Box,
  Card,
  Typography,
  Button,
  Fade,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import DownloadIcon from "@mui/icons-material/Download";
import AudioFileIcon from "@mui/icons-material/AudioFile";
import MusicNoteIcon from "@mui/icons-material/MusicNote";
import { ACCEPTED_FORMATS } from "../config/constants";

const instruments = [
  { name: "Acoustic Guitar", samples: 250 },
  { name: "Cello", samples: 388 },
  { name: "Clarinet", samples: 269 },
  { name: "Electric Guitar", samples: 308 },
  { name: "Flute", samples: 447 },
  { name: "Human Voice", samples: 250 },
  { name: "Organ", samples: 356 },
  { name: "Piano", samples: 250 },
  { name: "Saxophone", samples: 626 },
  { name: "Trumpet", samples: 250 },
  { name: "Violin", samples: 330 },
];

export default function FileUpload({
  file,
  onFileSelect,
  onAnalyze,
  onDownload,
  loading,
  hasResult,
}) {
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) onFileSelect(droppedFile);
  };

  return (
    <Box mb={5}>
      <Typography variant="h6" sx={{ color: "#3b82f6", mb: 1.5, fontSize: "0.75rem" }}>
        ANALYZE
      </Typography>

      <Grid container spacing={3}>
        {/* Left: Upload area */}
        <Grid item xs={12} md={5}>
          <Card
            sx={{
              p: 4,
              textAlign: "center",
              border: dragOver
                ? "1px dashed #3b82f6"
                : "1px solid rgba(255,255,255,0.06)",
              bgcolor: dragOver ? "rgba(59,130,246,0.03)" : "#141414",
              cursor: "pointer",
              transition: "all 0.2s ease",
              "&:hover": { border: "1px solid rgba(255,255,255,0.12)" },
              height: "100%",
              display: "flex",
              flexDirection: "column",
              justifyContent: "center",
              minHeight: 280,
            }}
            onClick={() => inputRef.current.click()}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
          >
            <input
              hidden
              ref={inputRef}
              type="file"
              accept={ACCEPTED_FORMATS}
              onChange={(e) => onFileSelect(e.target.files[0])}
            />

            {/* Upload icon with glow */}
            <Box
              sx={{
                width: 64,
                height: 64,
                borderRadius: "50%",
                bgcolor: "rgba(59,130,246,0.1)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                mx: "auto",
                mb: 2,
                boxShadow: "0 0 30px rgba(59,130,246,0.15)",
              }}
            >
              <CloudUploadIcon sx={{ fontSize: 28, color: "#3b82f6" }} />
            </Box>

            <Typography variant="body1" sx={{ color: "#a3a3a3", mb: 0.5 }}>
              Drop audio file here or click to browse
            </Typography>
            <Typography variant="caption">
              WAV, MP3, OGG, FLAC, M4A supported
            </Typography>

            <Fade in={!!file}>
              <Box
                display="flex"
                alignItems="center"
                justifyContent="center"
                gap={1}
                mt={2}
                sx={{
                  py: 1,
                  px: 2,
                  borderRadius: 1,
                  bgcolor: "rgba(59,130,246,0.08)",
                  border: "1px solid rgba(59,130,246,0.2)",
                  display: file ? "flex" : "none",
                }}
              >
                <AudioFileIcon sx={{ color: "#3b82f6", fontSize: 18 }} />
                <Typography variant="body2" sx={{ color: "#3b82f6", fontWeight: 500 }}>
                  {file?.name}
                </Typography>
              </Box>
            </Fade>
          </Card>
        </Grid>

        {/* Middle: Settings + Actions */}
        <Grid item xs={12} md={3}>
          <Card sx={{ p: 3, height: "100%" }}>
            <Box display="flex" alignItems="center" gap={1} mb={2}>
              <MusicNoteIcon sx={{ color: "#3b82f6", fontSize: 20 }} />
              <Typography variant="body2" sx={{ color: "#f5f5f5", fontWeight: 600 }}>
                Analysis Settings
              </Typography>
            </Box>

            <Box mb={2}>
              <Typography
                variant="caption"
                sx={{ color: "#737373", textTransform: "uppercase", letterSpacing: "0.05em" }}
              >
                Model
              </Typography>
              <Box
                sx={{
                  mt: 0.5, p: 1.5, borderRadius: 1,
                  bgcolor: "rgba(255,255,255,0.03)",
                  border: "1px solid rgba(255,255,255,0.08)",
                }}
              >
                <Typography variant="body2" sx={{ color: "#a3a3a3", fontFamily: "monospace" }}>
                  KNN (K=3, Cosine)
                </Typography>
              </Box>
            </Box>

            <Box mb={2}>
              <Typography
                variant="caption"
                sx={{ color: "#737373", textTransform: "uppercase", letterSpacing: "0.05em" }}
              >
                Features
              </Typography>
              <Box
                sx={{
                  mt: 0.5, p: 1.5, borderRadius: 1,
                  bgcolor: "rgba(255,255,255,0.03)",
                  border: "1px solid rgba(255,255,255,0.08)",
                }}
              >
                <Typography variant="body2" sx={{ color: "#a3a3a3", fontFamily: "monospace" }}>
                  26 dimensions
                </Typography>
              </Box>
            </Box>

            <Box mb={2}>
              <Typography
                variant="caption"
                sx={{ color: "#737373", textTransform: "uppercase", letterSpacing: "0.05em" }}
              >
                Accuracy
              </Typography>
              <Box
                sx={{
                  mt: 0.5, p: 1.5, borderRadius: 1,
                  bgcolor: "rgba(255,255,255,0.03)",
                  border: "1px solid rgba(255,255,255,0.08)",
                }}
              >
                <Typography variant="body2" sx={{ color: "#a3a3a3", fontFamily: "monospace" }}>
                  61.9% (5-fold CV)
                </Typography>
              </Box>
            </Box>

            <Button
              fullWidth
              variant="contained"
              startIcon={<PlayArrowIcon />}
              disabled={!file || loading}
              onClick={onAnalyze}
              sx={{ mb: 1.5 }}
            >
              {loading ? "Processing..." : "Analyze"}
            </Button>

            {hasResult && (
              <Button
                fullWidth
                variant="outlined"
                startIcon={<DownloadIcon />}
                onClick={onDownload}
                size="small"
              >
                Download Report
              </Button>
            )}
          </Card>
        </Grid>

        {/* Right: Supported instruments table */}
        <Grid item xs={12} md={4}>
          <Card sx={{ p: 3, height: "100%" }}>
            <Typography variant="body2" sx={{ color: "#f5f5f5", fontWeight: 600, mb: 2 }}>
              Supported Instruments
            </Typography>

            <TableContainer
              sx={{
                borderRadius: 1,
                border: "1px solid rgba(255,255,255,0.06)",
                bgcolor: "#0a0a0a",
                maxHeight: 320,
                overflow: "auto",
              }}
            >
              <Table size="small" stickyHeader>
                <TableHead>
                  <TableRow>
                    <TableCell
                      sx={{
                        color: "#525252",
                        fontWeight: 600,
                        fontSize: "0.7rem",
                        textTransform: "uppercase",
                        bgcolor: "#0f0f0f",
                        borderColor: "rgba(255,255,255,0.06)",
                        py: 1,
                      }}
                    >
                      Instrument
                    </TableCell>
                    <TableCell
                      align="right"
                      sx={{
                        color: "#525252",
                        fontWeight: 600,
                        fontSize: "0.7rem",
                        textTransform: "uppercase",
                        bgcolor: "#0f0f0f",
                        borderColor: "rgba(255,255,255,0.06)",
                        py: 1,
                      }}
                    >
                      Samples
                    </TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {instruments.map((inst) => (
                    <TableRow
                      key={inst.name}
                      sx={{
                        "& td": { borderColor: "rgba(255,255,255,0.04)", py: 0.8 },
                        "&:hover": { bgcolor: "rgba(255,255,255,0.02)" },
                      }}
                    >
                      <TableCell sx={{ color: "#d4d4d4", fontSize: "0.8rem" }}>
                        {inst.name}
                      </TableCell>
                      <TableCell
                        align="right"
                        sx={{ color: "#737373", fontSize: "0.8rem", fontFamily: "monospace" }}
                      >
                        {inst.samples}
                      </TableCell>
                    </TableRow>
                  ))}
                  {/* Total row */}
                  <TableRow sx={{ "& td": { borderColor: "rgba(255,255,255,0.06)", py: 0.8 } }}>
                    <TableCell sx={{ color: "#f5f5f5", fontSize: "0.8rem", fontWeight: 600 }}>
                      Total
                    </TableCell>
                    <TableCell
                      align="right"
                      sx={{ color: "#3b82f6", fontSize: "0.8rem", fontWeight: 600, fontFamily: "monospace" }}
                    >
                      3724
                    </TableCell>
                  </TableRow>
                </TableBody>
              </Table>
            </TableContainer>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}
