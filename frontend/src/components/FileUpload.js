import React, { useRef, useState } from "react";
import { Box, Card, Typography, Button, Fade, Grid } from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import DownloadIcon from "@mui/icons-material/Download";
import AudioFileIcon from "@mui/icons-material/AudioFile";
import MusicNoteIcon from "@mui/icons-material/MusicNote";
import { ACCEPTED_FORMATS } from "../config/constants";

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
    <Grid container spacing={3} sx={{ mb: 4 }}>
      {/* Upload area */}
      <Grid item xs={12} md={8}>
        <Card
          sx={{
            p: 5,
            height: "100%",
            minHeight: 220,
            textAlign: "center",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            border: dragOver
              ? "2px dashed #3b82f6"
              : "1px solid rgba(255,255,255,0.06)",
            bgcolor: dragOver ? "rgba(59,130,246,0.04)" : "#111111",
            cursor: "pointer",
            transition: "all 0.2s ease",
            "&:hover": {
              border: "1px solid rgba(59,130,246,0.3)",
              bgcolor: "rgba(59,130,246,0.02)",
            },
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

          <Box
            sx={{
              width: 64,
              height: 64,
              borderRadius: "50%",
              bgcolor: "rgba(59,130,246,0.08)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              mb: 2,
            }}
          >
            <CloudUploadIcon sx={{ fontSize: 28, color: "#3b82f6" }} />
          </Box>

          <Typography variant="body1" sx={{ color: "#d4d4d4", fontWeight: 500, mb: 0.5 }}>
            Drop audio file here or click to browse
          </Typography>
          <Typography variant="body2" sx={{ color: "#525252" }}>
            WAV, MP3, OGG, FLAC, M4A
          </Typography>

          {/* Selected file pill */}
          <Fade in={!!file}>
            <Box
              display="flex"
              alignItems="center"
              gap={1}
              mt={2.5}
              sx={{
                py: 1,
                px: 2.5,
                borderRadius: 2,
                bgcolor: "rgba(59,130,246,0.1)",
                border: "1px solid rgba(59,130,246,0.25)",
                display: file ? "flex" : "none",
              }}
            >
              <AudioFileIcon sx={{ color: "#3b82f6", fontSize: 18 }} />
              <Typography variant="body2" sx={{ color: "#60a5fa", fontWeight: 500 }}>
                {file?.name}
              </Typography>
            </Box>
          </Fade>
        </Card>
      </Grid>

      {/* Control panel */}
      <Grid item xs={12} md={4}>
        <Card
          sx={{
            p: 3,
            height: "100%",
            display: "flex",
            flexDirection: "column",
            justifyContent: "space-between",
            bgcolor: "#111111",
          }}
        >
          <Box>
            <Box display="flex" alignItems="center" gap={1} mb={2.5}>
              <MusicNoteIcon sx={{ color: "#3b82f6", fontSize: 20 }} />
              <Typography variant="body2" sx={{ color: "#f5f5f5", fontWeight: 600 }}>
                Analysis Settings
              </Typography>
            </Box>

            <Box mb={2}>
              <Typography variant="caption" sx={{ color: "#525252", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                Model
              </Typography>
              <Box
                sx={{
                  mt: 0.5, p: 1.5, borderRadius: 1.5,
                  bgcolor: "rgba(255,255,255,0.03)",
                  border: "1px solid rgba(255,255,255,0.06)",
                }}
              >
                <Typography variant="body2" sx={{ color: "#a3a3a3", fontFamily: "monospace" }}>
                  KNN (K=7, Cosine)
                </Typography>
              </Box>
            </Box>

            <Box mb={2}>
              <Typography variant="caption" sx={{ color: "#525252", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                Features
              </Typography>
              <Box
                sx={{
                  mt: 0.5, p: 1.5, borderRadius: 1.5,
                  bgcolor: "rgba(255,255,255,0.03)",
                  border: "1px solid rgba(255,255,255,0.06)",
                }}
              >
                <Typography variant="body2" sx={{ color: "#a3a3a3", fontFamily: "monospace" }}>
                  26 dimensions
                </Typography>
              </Box>
            </Box>
          </Box>

          <Box>
            <Button
              fullWidth
              variant="contained"
              startIcon={<PlayArrowIcon />}
              disabled={!file || loading}
              onClick={onAnalyze}
              sx={{ mb: 1.5, py: 1.3 }}
            >
              {loading ? "Analyzing..." : "Analyze"}
            </Button>

            {hasResult && (
              <Button
                fullWidth
                variant="outlined"
                startIcon={<DownloadIcon />}
                onClick={onDownload}
                sx={{ py: 1 }}
              >
                Download CSV Report
              </Button>
            )}
          </Box>
        </Card>
      </Grid>
    </Grid>
  );
}
