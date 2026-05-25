/**
 * File Upload Component
 * ---------------------
 * Documentation-style input section with clean layout.
 * Shows settings panel on the right similar to Music.AI's UI.
 */

import React, { useRef, useState } from "react";
import { Box, Card, Typography, Button, Fade, Grid } from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import DownloadIcon from "@mui/icons-material/Download";
import AudioFileIcon from "@mui/icons-material/AudioFile";
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
    <Box mb={5}>
      {/* Section label */}
      <Typography variant="h6" sx={{ color: "#3b82f6", mb: 1.5, fontSize: "0.75rem" }}>
        INPUT
      </Typography>

      <Grid container spacing={3}>
        {/* Left: Upload area */}
        <Grid item xs={12} md={8}>
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
              "&:hover": {
                border: "1px solid rgba(255,255,255,0.12)",
              },
            }}
            onClick={() => inputRef.current.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
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

            <CloudUploadIcon
              sx={{ fontSize: 40, color: "#525252", mb: 2 }}
            />

            <Typography variant="body1" sx={{ color: "#a3a3a3", mb: 0.5 }}>
              Drop audio file here or click to browse
            </Typography>
            <Typography variant="caption">
              WAV, MP3, OGG, FLAC, M4A supported
            </Typography>

            {/* Selected file */}
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
                <Typography
                  variant="body2"
                  sx={{ color: "#3b82f6", fontWeight: 500 }}
                >
                  {file?.name}
                </Typography>
              </Box>
            </Fade>
          </Card>
        </Grid>

        {/* Right: Settings panel */}
        <Grid item xs={12} md={4}>
          <Card sx={{ p: 3, height: "100%" }}>
            <Typography
              variant="body2"
              sx={{ color: "#f5f5f5", fontWeight: 600, mb: 2 }}
            >
              Instruments detection
            </Typography>

            <Box mb={2}>
              <Typography variant="caption" sx={{ color: "#737373" }}>
                Model
              </Typography>
              <Box
                sx={{
                  mt: 0.5,
                  p: 1.5,
                  borderRadius: 1,
                  bgcolor: "rgba(255,255,255,0.03)",
                  border: "1px solid rgba(255,255,255,0.08)",
                }}
              >
                <Typography variant="body2" sx={{ color: "#a3a3a3" }}>
                  KNN (K=7, Cosine)
                </Typography>
              </Box>
            </Box>

            <Box mb={3}>
              <Typography variant="caption" sx={{ color: "#737373" }}>
                Feature dimensions
              </Typography>
              <Box
                sx={{
                  mt: 0.5,
                  p: 1.5,
                  borderRadius: 1,
                  bgcolor: "rgba(255,255,255,0.03)",
                  border: "1px solid rgba(255,255,255,0.08)",
                }}
              >
                <Typography variant="body2" sx={{ color: "#a3a3a3" }}>
                  26
                </Typography>
              </Box>
            </Box>

            {/* Action buttons */}
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
      </Grid>
    </Box>
  );
}
