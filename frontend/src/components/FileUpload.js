/**
 * File Upload Component
 * ---------------------
 * Drag-and-drop or click-to-select audio file upload with
 * action buttons for analysis and report download.
 */

import React, { useRef, useState } from "react";
import { Box, Card, Typography, Button, Fade } from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import AnalyticsIcon from "@mui/icons-material/Analytics";
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
    <Card
      sx={{
        p: 4,
        mb: 4,
        textAlign: "center",
        border: dragOver ? "2px dashed" : undefined,
        borderColor: dragOver ? "primary.main" : undefined,
        transition: "border 0.2s ease",
      }}
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
    >
      {/* Hidden file input */}
      <input
        hidden
        ref={inputRef}
        type="file"
        accept={ACCEPTED_FORMATS}
        onChange={(e) => onFileSelect(e.target.files[0])}
      />

      {/* Drop zone prompt */}
      <Box mb={3}>
        <CloudUploadIcon sx={{ fontSize: 48, color: "text.secondary", mb: 1 }} />
        <Typography variant="body1" color="text.secondary">
          Drag & drop an audio file here, or click to browse
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Supports WAV, MP3, OGG, FLAC, M4A
        </Typography>
      </Box>

      {/* Selected file indicator */}
      <Fade in={!!file}>
        <Box
          display="flex"
          alignItems="center"
          justifyContent="center"
          gap={1}
          mb={2}
          sx={{
            py: 1,
            px: 2,
            borderRadius: 2,
            bgcolor: "rgba(110, 231, 183, 0.08)",
            display: file ? "flex" : "none",
          }}
        >
          <AudioFileIcon sx={{ color: "primary.main", fontSize: 20 }} />
          <Typography variant="body2" color="primary.main" fontWeight={500}>
            {file?.name}
          </Typography>
        </Box>
      </Fade>

      {/* Action buttons */}
      <Box display="flex" justifyContent="center" gap={2} flexWrap="wrap">
        <Button
          variant="outlined"
          startIcon={<CloudUploadIcon />}
          onClick={() => inputRef.current.click()}
        >
          Select File
        </Button>
        <Button
          variant="contained"
          startIcon={<AnalyticsIcon />}
          disabled={!file || loading}
          onClick={onAnalyze}
        >
          {loading ? "Analyzing..." : "Analyze"}
        </Button>
        {hasResult && (
          <Button
            variant="outlined"
            color="secondary"
            startIcon={<DownloadIcon />}
            onClick={onDownload}
          >
            CSV Report
          </Button>
        )}
      </Box>
    </Card>
  );
}
