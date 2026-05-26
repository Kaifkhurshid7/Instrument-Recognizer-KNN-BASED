import React from "react";
import {
  Card,
  Typography,
  Box,
  LinearProgress,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from "@mui/material";

export default function ResultCard({ instrument, confidence, probabilities }) {
  const getConfidenceColor = (score) => {
    if (score >= 80) return "success";
    if (score >= 50) return "warning";
    return "error";
  };

  const sorted = probabilities
    ? [...probabilities].sort((a, b) => b.score - a.score)
    : [];

  return (
    <Box mb={4}>
      <Typography variant="h6" sx={{ color: "#10b981", mb: 1.5, fontSize: "0.75rem", letterSpacing: "0.08em" }}>
        OUTPUT
      </Typography>

      <Grid container spacing={3}>
        {/* Left: Primary result */}
        <Grid item xs={12} md={5}>
          <Card sx={{ p: 4, height: "100%" }}>
            <Typography
              variant="caption"
              sx={{ color: "#525252", textTransform: "uppercase", letterSpacing: "0.05em" }}
            >
              Identified Instrument
            </Typography>
            <Typography variant="h3" sx={{ color: "#f5f5f5", mt: 1, mb: 2, fontWeight: 700 }}>
              {instrument}
            </Typography>

            <Typography
              variant="caption"
              sx={{ color: "#525252", textTransform: "uppercase", letterSpacing: "0.05em" }}
            >
              Confidence Score
            </Typography>
            <Typography variant="h4" sx={{ color: "#3b82f6", mt: 0.5, fontWeight: 700 }}>
              {confidence}%
            </Typography>
            <LinearProgress
              variant="determinate"
              value={confidence}
              color={getConfidenceColor(confidence)}
              sx={{ mt: 1.5, height: 6, borderRadius: 3 }}
            />

            <Box mt={3}>
              <Typography variant="body2" sx={{ color: "#737373", lineHeight: 1.7 }}>
                The model classified this audio as <strong style={{ color: "#f5f5f5" }}>{instrument}</strong> with{" "}
                {confidence}% confidence based on cosine distance voting across K=3 nearest neighbors
                in the reference database.
              </Typography>
            </Box>
          </Card>
        </Grid>

        {/* Right: All instruments with percentages */}
        <Grid item xs={12} md={7}>
          <Card sx={{ p: 4, height: "100%" }}>
            <Typography variant="body1" sx={{ color: "#f5f5f5", fontWeight: 600, mb: 0.5 }}>
              All Identified Instruments
            </Typography>
            <Typography variant="body2" sx={{ color: "#737373", mb: 2 }}>
              Probability distribution across all 11 instrument classes (sorted by score)
            </Typography>

            <TableContainer
              sx={{
                borderRadius: 1,
                border: "1px solid rgba(255,255,255,0.06)",
                bgcolor: "#0a0a0a",
              }}
            >
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell sx={{ color: "#525252", fontWeight: 600, fontSize: "0.7rem", textTransform: "uppercase", bgcolor: "#0f0f0f", borderColor: "rgba(255,255,255,0.06)", py: 1 }}>
                      Rank
                    </TableCell>
                    <TableCell sx={{ color: "#525252", fontWeight: 600, fontSize: "0.7rem", textTransform: "uppercase", bgcolor: "#0f0f0f", borderColor: "rgba(255,255,255,0.06)", py: 1 }}>
                      Instrument
                    </TableCell>
                    <TableCell align="right" sx={{ color: "#525252", fontWeight: 600, fontSize: "0.7rem", textTransform: "uppercase", bgcolor: "#0f0f0f", borderColor: "rgba(255,255,255,0.06)", py: 1 }}>
                      Probability
                    </TableCell>
                    <TableCell sx={{ color: "#525252", fontWeight: 600, fontSize: "0.7rem", textTransform: "uppercase", bgcolor: "#0f0f0f", borderColor: "rgba(255,255,255,0.06)", py: 1, width: "30%" }}>
                      Distribution
                    </TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {sorted.map((item, idx) => (
                    <TableRow
                      key={item.name}
                      sx={{
                        "& td": { borderColor: "rgba(255,255,255,0.04)", py: 1 },
                        bgcolor: idx === 0 ? "rgba(59,130,246,0.04)" : "transparent",
                      }}
                    >
                      <TableCell sx={{ color: "#525252", fontSize: "0.8rem", fontFamily: "monospace" }}>
                        #{idx + 1}
                      </TableCell>
                      <TableCell sx={{ color: idx === 0 ? "#3b82f6" : "#d4d4d4", fontSize: "0.85rem", fontWeight: idx === 0 ? 600 : 400 }}>
                        {item.name}
                      </TableCell>
                      <TableCell align="right" sx={{ color: idx === 0 ? "#3b82f6" : "#a3a3a3", fontSize: "0.85rem", fontFamily: "monospace", fontWeight: idx === 0 ? 600 : 400 }}>
                        {item.score}%
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                          <Box sx={{ flexGrow: 1, height: 4, borderRadius: 2, bgcolor: "rgba(255,255,255,0.05)" }}>
                            <Box
                              sx={{
                                height: "100%",
                                borderRadius: 2,
                                width: `${item.score}%`,
                                bgcolor: idx === 0 ? "#3b82f6" : "rgba(255,255,255,0.15)",
                                transition: "width 0.3s ease",
                              }}
                            />
                          </Box>
                        </Box>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}
