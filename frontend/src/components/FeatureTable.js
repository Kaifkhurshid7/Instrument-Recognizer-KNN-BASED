import React from "react";
import {
  Card,
  Typography,
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from "@mui/material";
import { FEATURE_LABELS } from "../config/constants";

export default function FeatureTable({ featureVector, comparedVector }) {
  return (
    <Card sx={{ p: 3 }}>
      <Box mb={2}>
        <Typography variant="h5" sx={{ color: "#f5f5f5" }}>
          Feature Breakdown
        </Typography>
        <Typography variant="body2" sx={{ mt: 0.5 }}>
          All 26 spectral dimensions extracted from the audio
        </Typography>
      </Box>

      <TableContainer
        sx={{
          borderRadius: 2,
          border: "1px solid rgba(255,255,255,0.06)",
          bgcolor: "#0a0a0a",
        }}
      >
        <Table size="small">
          <TableHead>
            <TableRow sx={{ "& th": { borderColor: "rgba(255,255,255,0.06)" } }}>
              <TableCell sx={{ color: "#737373", fontWeight: 600, fontSize: "0.75rem" }}>
                FEATURE
              </TableCell>
              <TableCell sx={{ color: "#737373", fontWeight: 600, fontSize: "0.75rem" }}>
                DESCRIPTION
              </TableCell>
              <TableCell align="right" sx={{ color: "#3b82f6", fontWeight: 600, fontSize: "0.75rem" }}>
                INPUT
              </TableCell>
              <TableCell align="right" sx={{ color: "#f59e0b", fontWeight: 600, fontSize: "0.75rem" }}>
                DB AVERAGE
              </TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {FEATURE_LABELS.map((feature, i) => (
              <TableRow
                key={feature.name}
                sx={{
                  "& td": { borderColor: "rgba(255,255,255,0.04)" },
                  "&:hover": { bgcolor: "rgba(255,255,255,0.02)" },
                }}
              >
                <TableCell sx={{ color: "#e5e5e5", fontWeight: 500, fontSize: "0.8rem" }}>
                  {feature.name}
                </TableCell>
                <TableCell sx={{ color: "#737373", fontSize: "0.8rem" }}>
                  {feature.description}
                </TableCell>
                <TableCell align="right" sx={{ color: "#3b82f6", fontFamily: "monospace", fontSize: "0.8rem" }}>
                  {featureVector[i]?.toFixed(4)}
                </TableCell>
                <TableCell align="right" sx={{ color: "#f59e0b", fontFamily: "monospace", fontSize: "0.8rem" }}>
                  {comparedVector[i]?.toFixed(4)}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Card>
  );
}
