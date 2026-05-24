/**
 * Feature Table Component
 * ------------------------
 * Detailed numerical comparison of each spectral feature
 * between the input audio and the database average.
 */

import React from "react";
import {
  Card,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from "@mui/material";
import { FEATURE_LABELS } from "../config/constants";

export default function FeatureTable({ featureVector, comparedVector }) {
  return (
    <Card sx={{ p: 3 }}>
      <Typography variant="h6" mb={2}>
        Feature-Level Breakdown
      </Typography>
      <TableContainer
        component={Paper}
        sx={{ bgcolor: "background.default", borderRadius: 2 }}
      >
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell sx={{ fontWeight: 600 }}>Feature</TableCell>
              <TableCell sx={{ fontWeight: 600 }}>Description</TableCell>
              <TableCell align="right" sx={{ fontWeight: 600 }}>
                Input
              </TableCell>
              <TableCell align="right" sx={{ fontWeight: 600 }}>
                DB Average
              </TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {FEATURE_LABELS.map((feature, i) => (
              <TableRow key={feature.name} hover>
                <TableCell sx={{ fontWeight: 500 }}>{feature.name}</TableCell>
                <TableCell sx={{ color: "text.secondary" }}>
                  {feature.description}
                </TableCell>
                <TableCell align="right" sx={{ color: "primary.main", fontFamily: "monospace" }}>
                  {featureVector[i]?.toFixed(4)}
                </TableCell>
                <TableCell align="right" sx={{ color: "warning.main", fontFamily: "monospace" }}>
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
