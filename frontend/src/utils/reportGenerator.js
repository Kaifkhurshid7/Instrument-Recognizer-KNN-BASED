/**
 * Report Generator Utility
 * -------------------------
 * Creates downloadable CSV reports from analysis results.
 */

import { FEATURE_LABELS } from "../config/constants";

/**
 * Generate and trigger download of a CSV analysis report.
 * @param {Object} result - The analysis result from the API
 */
export function downloadCSVReport(result) {
  if (!result) return;

  const lines = [
    "=== Instrument Recognition Report ===",
    "",
    `Identified Instrument,${result.instrument}`,
    `Confidence Score,${result.confidence_score}%`,
    "",
    "--- Probability Distribution ---",
    "Instrument,Probability (%)",
  ];

  result.knn_probabilities.forEach((p) => {
    lines.push(`${p.name},${p.score}`);
  });

  lines.push("");
  lines.push("--- Feature Comparison ---");
  lines.push("Feature,Description,Input Value,Database Average");

  FEATURE_LABELS.forEach((feature, i) => {
    const inputVal = result.feature_vector[i]?.toFixed(6) || "N/A";
    const dbVal = result.compared_vector[i]?.toFixed(6) || "N/A";
    lines.push(`${feature.name},${feature.description},${inputVal},${dbVal}`);
  });

  const csv = lines.join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);

  const link = document.createElement("a");
  link.href = url;
  link.download = `instrument_report_${Date.now()}.csv`;
  link.click();

  URL.revokeObjectURL(url);
}
