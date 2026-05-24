/**
 * API Service Layer
 * -----------------
 * Handles all communication with the Flask backend.
 * Separates network logic from UI components.
 */

import axios from "axios";
import { API_BASE_URL } from "../config/constants";

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // Audio processing can take time
});

/**
 * Upload an audio file for instrument classification.
 * @param {File} audioFile - The audio file to analyze
 * @returns {Promise<Object>} Classification results
 */
export async function analyzeAudio(audioFile) {
  const formData = new FormData();
  formData.append("audioFile", audioFile);

  const response = await apiClient.post("/analyze", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });

  return response.data;
}

/**
 * Check if the backend server is running and model is ready.
 * @returns {Promise<Object>} Health status
 */
export async function checkHealth() {
  const response = await apiClient.get("/health");
  return response.data;
}
