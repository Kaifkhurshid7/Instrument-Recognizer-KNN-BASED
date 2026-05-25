import axios from "axios";
import { API_BASE_URL } from "../config/constants";

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000,
});

/**
 * Upload an audio file for instrument classification.
 * @param {File} audioFile
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
 * Check backend health status.
 * @returns {Promise<Object>}
 */
export async function checkHealth() {
  const response = await apiClient.get("/health");
  return response.data;
}
