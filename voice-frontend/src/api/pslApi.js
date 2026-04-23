/**
 * PSL Recognition API Service
 *
 * Handles communication with the backend PSL recognition endpoints.
 */

import axios from 'axios';

const createClient = (baseURL) => axios.create({
  baseURL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 second timeout for model inference
});

const ENV_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').trim();
const BASE_URL_CANDIDATES = [
  ENV_BASE_URL,
  'http://127.0.0.1:8001',
  'http://localhost:8001',
  'http://127.0.0.1:8000',
  'http://localhost:8000'
].filter((url, index, arr) => url && arr.indexOf(url) === index);

let activeBaseURL = ENV_BASE_URL || null;
const clientCache = new Map();

const getClient = (baseURL) => {
  if (!clientCache.has(baseURL)) {
    clientCache.set(baseURL, createClient(baseURL));
  }
  return clientCache.get(baseURL);
};

const getOrderedBaseUrls = () => {
  if (activeBaseURL) {
    return [activeBaseURL, ...BASE_URL_CANDIDATES.filter((url) => url !== activeBaseURL)];
  }
  return BASE_URL_CANDIDATES;
};

const requestWithAutoBase = async (method, path, data = null) => {
  const errors = [];

  for (const baseURL of getOrderedBaseUrls()) {
    try {
      const client = getClient(baseURL);
      const response = await client.request({
        method,
        url: path,
        data
      });
      activeBaseURL = baseURL;
      return response.data;
    } catch (error) {
      const message = error?.response?.data?.detail || error.message || 'Unknown error';
      errors.push(`${baseURL} -> ${message}`);
    }
  }

  throw new Error(`Could not reach PSL backend. Tried: ${errors.join(' | ')}`);
};

/**
 * Recognize PSL sign from a sequence of landmark features
 *
 * @param {number[][]} sequence - Array of 60 frames, each with 188 features
 * @returns {Promise<{label: string, class_id: number, confidence: number, top_predictions: Array}>}
 * @throws {Error} If the request fails or response is invalid
 */
export const recognizePSL = async (sequence) => {
  try {
    console.log('Sending PSL recognition request...', {
      sequenceLength: sequence.length,
      featureLength: sequence[0]?.length
    });

    const response = await requestWithAutoBase('post', '/api/psl/recognize', {
      sequence
    });

    console.log('PSL recognition response:', response);
    return response;

  } catch (error) {
    console.error('PSL recognition error:', error);

    if (error.response) {
      // Server responded with error status
      const errorMessage = error.response.data?.detail || 'Recognition failed';
      throw new Error(`Server error: ${errorMessage}`);
    } else if (error.request) {
      // Request was made but no response received
      throw new Error('No response from server. Is the backend running?');
    } else {
      // Something else happened
      throw new Error(`Request error: ${error.message}`);
    }
  }
};

/**
 * Get PSL model information
 *
 * @returns {Promise<Object>} Model metadata
 */
export const getPSLModelInfo = async () => {
  try {
    return await requestWithAutoBase('get', '/api/psl/model-info');
  } catch (error) {
    console.error('Error fetching PSL model info:', error);
    throw error;
  }
};

/**
 * Check PSL service health
 *
 * @returns {Promise<Object>} Health status
 */
export const checkPSLHealth = async () => {
  try {
    return await requestWithAutoBase('get', '/api/psl/health');
  } catch (error) {
    console.error('Error checking PSL health:', error);
    throw error;
  }
};

export const getActivePSLBaseURL = () => activeBaseURL;

/**
 * Validate a sequence before sending to the backend
 *
 * @param {number[][]} sequence - Sequence to validate
 * @returns {{valid: boolean, error: string|null}}
 */
export const validateSequence = (sequence) => {
  if (!Array.isArray(sequence)) {
    return { valid: false, error: 'Sequence must be an array' };
  }

  if (sequence.length !== 60) {
    return { valid: false, error: `Sequence must have 60 frames, got ${sequence.length}` };
  }

  for (let i = 0; i < sequence.length; i++) {
    const frame = sequence[i];

    if (!Array.isArray(frame)) {
      return { valid: false, error: `Frame ${i} must be an array` };
    }

    if (frame.length !== 188) {
      return { valid: false, error: `Frame ${i} must have 188 features, got ${frame.length}` };
    }

    // Check for NaN or Infinity
    for (let j = 0; j < frame.length; j++) {
      if (typeof frame[j] !== 'number' || !isFinite(frame[j])) {
        return { valid: false, error: `Frame ${i}, feature ${j} is not a valid number` };
      }
    }
  }

  return { valid: true, error: null };
};

export default {
  recognizePSL,
  getPSLModelInfo,
  checkPSLHealth,
  validateSequence,
  getActivePSLBaseURL
};
