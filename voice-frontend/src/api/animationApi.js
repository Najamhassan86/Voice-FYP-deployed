import axios from 'axios';
import { normalizeBaseURL } from './normalizeBaseUrl';

// Get base URL from environment variable or use default
const API_BASE_URL = normalizeBaseURL(import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000');

const LOCAL_ANIMATION_IDS = new Set([
  'alert', 'book', 'careful', 'cheap', 'crazy', 'dangerous', 'decent', 'dumb',
  'excited', 'extreme', 'fantastic', 'far', 'fearful', 'foreign', 'funny', 'good',
  'healthy', 'heavy', 'important', 'intelligent', 'interesting', 'late', 'less',
  'new', 'no', 'noisy', 'peaceful', 'quick', 'ready', 'secure', 'smart', 'yes'
]);

const URDU_FALLBACK_MAP = {
  'خوش': 'excited',
  'اچھا': 'good',
  'ذہین': 'smart',
  'ہاں': 'yes',
  'کتاب': 'book',
  'تیار': 'ready',
  'تیز': 'quick',
  'اہم': 'important',
  'بہترین': 'fantastic',
  'تندرست': 'healthy',
  'پرامن': 'peaceful',
  'نیا': 'new'
};

const buildLocalAnimation = (id) => ({
  id,
  name: id.charAt(0).toUpperCase() + id.slice(1),
  description: `Local fallback animation for ${id}`,
  file_path: `/animations/${id}.glb`,
  category: 'fallback',
  tags: [id]
});

const resolveLocalAnimationFallback = (phrase, language = 'psl') => {
  const normalizedPhrase = (phrase || '').trim().toLowerCase();
  if (!normalizedPhrase) return null;

  if (language === 'ur') {
    const translated = URDU_FALLBACK_MAP[phrase.trim()];
    if (translated && LOCAL_ANIMATION_IDS.has(translated)) {
      return {
        animation: buildLocalAnimation(translated),
        matched_word: phrase.trim(),
        confidence: 0.5
      };
    }
  }

  const words = normalizedPhrase.split(/\s+/).filter(Boolean);
  const matched = words.find((word) => LOCAL_ANIMATION_IDS.has(word));

  if (!matched) return null;

  return {
    animation: buildLocalAnimation(matched),
    matched_word: matched,
    confidence: 0.5
  };
};

// Create axios instance with base configuration
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 10000, // 10 second timeout
});

/**
 * Health check - verify backend is running
 * @returns {Promise<Object>} Health status object
 */
export const checkHealth = async () => {
  try {
    const response = await apiClient.get('/health');
    return response.data;
  } catch (error) {
    console.error('Health check failed:', error);
    throw error;
  }
};

/**
 * Get all available animations
 * @returns {Promise<Array>} List of animation objects with metadata
 */
export const getAnimations = async () => {
  try {
    const response = await apiClient.get('/api/animations');
    return response.data;
  } catch (error) {
    console.error('Failed to fetch animations:', error);
    throw error;
  }
};

/**
 * Get a specific animation by ID
 * @param {string} animationId - The animation ID (e.g., 'hello', 'thanks')
 * @returns {Promise<Object>} Animation object with metadata
 */
export const getAnimationById = async (animationId) => {
  try {
    const response = await apiClient.get(`/api/animations/${animationId}`);
    return response.data;
  } catch (error) {
    console.error(`Failed to fetch animation ${animationId}:`, error);
    throw error;
  }
};

/**
 * Resolve the best animation for a given phrase
 * @param {string} phrase - The text phrase to translate
 * @param {string} language - The language code (default: 'psl')
 * @returns {Promise<Object>} Object with animation, matched_word, and confidence
 */
export const resolveAnimationForPhrase = async (phrase, language = 'psl') => {
  try {
    const response = await apiClient.post('/api/resolve-animation', {
      phrase,
      language,
    });
    return response.data;
  } catch (error) {
    console.error(`Failed to resolve animation for phrase "${phrase}":`, error);
    const fallback = resolveLocalAnimationFallback(phrase, language);
    if (fallback) {
      console.warn(`Using local animation fallback for phrase "${phrase}"`);
      return fallback;
    }
    throw error;
  }
};

/**
 * Get animations by category
 * @param {string} category - The category to filter by
 * @returns {Promise<Array>} Filtered list of animations
 */
export const getAnimationsByCategory = async (category) => {
  try {
    const animations = await getAnimations();
    return animations.filter(anim => anim.category === category);
  } catch (error) {
    console.error(`Failed to fetch animations by category ${category}:`, error);
    throw error;
  }
};

/**
 * Search animations by tag
 * @param {string} tag - The tag to search for
 * @returns {Promise<Array>} Animations matching the tag
 */
export const searchAnimationsByTag = async (tag) => {
  try {
    const animations = await getAnimations();
    return animations.filter(anim =>
      anim.tags.some(t => t.toLowerCase().includes(tag.toLowerCase()))
    );
  } catch (error) {
    console.error(`Failed to search animations by tag ${tag}:`, error);
    throw error;
  }
};

/**
 * Get Urdu word mappings for animations
 * @returns {Promise<Object>} Dictionary of English to Urdu mappings
 */
export const getUrduWords = async () => {
  try {
    const response = await apiClient.get('/api/urdu-words');
    return response.data;
  } catch (error) {
    console.error('Failed to fetch Urdu words:', error);
    throw error;
  }
};

export default {
  checkHealth,
  getAnimations,
  getAnimationById,
  resolveAnimationForPhrase,
  getAnimationsByCategory,
  searchAnimationsByTag,
  getUrduWords,
};
