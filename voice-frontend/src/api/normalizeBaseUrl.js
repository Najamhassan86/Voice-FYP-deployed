export const DEPLOYED_API_BASE_URL = 'https://voice-fyp-deployed-production.up.railway.app';

export const normalizeBaseURL = (value) => {
  const raw = (value || '').trim();
  if (!raw) return '';

  const withProtocol = /^https?:\/\//i.test(raw) ? raw : `https://${raw}`;
  return withProtocol.replace(/\/+$/, '');
};

export const isLocalFrontend = () => {
  const hostname = globalThis?.location?.hostname || '';
  return hostname === 'localhost' || hostname === '127.0.0.1';
};

export const getDefaultApiBaseURL = () => (
  isLocalFrontend() ? 'http://localhost:8000' : DEPLOYED_API_BASE_URL
);
