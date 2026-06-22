import { NAME_TO_HEX } from './state.js';

export function esc(value) {
  return String(value || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

export function escAttr(value) {
  return String(value || '')
    .replace(/&/g, '&amp;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

export function rgbToHex(rgb) {
  if (!rgb || !Array.isArray(rgb) || rgb.length < 3) return '#888';
  const r = Number(rgb[0]);
  const g = Number(rgb[1]);
  const b = Number(rgb[2]);
  if ([r, g, b].some((v) => Number.isNaN(v))) return '#888';
  return '#' + [r, g, b].map((v) => v.toString(16).padStart(2, '0')).join('');
}

export function extractGoogleDriveFileId(rawUrl) {
  const value = String(rawUrl || '').trim();
  if (!value) return '';
  try {
    const parsed = new URL(value);
    const host = parsed.hostname.toLowerCase();
    if (host !== 'drive.google.com' && host !== 'drive.usercontent.google.com') return '';
    const parts = parsed.pathname.split('/').filter(Boolean);
    if (parts.length >= 3 && parts[0] === 'file' && parts[1] === 'd' && parts[2]) return parts[2];
    return parsed.searchParams.get('id') || '';
  } catch (_) {
    return '';
  }
}

export function toRenderableImageUrl(rawUrl) {
  const original = String(rawUrl || '').trim();
  if (!original) return '';
  const fileId = extractGoogleDriveFileId(original);
  if (!fileId) return original;
  const normalizedDriveUrl =
    'https://drive.usercontent.google.com/download?id=' +
    encodeURIComponent(fileId) +
    '&export=view';
  return '/api/image-proxy?url=' + encodeURIComponent(normalizedDriveUrl);
}

export function extractApiError(data, fallback) {
  if (!data) return fallback;
  if (typeof data.detail === 'string' && data.detail.trim()) return data.detail;
  if (typeof data.message === 'string' && data.message.trim()) return data.message;
  if (typeof data.reply === 'string' && data.reply.trim()) return data.reply;
  if (Array.isArray(data.detail)) {
    const parts = data.detail
      .map((entry) => {
        if (typeof entry === 'string') return entry;
        if (entry && typeof entry.msg === 'string') return entry.msg;
        return '';
      })
      .filter(Boolean);
    if (parts.length) return parts.join(' ');
  }
  return fallback;
}

export function animateStagger(containerSelector, childSelector, opts = { delay: 80, start: 0 }) {
  const container = document.querySelector(containerSelector);
  if (!container) return;
  if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const children = container.querySelectorAll(childSelector);
  children.forEach((child, index) => {
    child.classList.remove('stagger');
    child.style.animationDelay = (opts.start + index * opts.delay) + 'ms';
    void child.offsetWidth;
    child.classList.add('stagger');
  });
}

export function colorNameToHex(name) {
  return NAME_TO_HEX[String(name || '').toLowerCase()] || '';
}
