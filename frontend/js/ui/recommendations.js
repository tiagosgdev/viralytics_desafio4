import { state, COLORS, EMOJI, NAME_TO_HEX } from '../state.js';
import { esc, escAttr, rgbToHex, toRenderableImageUrl, animateStagger, colorNameToHex } from '../utils.js';

export function formatAgentRec(item) {
  const typeName = (item.type || '').replace(/_/g, ' ');
  const brandName = item.brand || '';
  const scores = item.agent_scores || {};
  return {
    id: String(item.item_id || ''),
    name: [brandName, typeName].filter(Boolean).join(' ').trim() || `Item ${item.item_id}`,
    category: item.type || '',
    price: item.price != null ? `EUR ${Number(item.price).toFixed(2)}` : 'N/A',
    image_url: null,
    reason:
      `Agent rank ${item.rank} · body ${(scores.body || 0).toFixed(2)} · clothing ${(scores.clothing || 0).toFixed(2)} · colour ${(scores.colour || 0).toFixed(2)} · stock ${(scores.stock || 0).toFixed(2)}`,
    score: Math.round((1 - (item.rank - 1) / 10) * 100) / 100,
    brand: item.brand || null,
    metadata: Object.fromEntries(
      ['color', 'style', 'pattern', 'material', 'fit', 'season', 'occasion', 'gender', 'age_group', 'size']
        .filter((k) => item[k] != null)
        .map((k) => [k, item[k]])
    ),
  };
}

export function extractIncludeFilters(source) {
  if (!source || typeof source !== 'object') return {};
  if (source.include && typeof source.include === 'object') return source.include || {};
  const candidateKeys = Object.keys(source);
  const looksLikeFlatInclude = candidateKeys.some((key) => Array.isArray(source[key]));
  return looksLikeFlatInclude ? source : {};
}

export function getActiveIncludeFilters() {
  const directInclude = extractIncludeFilters(state.currentActiveFilters);
  if (Object.keys(directInclude).length) return directInclude;
  const stateFilters =
    state.chatConversationState && typeof state.chatConversationState === 'object'
      ? state.chatConversationState.filters
      : null;
  const stateInclude = extractIncludeFilters(stateFilters);
  if (Object.keys(stateInclude).length) return stateInclude;
  return {};
}

function normalizeComparableValue(value) {
  return String(value || '').trim().toLowerCase();
}

function attributeMatchesUserIntent(field, value, recommendation) {
  const include = getActiveIncludeFilters();
  const desired = include && Array.isArray(include[field]) ? include[field] : null;
  if (!desired || !desired.length) return false;
  const currentValue = normalizeComparableValue(value);
  const desiredValues = desired.map(normalizeComparableValue);
  if (field === 'type' && recommendation && recommendation.category) {
    return desiredValues.includes(normalizeComparableValue(recommendation.category));
  }
  return desiredValues.includes(currentValue);
}

function buildDetailRow(label, value, field, recommendation) {
  const highlight = attributeMatchesUserIntent(field, value, recommendation) ? ' match' : '';
  const badge = highlight ? '<em class="match-badge">match</em>' : '';
  return (
    '<div class="rec-detail-row' +
    highlight +
    '"><span>' +
    esc(label) +
    '</span>' +
    badge +
    '<strong>' +
    esc(value) +
    '</strong></div>'
  );
}

export function buildRecommendationThumb(rec, baseClass) {
  const icon = EMOJI[rec.category] || '<i class="fas fa-shirt"></i>';
  const rawImageUrl = rec && rec.image_url ? String(rec.image_url).trim() : '';
  const imageUrl = toRenderableImageUrl(rawImageUrl);
  const classes = imageUrl ? baseClass + ' thumb-media has-image' : baseClass + ' thumb-media';

  let swatchHex = '';
  if (rec && rec.color) {
    try {
      swatchHex = rgbToHex(rec.color);
    } catch (_) {
      swatchHex = '';
    }
  } else if (rec && rec.metadata && rec.metadata.color) {
    swatchHex = colorNameToHex(rec.metadata.color);
  }

  const swatchHtml = swatchHex
    ? '<div class="thumb-swatch" aria-hidden="true" style="background:' + swatchHex + '"></div>'
    : '';

  if (!imageUrl) {
    return '<div class="' + classes + '">' + swatchHtml + '<div class="thumb-fallback">' + icon + '</div></div>';
  }

  return (
    '<div class="' +
    classes +
    '">' +
    swatchHtml +
    '<img class="thumb-img" src="' +
    escAttr(imageUrl) +
    '" alt="' +
    escAttr(rec.name || 'Clothing item') +
    '" loading="lazy" onerror="this.parentNode.classList.remove(\'has-image\'); this.remove();">' +
    '<div class="thumb-fallback">' +
    icon +
    '</div>' +
    '</div>'
  );
}

export function renderRecs(recs, options = {}) {
  const autoOpen = Boolean(options.autoOpen);
  const recsEl = document.getElementById('recs');
  const recsScrollLeftBtn = document.getElementById('recs-scroll-left');
  const recsScrollRightBtn = document.getElementById('recs-scroll-right');
  const recsModalList = document.getElementById('rec-modal-list');
  const recDetail = document.getElementById('rec-detail');
  const recModalSubtitle = document.getElementById('rec-modal-subtitle');

  if (!recs.length) {
    state.currentRecommendations = [];
    recsEl.innerHTML = '<div class="ph">No matches - try retrying</div>';
    recsScrollLeftBtn.disabled = true;
    recsScrollRightBtn.disabled = true;
    recsModalList.innerHTML = '';
    recDetail.innerHTML = '<div class="ph">Recommendation details will appear here.</div>';
    recModalSubtitle.textContent = 'Select an item to inspect its details.';
    toggleRecommendationsModal(false);
    return;
  }

  state.currentRecommendations = recs;
  state.recommendationDetailIndex = 0;
  recsScrollLeftBtn.disabled = false;
  recsScrollRightBtn.disabled = false;

  recsEl.innerHTML = recs
    .map(
      (r, index) =>
        '<button class="card rec-card-button" type="button" onclick="openRecommendationDetail(' +
        index +
        ', true)">' +
        buildRecommendationThumb(r, 'cthumb') +
        '<div class="cinfo"><div class="cname">' +
        esc(r.name) +
        '</div>' +
        '<div class="ccat">' +
        (r.category || '').replace(/_/g, ' ') +
        '</div>' +
        '<div class="cwhy">' +
        esc(r.reason) +
        '</div>' +
        '<div class="cprice">' +
        esc(r.price) +
        '</div></div></button>'
    )
    .join('');

  recsEl.scrollTo({ left: 0, behavior: 'auto' });

  const cards = document.querySelectorAll('#recs .card');
  cards.forEach((card, index) => {
    card.style.animationDelay = index * 0.08 + 's';
    card.classList.add('rec-animate');
  });

  renderRecommendationsModal(recs);
  if (autoOpen) {
    state.hasShownInitialRecommendationsModal = true;
    toggleRecommendationsModal(true);
  }
}

export function renderRecommendationsModal(recs) {
  const recsModalList = document.getElementById('rec-modal-list');
  recsModalList.innerHTML = recs
    .map((rec, index) => {
      const active = index === state.recommendationDetailIndex ? ' active' : '';
      const category = (rec.category || 'item').replace(/_/g, ' ');
      return (
        '<button class="rec-modal-item' +
        active +
        '" type="button" onclick="openRecommendationDetail(' +
        index +
        ')">' +
        buildRecommendationThumb(rec, 'rec-modal-thumb') +
        '<div class="rec-modal-copy">' +
        '<div class="rec-modal-name">' +
        esc(rec.name) +
        '</div>' +
        '<div class="rec-modal-category">' +
        esc(category) +
        '</div>' +
        '<div class="rec-modal-price">' +
        esc(rec.price) +
        '</div>' +
        '</div></button>'
      );
    })
    .join('');

  openRecommendationDetail(state.recommendationDetailIndex);
}

export function openRecommendationDetail(index, openModal = false) {
  if (!state.currentRecommendations.length || !state.currentRecommendations[index]) return;
  const recDetail = document.getElementById('rec-detail');
  const recModalSubtitle = document.getElementById('rec-modal-subtitle');
  const recsModalList = document.getElementById('rec-modal-list');

  state.recommendationDetailIndex = index;
  const rec = state.currentRecommendations[index];
  const metadata = rec.metadata && typeof rec.metadata === 'object' ? rec.metadata : {};
  const sizes = Array.isArray(rec.sizes) ? rec.sizes.filter(Boolean) : [];
  const stockStatus = rec.stock_status ? String(rec.stock_status).replace(/_/g, ' ') : null;
  const detailRows = Object.entries(metadata)
    .map(([key, value]) => buildDetailRow(key.replace(/_/g, ' '), value, key, rec))
    .join('');
  const overviewRows = [
    buildDetailRow('type', (rec.category || 'item').replace(/_/g, ' '), 'type', rec),
    rec.brand ? buildDetailRow('brand', rec.brand, 'brand', rec) : '',
    rec.sku ? buildDetailRow('sku', rec.sku, 'sku', rec) : '',
    stockStatus ? buildDetailRow('stock', stockStatus, 'stock_status', rec) : '',
    sizes.length ? buildDetailRow('sizes', sizes.join(', '), 'sizes', rec) : '',
  ]
    .filter(Boolean)
    .join('');

  let colorBlock = '';
  let hex = '';
  let cname = '';
  let rgbText = '';
  if (rec && rec.color) {
    try {
      hex = rgbToHex(rec.color);
      rgbText = 'RGB(' + [rec.color[0], rec.color[1], rec.color[2]].join(',') + ')';
    } catch (_) {}
  }
  if (!hex && rec && rec.metadata && rec.metadata.color) {
    hex = colorNameToHex(rec.metadata.color);
  }
  if (rec && rec.color_name) cname = String(rec.color_name);
  if (!cname && rec && rec.metadata && rec.metadata.color) cname = String(rec.metadata.color);
  if (hex || cname) {
    colorBlock =
      '<div class="rec-color-palette"' +
      (cname ? ' title="' + escAttr(cname + (rgbText ? ' — ' + rgbText : '')) + '"' : '') +
      '>' +
      (hex ? '<div class="rec-color-swatch" style="background:' + hex + '" aria-hidden="true"></div>' : '') +
      '<div class="rec-color-meta">' +
      (cname ? '<div class="rec-color-name">' + esc(cname) + '</div>' : '') +
      (rgbText ? '<div class="rec-color-rgb">' + esc(rgbText) + '</div>' : '') +
      '</div></div>';
  }

  recDetail.innerHTML =
    '<div class="rec-detail-hero">' +
    buildRecommendationThumb(rec, 'rec-detail-icon') +
    '<div class="rec-detail-copy">' +
    '<h3>' + esc(rec.name) + '</h3>' +
    '<div class="rec-detail-category">' + esc((rec.category || 'item').replace(/_/g, ' ')) + '</div>' +
    '<div class="rec-detail-price">' + esc(rec.price) + '</div>' +
    colorBlock +
    '</div></div>' +
    (overviewRows
      ? '<div class="rec-detail-section"><div class="rec-detail-label">Store details</div><div class="rec-detail-grid">' + overviewRows + '</div></div>'
      : '') +
    '<div class="rec-detail-section"><div class="rec-detail-label">Why it fits</div><p>' + esc(rec.reason || 'Recommended from your search context.') + '</p></div>' +
    '<div class="rec-detail-section"><div class="rec-detail-label">Description</div><p>' + esc(rec.description || 'No extra product description is available for this item yet.') + '</p></div>' +
    '<div class="rec-detail-grid">' +
    (detailRows || '<div class="ph">No extra attributes available for this recommendation.</div>') +
    '</div>';

  recModalSubtitle.textContent = state.currentRecommendations.length + ' recommendation(s) ready to explore.';
  Array.from(recsModalList.querySelectorAll('.rec-modal-item')).forEach((item, itemIndex) => {
    item.classList.toggle('active', itemIndex === index);
  });

  if (openModal) toggleRecommendationsModal(true);
}

export function scrollRecommendations(direction) {
  const recsEl = document.getElementById('recs');
  const card = recsEl.querySelector('.rec-card-button');
  const step = card ? card.getBoundingClientRect().width + 14 : 280;
  const maxScroll = Math.max(0, recsEl.scrollWidth - recsEl.clientWidth);
  const current = recsEl.scrollLeft;
  let target = current + direction * step;
  if (direction > 0 && current >= maxScroll - 8) target = 0;
  else if (direction < 0 && current <= 8) target = maxScroll;
  else target = Math.max(0, Math.min(target, maxScroll));
  recsEl.scrollTo({ left: target, behavior: 'smooth' });
}

export function toggleRecommendationsModal(open) {
  const recsModal = document.getElementById('recommendations-modal');
  const shouldOpen = Boolean(open);
  if (shouldOpen && !state.currentRecommendations.length) return;
  recsModal.classList.toggle('open', shouldOpen);
  recsModal.setAttribute('aria-hidden', shouldOpen ? 'false' : 'true');
  document.body.classList.toggle('modal-open', shouldOpen);
}
