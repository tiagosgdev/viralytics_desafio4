import { state } from './state.js';

export function buildWsUrl() {
  const protocol = location.protocol === 'https:' ? 'wss://' : 'ws://';
  const persona = encodeURIComponent(state.selectedPersona || 'cruella');
  const token = localStorage.getItem('authToken');
  const tokenParam = token ? '&token=' + encodeURIComponent(token) : '';
  return protocol + location.host + '/ws/camera?persona=' + persona + tokenParam;
}

export function warmupChat() {
  fetch('/api/chat/warmup', { method: 'POST' }).catch(() => {});
}

export function updateConf(val) {
  document.getElementById('conf-val').textContent = val + '%';
  if (!state.selectedPersona) return;
  fetch('/api/conf/' + val / 100 + '?persona=' + encodeURIComponent(state.selectedPersona), {
    method: 'POST',
  }).catch(() => {});
}

export function loadConfidence() {
  if (!state.selectedPersona) return;
  fetch('/api/conf?persona=' + encodeURIComponent(state.selectedPersona))
    .then((r) => r.json())
    .then((data) => {
      const pct = Math.round(data.conf_thres * 100);
      document.getElementById('conf-slider').value = pct;
      document.getElementById('conf-val').textContent = pct + '%';
    })
    .catch(() => {});
}

export function initializeSearchSession(detectedCategories, recommendations) {
  const token = localStorage.getItem('authToken');
  const headers = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = 'Bearer ' + token;

  fetch('/api/session/start', {
    method: 'POST',
    headers,
    body: JSON.stringify({
      persona: state.selectedPersona || 'cruella',
      detected_categories: detectedCategories || [],
      recommendations: recommendations || [],
    }),
  })
    .then((r) => r.json())
    .then((data) => {
      if (data.session_id) state.currentSessionId = data.session_id;
      if (token && Array.isArray(data.results) && data.results.length) {
        state.currentRecommendations = data.results;
        import('./ui/recommendations.js').then(({ renderRecs }) => renderRecs(data.results));
      }
    })
    .catch(() => {});
}

export async function triggerAgentRecommendations(detectedType, detectedBodyType, detectedColor) {
  const myGen = ++state.agentRoundGeneration;
  const agentEl = document.getElementById('agent-status');

  if (agentEl) {
    agentEl.hidden = false;
    agentEl.className = 'agent-status loading';
    agentEl.textContent = '⏳ Agents computing…';
  }

  const genderEl = document.getElementById('gender-select');
  const payload = {
    detected_type: detectedType || '',
    detected_body_type: detectedBodyType || '',
    detected_color: detectedColor || '',
    user_gender: genderEl ? genderEl.value || '' : '',
  };

  try {
    const resp = await fetch('/api/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (myGen !== state.agentRoundGeneration) return;

    if (resp.status === 503 || resp.status === 504) {
      if (agentEl) agentEl.hidden = true;
      return;
    }

    if (!resp.ok) {
      if (agentEl) {
        agentEl.className = 'agent-status error';
        agentEl.textContent = '⚠ Agent round failed';
      }
      return;
    }

    const data = await resp.json();
    if (myGen !== state.agentRoundGeneration) return;

    const { formatAgentRec, renderRecs } = await import('./ui/recommendations.js');
    const agentRecs = (data.recommendations || []).map(formatAgentRec);
    if (agentRecs.length) {
      renderRecs(agentRecs, { autoOpen: false });
      state.currentRecommendations = agentRecs;
    }

    if (agentEl) {
      agentEl.className = 'agent-status done';
      agentEl.textContent = '✓ Agent recommendations';
    }
  } catch (err) {
    if (myGen !== state.agentRoundGeneration) return;
    console.warn('Agent round error:', err);
    if (agentEl) {
      agentEl.className = 'agent-status error';
      agentEl.textContent = '⚠ Agent unavailable';
    }
  }
}
