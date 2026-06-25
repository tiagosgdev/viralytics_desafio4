import { state, AUTO_SCAN_FRAMES_REQUIRED, COLORS } from '../state.js';
import { esc, escAttr, rgbToHex, animateStagger } from '../utils.js';
import { buildWsUrl, initializeSearchSession, triggerAgentRecommendations } from '../api.js';
import { renderRecs } from './recommendations.js';
import { showNotification } from '../notifications.js';

// ── DOM references (resolved once on module load, after body has parsed) ──
const feed = document.getElementById('feed');
const idle = document.getElementById('idle');
const cam = document.getElementById('cam');
const dot = document.getElementById('dot');
const statusEl = document.getElementById('status');
const s1 = document.getElementById('s1');
const s2 = document.getElementById('s2');
const s3 = document.getElementById('s3');
const plbl = document.getElementById('plbl');
const btnStart = document.getElementById('btn-start');
const actions = document.getElementById('actions');
const tagsEl = document.getElementById('tags');
const personGuide = document.getElementById('person-guide');
const scanAutoHint = document.getElementById('scan-auto-hint');
const scanHintText = document.getElementById('scan-hint-text');
const bodySummaryEl = document.getElementById('body-summary');
const flashEl = document.getElementById('flash');
const frozenLbl = document.getElementById('frozen-label');

export function triggerFlash() {
  flashEl.classList.remove('go');
  void flashEl.offsetWidth;
  flashEl.classList.add('go');
  frozenLbl.classList.add('show');
}

export function hideFrozenLabel() {
  frozenLbl.classList.remove('show');
}

export function setSt(statusState, text) {
  dot.className = 'dot';
  if (statusState === 'live') dot.classList.add('live');
  if (statusState === 'scanning') dot.classList.add('scanning');
  if (statusState === 'error') dot.classList.add('error');
  statusEl.textContent = text;
}

export function setSteps(phase) {
  s1.className = 'step';
  s2.className = 'step';
  s3.className = 'step';
  if (phase === 'capturing') {
    s1.classList.add('active');
    plbl.innerHTML = '<em>Scanning</em> your outfit...';
  } else if (phase === 'analysing') {
    s1.classList.add('done');
    s2.classList.add('active');
    plbl.innerHTML = '<em>Analysing</em> detections...';
  } else if (phase === 'results') {
    s1.classList.add('done');
    s2.classList.add('done');
    s3.classList.add('done');
  }
}

export function resetIdle() {
  if (state.autoScanTimeout) { clearTimeout(state.autoScanTimeout); state.autoScanTimeout = null; }
  state.frozen = false;
  state.pendingStart = false;
  state.autoScanFrames = 0;
  hideFrozenLabel();
  feed.style.display = 'block';
  idle.classList.add('hidden');
  cam.classList.remove('active', 'person-detected', 'person-ready');
  personGuide.hidden = false;
  scanAutoHint.classList.remove('person-detected', 'person-ready');
  scanHintText.textContent = 'Stand in front of the camera';
  btnStart.disabled = false;
  actions.classList.remove('show');
  s1.className = 'step';
  s2.className = 'step';
  s3.className = 'step';
  plbl.textContent = 'Stand in frame to begin';
  setSt('idle', 'Ready');
}

export function beginCaptureUi() {
  if (state.autoScanTimeout) { clearTimeout(state.autoScanTimeout); state.autoScanTimeout = null; }
  feed.style.display = 'block';
  idle.classList.add('hidden');
  cam.classList.add('active');
  cam.classList.remove('person-detected', 'person-ready');
  personGuide.hidden = true;
  scanAutoHint.classList.remove('person-detected', 'person-ready');
  scanHintText.textContent = 'Stand in front of the camera';
  state.autoScanFrames = 0;
  btnStart.disabled = true;
  actions.classList.remove('show');
  setSteps('capturing');
  setSt('scanning', 'Starting scan...');
}

export function triggerAutoScan() {
  state.autoScanTimeout = null;
  if (!state.cameraReady || !state.ws || state.ws.readyState !== WebSocket.OPEN) return;
  if (actions.classList.contains('show')) return;
  const heightVal = parseInt(document.getElementById('height-input').value, 10);
  state.userHeightCm = heightVal >= 100 && heightVal <= 250 ? heightVal : null;
  state.ws.send(
    JSON.stringify({ cmd: 'start', user_height_cm: state.userHeightCm, user_gender: document.getElementById('gender-select').value || '' })
  );
  beginCaptureUi();
}

export function connectCamera() {
  if (!state.selectedPersona) return;
  if (state.ws && (state.ws.readyState === WebSocket.OPEN || state.ws.readyState === WebSocket.CONNECTING)) return;

  state.cameraSuspended = false;
  setSt('scanning', 'Connecting camera...');
  state.ws = new WebSocket(buildWsUrl());

  state.ws.onopen = () => {
    setSt('scanning', 'Camera handshake...');
  };

  state.ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === 'ready') {
      state.cameraReady = true;
      if (!state.pendingStart && !actions.classList.contains('show')) {
        setSt('idle', 'Camera ready');
      }
      if (state.pendingStart) {
        state.ws.send(
          JSON.stringify({ cmd: 'start', user_height_cm: state.userHeightCm, user_gender: document.getElementById('gender-select').value || '' })
        );
        state.pendingStart = false;
        beginCaptureUi();
      } else if (msg.pose_available === false) {
        btnStart.hidden = false;
        personGuide.hidden = true;
        scanAutoHint.style.display = 'none';
        setSt('idle', 'Camera ready');
      } else {
        btnStart.hidden = true;
        scanAutoHint.style.display = '';
        personGuide.hidden = false;
        setSt('idle', 'Stand in frame to begin');
      }
      return;
    }
    handleCameraMessage(msg);
  };

  state.ws.onerror = () => {
    state.cameraReady = false;
    if (state.cameraSuspended) return;
    setSt('error', 'Camera failed');
    showNotification('error', 'Camera Error', 'The camera connection failed. Check the server is running and reload.', null, 'OK', 5000);
    resetIdle();
  };

  state.ws.onclose = () => {
    state.cameraReady = false;
    state.ws = null;
    if (state.cameraSuspended) return;
    if (!actions.classList.contains('show')) {
      setSt('error', 'Camera offline');
      showNotification('warning', 'Camera Offline', 'The camera stream disconnected. Reload or reconnect.', null, 'OK', 4000);
      resetIdle();
    }
  };
}

export function disconnectCamera() {
  state.cameraSuspended = true;
  state.cameraReady = false;
  state.pendingStart = false;
  state.frozen = false;
  hideFrozenLabel();
  if (state.ws && (state.ws.readyState === WebSocket.OPEN || state.ws.readyState === WebSocket.CONNECTING)) {
    state.ws.close();
  } else {
    state.ws = null;
  }
  feed.removeAttribute('src');
  feed.style.display = 'none';
  idle.classList.remove('hidden');
  cam.classList.remove('active');
  btnStart.disabled = false;
  actions.classList.remove('show');
  s1.className = 'step';
  s2.className = 'step';
  s3.className = 'step';
  plbl.textContent = 'Press start to begin';
  setSt('idle', 'Camera paused');
}

function handleCameraMessage(msg) {
  if (msg.type === 'frame') {
    if (msg.phase === 'preview') {
      if (!state.frozen) {
        feed.style.display = 'block';
        feed.src = 'data:image/jpeg;base64,' + msg.frame;
        idle.classList.add('hidden');
        cam.classList.remove('active');

        if (msg.person_in_frame) {
          state.autoScanFrames++;
          cam.classList.add('person-detected');
          cam.classList.remove('person-ready');
          scanAutoHint.classList.add('person-detected');
          scanAutoHint.classList.remove('person-ready');
          const remaining = AUTO_SCAN_FRAMES_REQUIRED - state.autoScanFrames;
          if (remaining <= 0 && !state.autoScanTimeout) {
            cam.classList.remove('person-detected');
            cam.classList.add('person-ready');
            scanAutoHint.classList.remove('person-detected');
            scanAutoHint.classList.add('person-ready');
            scanHintText.textContent = 'Starting scan...';
            state.autoScanFrames = 0;
            // 500ms grace period — confirms person is still in frame before committing
            state.autoScanTimeout = setTimeout(triggerAutoScan, 500);
          } else if (remaining > 0) {
            scanHintText.textContent = 'Hold still… (' + remaining + ')';
          }
        } else {
          // Any single false frame resets the countdown entirely
          if (state.autoScanTimeout) { clearTimeout(state.autoScanTimeout); state.autoScanTimeout = null; }
          state.autoScanFrames = 0;
          cam.classList.remove('person-detected', 'person-ready');
          scanAutoHint.classList.remove('person-detected', 'person-ready');
          scanHintText.textContent = 'Stand in front of the camera';
        }
      }
      return;
    }

    if (msg.phase === 'capturing') {
      if (msg.flash) {
        feed.src = 'data:image/jpeg;base64,' + msg.frame;
        state.frozen = true;
        triggerFlash();
        setSt('scanning', 'Captured!');
        setSteps('analysing');
      } else if (!state.frozen) {
        feed.src = 'data:image/jpeg;base64,' + msg.frame;
        setSt('scanning', 'Scanning... ' + msg.countdown + 's');
        setSteps('capturing');
      }
    } else if (msg.phase === 'analysing') {
      setSt('scanning', 'Analysing...');
      setSteps('analysing');
    }
  } else if (msg.type === 'results') {
    state.frozen = false;
    hideFrozenLabel();
    state.chatConversationState = null;
    state.currentActiveFilters = {};
    if (msg.body_annotated_frame) {
      feed.style.display = 'block';
      feed.src = 'data:image/jpeg;base64,' + msg.body_annotated_frame;
    } else if (msg.annotated_frame) {
      feed.style.display = 'block';
      feed.src = 'data:image/jpeg;base64,' + msg.annotated_frame;
    }
    setSt('live', 'Done');
    setSteps('results');
    renderTags(msg.detections || []);
    renderBodyAnalysis(msg.body_analysis || null);
    renderRecs(msg.recommendations || [], { autoOpen: false });
    state.currentDetectedCategories = msg.dominant || (msg.detections || []).map((d) => d.class_name);
    state.currentRecommendations = msg.recommendations || [];
    initializeSearchSession(state.currentDetectedCategories, state.currentRecommendations);
    actions.classList.add('show');
    plbl.textContent = 'Choose an action below';
    state.currentDetectedClothingType = (msg.detections || [])[0]?.class_name || '';
    state.currentDetectedBodyType = msg.body_analysis?.body_shape || '';
    state.currentDetectedColor = msg.dominant_color?.name || (msg.detections || [])[0]?.color_name || '';
    triggerAgentRecommendations(state.currentDetectedClothingType, state.currentDetectedBodyType, state.currentDetectedColor);
  } else if (msg.type === 'error') {
    setSt('error', 'Error');
    resetIdle();
  }
}

export function startScan() {
  if (!state.selectedPersona) return;
  state.frozen = false;
  state.currentSessionId = null;
  state.currentDetectedCategories = [];
  state.currentRecommendations = [];
  state.chatHistory = [];
  state.chatConversationState = null;
  state.currentActiveFilters = {};
  state.hasShownInitialRecommendationsModal = false;
  state.recommendationDetailIndex = 0;
  state.agentRoundGeneration++;
  state.currentDetectedBodyType = '';
  state.currentDetectedClothingType = '';
  state.currentDetectedColor = '';
  const agentEl = document.getElementById('agent-status');
  if (agentEl) agentEl.hidden = true;

  import('./recommendations.js').then(({ toggleRecommendationsModal }) => toggleRecommendationsModal(false));
  hideFrozenLabel();
  state.pendingStart = true;
  if (!state.ws || state.ws.readyState === WebSocket.CLOSED) {
    connectCamera();
    return;
  }
  if (state.cameraReady && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(
      JSON.stringify({ cmd: 'start', user_height_cm: state.userHeightCm, user_gender: document.getElementById('gender-select').value || '' })
    );
    state.pendingStart = false;
    beginCaptureUi();
  } else {
    setSt('scanning', 'Preparing camera...');
  }
}

export function sendCmd(cmd) {
  const recsEl = document.getElementById('recs');
  const recsScrollLeftBtn = document.getElementById('recs-scroll-left');
  const recsScrollRightBtn = document.getElementById('recs-scroll-right');
  const recsModalList = document.getElementById('rec-modal-list');
  const recDetail = document.getElementById('rec-detail');

  if (!state.ws || state.ws.readyState !== 1) {
    startScan();
    return;
  }
  state.ws.send(
    JSON.stringify({ cmd, user_height_cm: state.userHeightCm, user_gender: document.getElementById('gender-select').value || '' })
  );
  if (cmd === 'retry') {
    state.frozen = false;
    state.currentSessionId = null;
    state.currentDetectedCategories = [];
    state.currentRecommendations = [];
    state.chatHistory = [];
    state.chatConversationState = null;
    state.currentActiveFilters = {};
    state.hasShownInitialRecommendationsModal = false;
    state.recommendationDetailIndex = 0;
    import('./recommendations.js').then(({ toggleRecommendationsModal }) => toggleRecommendationsModal(false));
    hideFrozenLabel();
    actions.classList.remove('show');
    tagsEl.innerHTML = '<span class="notag">Scanning...</span>';
    bodySummaryEl.innerHTML = '<div class="ph">Analysing pose and measurements...</div>';
    recsEl.innerHTML = '<div class="ph">Scanning your outfit...</div>';
    recsScrollLeftBtn.disabled = true;
    recsScrollRightBtn.disabled = true;
    recsModalList.innerHTML = '';
    recDetail.innerHTML = '<div class="ph">Recommendation details will appear here.</div>';
    beginCaptureUi();
  } else {
    recsEl.innerHTML = '<div class="ph">Finding new suggestions...</div>';
    setSt('live', 'Loading...');
  }
}

export function renderTags(dets) {
  if (!dets.length) {
    tagsEl.innerHTML = '<span class="notag">No clothing detected</span>';
    return;
  }
  const seen = {};
  dets.forEach((d) => {
    if (!seen[d.class_name] || seen[d.class_name].confidence < d.confidence) seen[d.class_name] = d;
  });
  tagsEl.innerHTML = Object.values(seen)
    .map((d) => {
      const detectedHex = d.color ? rgbToHex(d.color) : null;
      const paletteHex = COLORS[d.class_name] || '#888';
      const swatch = detectedHex || paletteHex;
      const lbl = d.class_name.replace(/_/g, ' ');
      const conf = Math.round(d.confidence * 100);
      const colorName = d.color_name ? String(d.color_name) : '';
      const rgbText =
        d.color && Array.isArray(d.color) ? 'RGB(' + [d.color[0], d.color[1], d.color[2]].join(',') + ')' : '';
      const titleParts = [];
      if (colorName) titleParts.push(colorName);
      if (rgbText) titleParts.push(rgbText);
      const titleAttr = titleParts.length ? ' title="' + escAttr(titleParts.join(' — ')) + '"' : '';
      return (
        '<div class="tag"' + titleAttr + '>' +
        '<div class="tdot" style="background:' + swatch + '"></div>' +
        '<span class="tlabel">' + lbl + '<small class="color-name">' + (colorName ? esc(colorName) : '') + '</small></span>' +
        '<span class="tconf">' + conf + '%</span>' +
        '</div>'
      );
    })
    .join('');
  animateStagger('#tags', '.tag');
}

export function renderBodyAnalysis(analysis) {
  if (!analysis || !analysis.landmarks_detected) {
    bodySummaryEl.innerHTML =
      '<div class="ph">No reliable body landmarks detected. Try a full-body, front-facing scan with more space around you.</div>';
    return;
  }
  const measurements = analysis.measurements || {};
  const warnings = Array.isArray(analysis.warnings) ? analysis.warnings : [];
  const poseValidation = analysis.pose_validation || {};
  const silhouetteWidths = (analysis.silhouette && analysis.silhouette.widths) || {};
  const bodyShape = String(analysis.body_shape || 'unknown').replace(/_/g, ' ');
  const confidence = Math.round(Number(analysis.confidence || 0) * 100);
  const poseScore = Math.round(Number(poseValidation.score || 0) * 100);
  const landmarkCount = Number(analysis.landmarks_detected || 0);
  const hasCm = measurements.shoulder_width_cm != null;
  const metricRow = (label, raw, cm) => {
    const display =
      hasCm && cm != null && Number(cm) > 0 ? Number(cm).toFixed(1) + ' cm' : Number(raw || 0).toFixed(2);
    return '<div class="body-metric"><span>' + label + '</span><strong>' + display + '</strong></div>';
  };
  const ratioRow = (label, value) =>
    '<div class="body-metric"><span>' + label + '</span><strong>' + Number(value || 0).toFixed(2) + '</strong></div>';

  bodySummaryEl.innerHTML =
    '<div class="body-chip-row">' +
    '<div class="body-chip"><span>Shape: </span><strong>' + esc(bodyShape) + '</strong></div>' +
    '<div class="body-chip"><span>Confidence: </span><strong>' + confidence + '%</strong></div>' +
    '<div class="body-chip"><span>Landmarks: </span><strong>' + landmarkCount + '</strong></div>' +
    '<div class="body-chip"><span>Pose: </span><strong>' + poseScore + '%</strong></div>' +
    '</div>' +
    '<div class="body-metrics-grid">' +
    metricRow('Shoulders: ', measurements.shoulder_width, measurements.shoulder_width_cm) +
    metricRow('Hips: ', measurements.hip_width, measurements.hip_width_cm) +
    metricRow('Waist: ', measurements.waist_width, measurements.waist_width_cm) +
    ratioRow('S/H Ratio: ', measurements.shoulder_hip_ratio) +
    ratioRow('Sil Chest: ', silhouetteWidths.chest_width) +
    ratioRow('Sil Thigh: ', silhouetteWidths.thigh_width) +
    '</div>' +
    (hasCm && measurements.torso_length_cm
      ? '<div class="body-chip-row" style="margin-top:6px"><div class="body-chip"><span>Torso: </span><strong>' +
        Number(measurements.torso_length_cm).toFixed(1) +
        ' cm</strong></div></div>'
      : '') +
    (warnings.length ? '<div class="body-warning">' + esc(warnings[0]) + '</div>' : '');
}
