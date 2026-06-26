import { state } from './state.js';
import { showNotification } from './notifications.js';
import { loadConfidence, warmupChat } from './api.js';
import { updateLoginButton } from './auth.js';
import { connectCamera, disconnectCamera, setSt } from './ui/camera.js';
import { switchView, requestPersonaIntro } from './ui/chat.js';
import { toggleRecommendationsModal } from './ui/recommendations.js';

// Re-export showNotification so inline HTML event handlers can reach it via window
export { showNotification };

function initHelpers() {
  // IntersectionObserver lazy-load for any data-src images
  const lazyTargets = document.querySelectorAll('img[data-src], source[data-srcset]');
  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver(
      (entries, obs) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          const el = entry.target;
          if (el.dataset.src) { el.src = el.dataset.src; el.removeAttribute('data-src'); }
          if (el.dataset.srcset) { el.srcset = el.dataset.srcset; el.removeAttribute('data-srcset'); }
          obs.unobserve(el);
        });
      },
      { rootMargin: '200px 0px' }
    );
    lazyTargets.forEach((t) => io.observe(t));
  } else {
    lazyTargets.forEach((el) => {
      if (el.dataset.src) el.src = el.dataset.src;
      if (el.dataset.srcset) el.srcset = el.dataset.srcset;
    });
  }
  if (!state.selectedPersona) return;
  warmupChat();
  connectCamera();
}

function initReveal() {
  const els = document.querySelectorAll('.reveal');
  if (!els.length) return;
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(
      (entries, obs) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) { entry.target.classList.add('visible'); obs.unobserve(entry.target); }
        });
      },
      { threshold: 0.12 }
    );
    els.forEach((el) => observer.observe(el));
  } else {
    els.forEach((el) => el.classList.add('visible'));
  }
}

function initCardTilt() {
  const cards = document.querySelectorAll('.card');
  cards.forEach((card) => {
    card.addEventListener('pointermove', (event) => {
      const rect = card.getBoundingClientRect();
      const cx = rect.left + rect.width / 2;
      const cy = rect.top + rect.height / 2;
      const dx = (event.clientX - cx) / rect.width;
      const dy = (event.clientY - cy) / rect.height;
      card.style.transform = `perspective(900px) translateZ(0) rotateX(${(-dy * 6).toFixed(2)}deg) rotateY(${(dx * 6).toFixed(2)}deg) translateY(-6px)`;
    });
    card.addEventListener('pointerleave', () => { card.style.transform = ''; });
  });
}

function resetPersonaState() {
  if (state.autoScanTimeout) { clearTimeout(state.autoScanTimeout); state.autoScanTimeout = null; }
  disconnectCamera();
  state.autoScanFrames = 0;

  const cam = document.getElementById('cam');
  const personGuide = document.getElementById('person-guide');
  const scanAutoHint = document.getElementById('scan-auto-hint');
  const scanHintText = document.getElementById('scan-hint-text');
  const tagsEl = document.getElementById('tags');
  const recsEl = document.getElementById('recs');
  const recsScrollLeftBtn = document.getElementById('recs-scroll-left');
  const recsScrollRightBtn = document.getElementById('recs-scroll-right');
  const recsModalList = document.getElementById('rec-modal-list');
  const recDetail = document.getElementById('rec-detail');
  const chatMsgs = document.getElementById('chat-messages');
  const chatInput = document.getElementById('chat-input');
  const micBtn = document.getElementById('mic-btn');

  cam.classList.remove('person-detected', 'person-ready');
  personGuide.hidden = true;
  scanAutoHint.classList.remove('person-detected', 'person-ready');
  scanHintText.textContent = 'Stand in front of the camera';
  state.currentSessionId = null;
  state.currentDetectedCategories = [];
  state.currentRecommendations = [];
  state.chatHistory = [];
  state.searchIntentMessages = [];
  state.chatConversationState = null;
  state.currentActiveFilters = {};
  state.hasShownInitialRecommendationsModal = false;
  state.recommendationDetailIndex = 0;
  state.introRequestToken += 1;
  state.hasChatIntroLoaded = false;
  state.introRequestInFlight = false;
  tagsEl.innerHTML = '<span class="notag">Nothing detected yet</span>';
  recsEl.innerHTML = '<div class="ph">Recommendations will appear here after a scan or chat refinement.</div>';
  recsScrollLeftBtn.disabled = true;
  recsScrollRightBtn.disabled = true;
  recsModalList.innerHTML = '';
  recDetail.innerHTML = '<div class="ph">Recommendation details will appear here.</div>';
  chatMsgs.innerHTML =
    '<div class="chat-welcome">' +
    '<div class="chat-welcome-icon"><i class="fas fa-message"></i></div>' +
    '<div class="chat-welcome-title">FashionSense Assistant</div>' +
    '<div class="chat-welcome-sub">Ask for different colors, styles, formality, or tell it to ignore the scan entirely.</div>' +
    '</div>';
  chatInput.value = '';
  micBtn.classList.remove('listening');
  chatInput.placeholder = 'Type a message...';
  toggleRecommendationsModal(false);
}

function selectPersona(persona) {
  const landingScreen = document.getElementById('landing-screen');
  const personaChip = document.getElementById('persona-chip');
  const personaResetBtn = document.getElementById('persona-reset-btn');

  state.selectedPersona = persona === 'edna' ? 'edna' : 'cruella';
  window.localStorage.setItem('fashionSensePersona', state.selectedPersona);
  landingScreen.classList.add('hidden');
  document.body.classList.add('app-started');
  document.body.dataset.persona = state.selectedPersona;
  personaChip.hidden = false;
  personaResetBtn.hidden = false;
  personaChip.textContent = state.selectedPersona === 'edna' ? 'Edna active' : 'Cruella active';
  resetPersonaState();
  loadConfidence();
  initHelpers();
  requestPersonaIntro();
}

function returnToPersonaSelection() {
  const landingScreen = document.getElementById('landing-screen');
  const personaChip = document.getElementById('persona-chip');
  const personaResetBtn = document.getElementById('persona-reset-btn');
  const camEl = document.getElementById('cam');
  const chatView = document.getElementById('chat-view');
  const tabCam = document.getElementById('tab-cam');
  const tabChat = document.getElementById('tab-chat');

  state.selectedPersona = null;
  window.localStorage.removeItem('fashionSensePersona');
  landingScreen.classList.remove('hidden');
  document.body.classList.remove('app-started');
  delete document.body.dataset.persona;
  personaChip.hidden = true;
  personaChip.textContent = '';
  personaResetBtn.hidden = true;
  resetPersonaState();
  state.currentView = 'camera';
  camEl.style.display = 'flex';
  chatView.classList.remove('active');
  tabCam.classList.add('active');
  tabChat.classList.remove('active');
  setSt('idle', 'Choose a stylist');
}

// ── Expose functions needed by inline HTML event handlers ───────────────────
import { handleLogin, handleRegister, toggleLoginModal, switchAuthForm } from './auth.js';
import { startScan, sendCmd } from './ui/camera.js';
import { sendChat, toggleVoice } from './ui/chat.js';
import { openRecommendationDetail, scrollRecommendations, submitItemFeedback } from './ui/recommendations.js';
import { updateConf } from './api.js';

window.selectPersona = selectPersona;
window.returnToPersonaSelection = returnToPersonaSelection;
window.switchView = switchView;
window.toggleLoginModal = toggleLoginModal;
window.switchAuthForm = switchAuthForm;
window.handleLogin = handleLogin;
window.handleRegister = handleRegister;
window.startScan = startScan;
window.sendCmd = sendCmd;
window.sendChat = sendChat;
window.toggleVoice = toggleVoice;
window.openRecommendationDetail = openRecommendationDetail;
window.scrollRecommendations = scrollRecommendations;
window.submitItemFeedback = submitItemFeedback;
window.toggleRecommendationsModal = toggleRecommendationsModal;
window.updateConf = updateConf;
window.showNotification = showNotification;

// ── Bootstrap ───────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initReveal();
  initCardTilt();
  updateLoginButton();

  const storedPersona = window.localStorage.getItem('fashionSensePersona');
  if (storedPersona === 'cruella' || storedPersona === 'edna') {
    selectPersona(storedPersona);
  }

  const recsModal = document.getElementById('recommendations-modal');
  const loginModal = document.getElementById('login-modal');
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      if (recsModal.classList.contains('open')) toggleRecommendationsModal(false);
      if (loginModal.classList.contains('open')) toggleLoginModal(false);
    }
  });
});
