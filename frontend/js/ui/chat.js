import { state, SILENCE_TIMEOUT_MS, SPEECH_START_THRESHOLD, SPEECH_CONTINUE_THRESHOLD } from '../state.js';
import { extractApiError } from '../utils.js';
import { extractIncludeFilters, renderRecs } from './recommendations.js';
import { connectCamera, disconnectCamera } from './camera.js';
import { showNotification } from '../notifications.js';

const chatMsgs = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const micBtn = document.getElementById('mic-btn');
const micSelect = document.getElementById('mic-select');

export function switchView(view) {
  state.currentView = view;
  const camEl = document.getElementById('cam');
  const chatView = document.getElementById('chat-view');
  const tabCam = document.getElementById('tab-cam');
  const tabChat = document.getElementById('tab-chat');
  if (view === 'camera') {
    camEl.style.display = 'flex';
    chatView.classList.remove('active');
    tabCam.classList.add('active');
    tabChat.classList.remove('active');
    connectCamera();
  } else {
    disconnectCamera();
    camEl.style.display = 'none';
    chatView.classList.add('active');
    tabCam.classList.remove('active');
    tabChat.classList.add('active');
    chatInput.focus();
    requestPersonaIntro();
  }
}

export function requestPersonaIntro() {
  if (!state.selectedPersona || state.hasChatIntroLoaded || state.introRequestInFlight) return;
  if (state.chatHistory.length > 0) {
    state.hasChatIntroLoaded = true;
    return;
  }
  const reply =
    state.selectedPersona === 'edna'
      ? 'I am Edna. I handle fashion, and I handle it correctly. Tell me what you need.'
      : 'Darling, I am Cruella, your fashion accomplice. Tell me what power look you want.';
  addChatBubble(reply, 'bot');
  state.chatHistory.push({ role: 'assistant', content: reply });
  state.hasChatIntroLoaded = true;
}

export function addChatBubble(text, sender) {
  const welcome = chatMsgs.querySelector('.chat-welcome');
  if (welcome) welcome.remove();
  const bubble = document.createElement('div');
  bubble.className = 'chat-bubble ' + sender;
  bubble.textContent = text;
  chatMsgs.appendChild(bubble);
  chatMsgs.scrollTop = chatMsgs.scrollHeight;
  return bubble;
}

export function addTypingIndicator() {
  const welcome = chatMsgs.querySelector('.chat-welcome');
  if (welcome) welcome.remove();
  const bubble = document.createElement('div');
  bubble.className = 'chat-bubble bot';
  bubble.id = 'typing-indicator';
  bubble.innerHTML = '<span class="typing-dots"><span></span><span></span><span></span></span>';
  chatMsgs.appendChild(bubble);
  chatMsgs.scrollTop = chatMsgs.scrollHeight;
  return bubble;
}

export function removeTypingIndicator() {
  const el = document.getElementById('typing-indicator');
  if (el) el.remove();
}

export function sendChat() {
  if (!state.selectedPersona) return;
  const text = chatInput.value.trim();
  if (!text) return;
  state.hasChatIntroLoaded = true;
  chatInput.value = '';
  addChatBubble(text, 'user');
  state.chatHistory.push({ role: 'user', content: text });
  addTypingIndicator();

  fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message: text,
      persona: state.selectedPersona,
      assistant_mode: state.selectedPersona,
      history: state.chatHistory,
      session_id: state.currentSessionId,
      state: state.chatConversationState,
      detected_categories: state.currentDetectedCategories,
      recommendations: state.currentRecommendations,
    }),
  })
    .then((response) => response.json().then((data) => ({ ok: response.ok, status: response.status, data })))
    .then(({ ok, status, data }) => {
      if (!ok) throw new Error(extractApiError(data, 'Chat failed (HTTP ' + status + ')'));
      removeTypingIndicator();
      const reply = data.reply || data.message || data.response || 'Sorry, I could not process that.';
      if (data.session_id) state.currentSessionId = data.session_id;
      if (data.state) state.chatConversationState = data.state;
      if (data.active_filters && Object.keys(extractIncludeFilters(data.active_filters)).length) {
        state.currentActiveFilters = data.active_filters;
      } else if (data.state && data.state.filters && Object.keys(extractIncludeFilters(data.state.filters)).length) {
        state.currentActiveFilters = data.state.filters;
      }
      addChatBubble(reply, 'bot');
      state.chatHistory.push({ role: 'assistant', content: reply });
      if (Array.isArray(data.results) && data.results.length) renderRecs(data.results);
    })
    .catch((error) => {
      removeTypingIndicator();
      const msg = error && error.message ? error.message : 'Connection error. Please try again.';
      addChatBubble(msg, 'bot');
      showNotification('error', 'Chat Error', msg, null, 'OK', 4000);
    });
}

export function pickSupportedAudioMimeType() {
  const candidates = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/mp4', ''];
  return candidates.find((c) => !c || MediaRecorder.isTypeSupported(c)) || '';
}

export function computeRmsLevel(analyser, buffer) {
  analyser.getByteTimeDomainData(buffer);
  let sumSquares = 0;
  for (let i = 0; i < buffer.length; i++) {
    const normalized = (buffer[i] - 128) / 128;
    sumSquares += normalized * normalized;
  }
  return Math.sqrt(sumSquares / buffer.length);
}

export function stopVoiceUI() {
  state.isListening = false;
  micBtn.classList.remove('listening');
  micBtn.title = 'Voice input';
}

async function loadMicDevices() {
  try {
    const tmp = await navigator.mediaDevices.getUserMedia({ audio: true });
    tmp.getTracks().forEach((t) => t.stop());
  } catch (_) {}
  const devices = await navigator.mediaDevices.enumerateDevices();
  const mics = devices.filter((d) => d.kind === 'audioinput');
  micSelect.innerHTML = '';
  mics.forEach((device) => {
    const opt = document.createElement('option');
    opt.value = device.deviceId;
    const label = device.label || 'Mic ' + (micSelect.length + 1);
    opt.textContent = label.length > 30 ? label.slice(0, 28) + '...' : label;
    opt.title = label;
    micSelect.appendChild(opt);
  });
  const real = mics.find((d) => !d.label.toLowerCase().includes('virtual'));
  if (real) micSelect.value = real.deviceId;
}

export async function toggleVoice() {
  if (!state.micDevicesLoaded) {
    await loadMicDevices();
    state.micDevicesLoaded = true;
  }
  if (state.isListening) {
    state.mediaRecorder.stop();
    return;
  }
  if (state.currentView !== 'chat') switchView('chat');

  const deviceId = micSelect.value;
  navigator.mediaDevices
    .getUserMedia({
      audio: {
        deviceId: deviceId ? { exact: deviceId } : undefined,
        echoCancellation: true,
        noiseSuppression: true,
        sampleRate: 16000,
      },
    })
    .then((stream) => {
      const actx = new AudioContext();
      const src = actx.createMediaStreamSource(stream);
      const analyser = actx.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.2;
      src.connect(analyser);
      const buf = new Uint8Array(analyser.fftSize);
      let speechDetected = false;
      let lastSoundAt = null;

      const levelCheck = setInterval(() => {
        const rms = computeRmsLevel(analyser, buf);
        if (!speechDetected && rms >= SPEECH_START_THRESHOLD) {
          speechDetected = true;
          lastSoundAt = Date.now();
          chatInput.placeholder = 'Listening... speech detected';
          return;
        }
        if (!speechDetected) return;
        if (rms >= SPEECH_CONTINUE_THRESHOLD) { lastSoundAt = Date.now(); return; }
        if (
          state.isListening &&
          state.mediaRecorder &&
          state.mediaRecorder.state === 'recording' &&
          lastSoundAt !== null &&
          Date.now() - lastSoundAt >= SILENCE_TIMEOUT_MS
        ) {
          chatInput.placeholder = 'Silence detected, sending...';
          state.mediaRecorder.stop();
        }
      }, 500);

      state.isListening = true;
      micBtn.classList.add('listening');
      micBtn.title = 'Listening... click to stop';
      chatInput.value = '';
      chatInput.placeholder = 'Listening... starts auto-send after speech + 5s silence';
      state.audioChunks = [];

      const mimeType = pickSupportedAudioMimeType();
      state.currentAudioMimeType = mimeType || 'audio/webm';
      const opts = mimeType ? { mimeType, audioBitsPerSecond: 128000 } : { audioBitsPerSecond: 128000 };
      state.mediaRecorder = new MediaRecorder(stream, opts);

      state.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) state.audioChunks.push(event.data);
      };

      state.mediaRecorder.onstop = () => {
        clearInterval(levelCheck);
        actx.close();
        stream.getTracks().forEach((t) => t.stop());
        stopVoiceUI();

        const blob = new Blob(state.audioChunks, { type: state.currentAudioMimeType || 'audio/webm' });
        if (blob.size < 1000) {
          chatInput.placeholder = 'Type a message...';
          addChatBubble('The recording was too short. Please hold the microphone a bit longer.', 'bot');
          return;
        }

        chatInput.placeholder = 'Transcribing...';
        const form = new FormData();
        const extension = state.currentAudioMimeType.includes('ogg')
          ? 'ogg'
          : state.currentAudioMimeType.includes('mp4')
          ? 'm4a'
          : 'webm';
        form.append('audio', blob, 'voice.' + extension);

        fetch('/api/transcribe', { method: 'POST', body: form })
          .then((r) => r.json().then((data) => ({ ok: r.ok, status: r.status, data })))
          .then(({ ok, status, data }) => {
            if (!ok) {
              const detail = data && (data.detail || data.message);
              if (status === 503) throw new Error(detail || 'Voice model is still loading. Please wait and try again.');
              throw new Error(detail || 'Transcription failed (HTTP ' + status + ')');
            }
            return data;
          })
          .then((data) => {
            chatInput.placeholder = 'Type a message...';
            const text = (data.text || data.transcript || data.transcription || data.message || '').trim();
            if (!text) { addChatBubble('Could not understand audio. Try again.', 'bot'); return; }
            chatInput.value = text;
            addChatBubble(text, 'user');
            state.chatHistory.push({ role: 'user', content: text });
            chatInput.value = '';
            addTypingIndicator();

            fetch('/api/chat', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                message: text,
                persona: state.selectedPersona,
                assistant_mode: state.selectedPersona,
                history: state.chatHistory,
                session_id: state.currentSessionId,
                state: state.chatConversationState,
                detected_categories: state.currentDetectedCategories,
                recommendations: state.currentRecommendations,
              }),
            })
              .then((r) => r.json().then((data) => ({ ok: r.ok, status: r.status, data })))
              .then(({ ok, status, data: chatData }) => {
                if (!ok) throw new Error(extractApiError(chatData, 'Chat failed (HTTP ' + status + ')'));
                removeTypingIndicator();
                const reply = chatData.reply || chatData.message || chatData.response || 'Sorry, I could not process that.';
                if (chatData.session_id) state.currentSessionId = chatData.session_id;
                if (chatData.state) state.chatConversationState = chatData.state;
                if (chatData.active_filters && Object.keys(extractIncludeFilters(chatData.active_filters)).length) {
                  state.currentActiveFilters = chatData.active_filters;
                } else if (chatData.state && chatData.state.filters && Object.keys(extractIncludeFilters(chatData.state.filters)).length) {
                  state.currentActiveFilters = chatData.state.filters;
                }
                addChatBubble(reply, 'bot');
                state.chatHistory.push({ role: 'assistant', content: reply });
                if (Array.isArray(chatData.results) && chatData.results.length) renderRecs(chatData.results);
              })
              .catch((error) => {
                removeTypingIndicator();
                addChatBubble(error && error.message ? error.message : 'Connection error.', 'bot');
              });
          })
          .catch((err) => {
            chatInput.placeholder = 'Type a message...';
            addChatBubble('Transcription failed: ' + (err.message || 'unknown error'), 'bot');
          });
      };

      state.mediaRecorder.start(250);
    })
    .catch(() => {
      addChatBubble('Microphone access denied. Please allow mic permissions.', 'bot');
    });
}
