/**
 * ViajeBot — Frontend Logic
 * Handles: chat API, TTS audio, mode toggle, tool badge rendering, chat history.
 */

const API_BASE = 'https://viagebot-backend.onrender.com';

// ── State ──────────────────────────────────────────────────────────────────
let responseMode = 'text';          // 'text' | 'voice'
let sessionId    = generateSessionId();
let isLoading    = false;
let mediaRecorder = null;
let audioChunks   = [];
let isListening   = false;
let activeAudio   = null;
let lastSpeechError = false;

// ── DOM References ─────────────────────────────────────────────────────────
const chatArea    = document.getElementById('chat-area');
const inputEl     = document.getElementById('user-input');
const sendBtn     = document.getElementById('send-btn');
const micBtn      = document.getElementById('mic-btn');
const btnText     = document.getElementById('btn-text');
const btnVoice    = document.getElementById('btn-voice');
const welcomeCard = document.getElementById('welcome-card');
const statusDot   = document.getElementById('status-dot');
const voiceStatus = document.getElementById('voice-status');
const themeToggle = document.getElementById('theme-toggle');

const DESTINATION_IMAGES = {
  tokio: [
    'https://images.unsplash.com/photo-1542051841857-5f90071e7989?auto=format&fit=crop&w=700&q=80',
    'https://images.unsplash.com/photo-1536098561742-ca998e48cbcc?auto=format&fit=crop&w=700&q=80',
    'https://images.unsplash.com/photo-1503899036084-c55cdd92da26?auto=format&fit=crop&w=700&q=80',
  ],
  tokyo: [
    'https://images.unsplash.com/photo-1542051841857-5f90071e7989?auto=format&fit=crop&w=700&q=80',
    'https://images.unsplash.com/photo-1536098561742-ca998e48cbcc?auto=format&fit=crop&w=700&q=80',
    'https://images.unsplash.com/photo-1503899036084-c55cdd92da26?auto=format&fit=crop&w=700&q=80',
  ],
  cartagena: [
    'https://images.unsplash.com/photo-1583531352515-8884af319dc1?auto=format&fit=crop&w=700&q=80',
    'https://images.unsplash.com/photo-1624806992066-5ffcf7ca186b?auto=format&fit=crop&w=700&q=80',
    'https://images.unsplash.com/photo-1600195077909-46e573870d99?auto=format&fit=crop&w=700&q=80',
  ],
  paris: [
    'https://images.unsplash.com/photo-1502602898657-3e91760cbb34?auto=format&fit=crop&w=700&q=80',
    'https://images.unsplash.com/photo-1499856871958-5b9627545d1a?auto=format&fit=crop&w=700&q=80',
    'https://images.unsplash.com/photo-1522093007474-d86e9bf7ba6f?auto=format&fit=crop&w=700&q=80',
  ],
  medellin: [
    'https://images.unsplash.com/photo-1581781870027-04212e231e96?auto=format&fit=crop&w=700&q=80',
    'https://images.unsplash.com/photo-1607990281513-2c110a25bd8c?auto=format&fit=crop&w=700&q=80',
    'https://images.unsplash.com/photo-1534269222346-5a896154c41d?auto=format&fit=crop&w=700&q=80',
  ],
  bogota: [
    'https://images.unsplash.com/photo-1568632234157-ce7aecd03d0d?auto=format&fit=crop&w=700&q=80',
    'https://images.unsplash.com/photo-1596120236172-231999844ade?auto=format&fit=crop&w=700&q=80',
    'https://images.unsplash.com/photo-1599708153386-62bf3f03555f?auto=format&fit=crop&w=700&q=80',
  ],
  "costa rica": [
    'https://images.unsplash.com/photo-1518182170546-07661fd94144?auto=format&fit=crop&w=700&q=80',
    'https://images.unsplash.com/photo-1508726096737-5ac7ca26345d?auto=format&fit=crop&w=700&q=80',
    'https://images.unsplash.com/photo-1526392060635-9d6019884377?auto=format&fit=crop&w=700&q=80',
  ],
  kyoto: [
    'https://images.unsplash.com/photo-1493976040374-85c8e12f0c0e?auto=format&fit=crop&w=700&q=80',
    'https://images.unsplash.com/photo-1528360983277-13d401cdc186?auto=format&fit=crop&w=700&q=80',
    'https://images.unsplash.com/photo-1545569341-9eb8b30979d9?auto=format&fit=crop&w=700&q=80',
  ],
  roma: [
    'https://images.unsplash.com/photo-1552832230-c0197dd311b5?auto=format&fit=crop&w=700&q=80',
    'https://images.unsplash.com/photo-1529260830199-42c24126f198?auto=format&fit=crop&w=700&q=80',
    'https://images.unsplash.com/photo-1529154036614-a60975f5c760?auto=format&fit=crop&w=700&q=80',
  ],
  rome: [
    'https://images.unsplash.com/photo-1552832230-c0197dd311b5?auto=format&fit=crop&w=700&q=80',
    'https://images.unsplash.com/photo-1529260830199-42c24126f198?auto=format&fit=crop&w=700&q=80',
    'https://images.unsplash.com/photo-1529154036614-a60975f5c760?auto=format&fit=crop&w=700&q=80',
  ],
  espana: [
    'https://images.unsplash.com/photo-1509840841025-9088ba78a826?auto=format&fit=crop&w=700&q=80',
    'https://images.unsplash.com/photo-1543783207-ec64e4d95325?auto=format&fit=crop&w=700&q=80',
    'https://images.unsplash.com/photo-1539037116277-4db20889f2d4?auto=format&fit=crop&w=700&q=80',
  ],
  spain: [
    'https://images.unsplash.com/photo-1509840841025-9088ba78a826?auto=format&fit=crop&w=700&q=80',
    'https://images.unsplash.com/photo-1543783207-ec64e4d95325?auto=format&fit=crop&w=700&q=80',
    'https://images.unsplash.com/photo-1539037116277-4db20889f2d4?auto=format&fit=crop&w=700&q=80',
  ],
  jordania: [
    'https://images.unsplash.com/photo-1547234935-80c7145ec969?auto=format&fit=crop&w=700&q=80',
    'https://images.unsplash.com/photo-1501233339699-2051864573af?auto=format&fit=crop&w=700&q=80',
    'https://images.unsplash.com/photo-1526481280693-3bfa7568e0f3?auto=format&fit=crop&w=700&q=80',
  ],
};

const COMMON_DESTINATIONS = [
  'spain', 'espana', 'españa', 'japan', 'japon', 'colombia', 'france', 'francia',
  'italy', 'italia', 'germany', 'alemania', 'portugal', 'morocco', 'marruecos',
  'egypt', 'egipto', 'turkey', 'turquia', 'turquía', 'greece', 'grecia',
  'iceland', 'islandia', 'norway', 'noruega', 'sweden', 'suecia', 'finland',
  'finlandia', 'india', 'vietnam', 'indonesia', 'philippines', 'filipinas',
  'australia', 'new zealand', 'nueva zelanda', 'south africa', 'sudafrica',
  'sudáfrica', 'kenya', 'tanzania', 'namibia', 'georgia', 'armenia', 'albania',
  'montenegro', 'slovenia', 'eslovenia', 'croatia', 'croacia', 'estonia',
  'latvia', 'letonia', 'lithuania', 'lituania', 'nepal', 'bhutan', 'butan',
  'bután', 'uzbekistan', 'uzbekistán', 'kazakhstan', 'kazajistan', 'kazajistán',
  'tokyo', 'tokio', 'kyoto', 'osaka', 'paris', 'london', 'londres', 'madrid',
  'barcelona', 'rome', 'roma', 'venice', 'venecia', 'cartagena', 'bogota',
  'bogotá', 'medellin', 'medellín', 'costa rica', 'dubai', 'jordania', 'jordan',
];

// ── Helpers ────────────────────────────────────────────────────────────────
function generateSessionId() {
  return 'sess_' + Math.random().toString(36).slice(2, 10);
}

function setLoading(state) {
  isLoading = state;
  sendBtn.disabled = state;
  inputEl.disabled = state;
  micBtn.disabled = state;
}

function hideWelcome() {
  if (welcomeCard && welcomeCard.parentElement) {
    welcomeCard.style.opacity = '0';
    welcomeCard.style.transform = 'translateY(-8px)';
    welcomeCard.style.transition = 'all 0.3s ease';
    setTimeout(() => welcomeCard.remove(), 300);
  }
}

function scrollToBottom() {
  chatArea.scrollTop = chatArea.scrollHeight;
}

/** Maps tool name to a human-readable label + emoji */
function toolLabel(toolName) {
  const map = {
    web_search:          '🔍 Web Search',
    currency_converter:  '💱 Currency Converter',
    travel_knowledge:    '📚 Travel Knowledge',
  };
  return map[toolName] || `🔧 ${toolName}`;
}

function normalizeText(text) {
  return text
    .toLowerCase()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '');
}

function extractDestination(text) {
  const normalized = normalizeText(text);
  const known = Object.keys(DESTINATION_IMAGES).find(place => normalized.includes(place));
  if (known) return known;
  const common = COMMON_DESTINATIONS
    .sort((a, b) => b.length - a.length)
    .find(place => normalized.includes(normalizeText(place)));
  if (common) return common;

  const patterns = [
    /(?:where\s+is|where's)\s+([a-z\s]{2,50}?)(?:\s+located|\s+situated|\s+on\s+the\s+map|\?|$)/i,
    /(?:location\s+of|map\s+of|google\s+maps\s+of|photos\s+of|images\s+of|pictures\s+of)\s+([a-z\s]{2,50})/i,
    /(?:donde esta ubicado|donde esta ubicada|donde queda ubicado|donde queda ubicada|ubicacion de|ubicacion en)\s+(?:de\s+|en\s+|a\s+)?([a-z\s]{2,50})/i,
    /(?:imagenes|fotos|mapa|maps|ubicado|ubicacion|donde queda|donde esta|visitar|viaje a|viajar a|lugares en|lugares de)\s+(?:de\s+|en\s+|a\s+)?([a-z\s]{3,40})/i,
    /(?:show|images|photos|map|location|visit|travel to|places in)\s+(?:of\s+|in\s+|to\s+)?([a-z\s]{3,40})/i,
  ];

  for (const pattern of patterns) {
    const match = normalizeText(text).match(pattern);
    if (match?.[1]) {
      return match[1]
        .replace(/\b(por favor|please|google maps|mapa|imagenes|fotos|located|situated|ubicado|ubicada|location|on the map|travel|destination)\b/g, '')
        .replace(/\s+/g, ' ')
        .trim();
    }
  }
  return null;
}

function buildVisualContext(userText, backendDestination = null) {
  const normalized = normalizeText(userText);
  const destination = backendDestination || extractDestination(userText);
  if (!destination) return null;

  // If the backend explicitly gave us a destination, we likely want to show SOMETHING.
  const wantsMap = /(mapa|maps|google maps|ubicacion|ubicado|donde queda|donde esta|location|located|where is)/i.test(normalized);
  const wantsImages = /(imagen|imagenes|foto|fotos|galeria|muestrame|mostrar|images|photos|gallery|show me|visitar|lugares|places|visit)/i.test(normalized);

  // If it's a "places to visit" or similar, show images by default
  const isGeneralQuery = /(visitar|viaje|lugares|places|visit|trip|travel)/i.test(normalized);

  return {
    destination,
    map: wantsMap,
    images: wantsImages || wantsMap || isGeneralQuery || !!backendDestination,
  };
}

function imageSetFor(destination) {
  const normalized = normalizeText(destination);
  if (DESTINATION_IMAGES[normalized]) return DESTINATION_IMAGES[normalized];
  return [
    `https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?auto=format&fit=crop&w=700&q=80`,
    `https://images.unsplash.com/photo-1469854523086-cc02fe5d8800?auto=format&fit=crop&w=700&q=80`,
    `https://images.unsplash.com/photo-1500534314209-a25ddb2bd429?auto=format&fit=crop&w=700&q=80`,
  ];
}

async function fetchDestinationImages(destination) {
  try {
    const res = await fetch(`${API_BASE}/destination-media`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ destination }),
    });
    if (!res.ok) return [];
    const data = await res.json();
    return Array.isArray(data.images) ? data.images : [];
  } catch {
    return [];
  }
}

async function hydrateImageGallery(row, destination) {
  const gallery = row.querySelector('.image-gallery');
  if (!gallery) return;
  const images = await fetchDestinationImages(destination);
  if (!images.length) return;

  gallery.innerHTML = images.slice(0, 6).map((src, index) => `
    <img src="${src}" alt="${escapeHtml(destination)} travel view ${index + 1}" loading="lazy" />
  `).join('');
}

// ── Mode Toggle ────────────────────────────────────────────────────────────
btnText.addEventListener('click', () => setMode('text'));
btnVoice.addEventListener('click', () => {
  setMode('voice');
  toggleSpeechRecognition();
});
themeToggle.addEventListener('click', toggleTheme);

function setMode(mode) {
  if (mode === 'text' && isListening) {
    recognition?.stop();
  }
  responseMode = mode;
  btnText.classList.toggle('active', mode === 'text');
  btnVoice.classList.toggle('active', mode === 'voice');
  btnText.setAttribute('aria-pressed', mode === 'text');
  btnVoice.setAttribute('aria-pressed', mode === 'voice');
  micBtn.classList.toggle('voice-mode', mode === 'voice');
}

function toggleTheme() {
  const light = document.body.classList.toggle('light-theme');
  themeToggle.setAttribute('aria-label', light ? 'Switch to dark mode' : 'Switch to light mode');
  themeToggle.querySelector('span').textContent = light ? '🌙' : '☀️';
}

function setVoiceStatus(message, tone = 'muted') {
  voiceStatus.textContent = message;
  voiceStatus.className = `voice-status ${tone}`;
}

// ── Quick Chips ────────────────────────────────────────────────────────────
document.querySelectorAll('.chip').forEach(chip => {
  chip.addEventListener('click', () => {
    const text = chip.textContent.replace(/^[\u{1F300}-\u{1FFFF}]\s*/u, '').trim();
    inputEl.value = text;
    sendMessage();
  });
});

// ── Input Auto-resize ──────────────────────────────────────────────────────
inputEl.addEventListener('input', () => {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 130) + 'px';
});

inputEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

sendBtn.addEventListener('click', sendMessage);
micBtn.addEventListener('click', toggleSpeechRecognition);

document.querySelectorAll('.destination-card').forEach(card => {
  card.addEventListener('click', () => {
    inputEl.value = card.dataset.prompt;
    sendMessage();
  });
});

function setupSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) return null;

  const speech = new SpeechRecognition();
  speech.lang = 'es-CO';
  speech.interimResults = true;
  speech.continuous = false;

  speech.addEventListener('result', event => {
    lastSpeechError = false;
    const transcript = Array.from(event.results)
      .map(result => result[0].transcript)
      .join('');
    inputEl.value = transcript;
    inputEl.dispatchEvent(new Event('input'));
    setVoiceStatus('Escuchando... puedes enviar cuando termine la frase.', 'active');
  });

  speech.addEventListener('end', () => {
    isListening = false;
    micBtn.classList.remove('listening');
    micBtn.title = 'Dictate';
    if (lastSpeechError) return;
    if (inputEl.value.trim()) {
      setVoiceStatus('Texto reconocido. Presiona enviar o Enter.', 'ok');
    } else if (responseMode === 'voice') {
      setVoiceStatus('Pulsa el microfono para dictar tu pregunta de viaje.', 'muted');
    } else {
      setVoiceStatus('');
    }
  });

  speech.addEventListener('error', event => {
    lastSpeechError = true;
    isListening = false;
    micBtn.classList.remove('listening');
    const messages = {
      'not-allowed': 'Permiso de microfono bloqueado. Activalo en el navegador y vuelve a intentar.',
      'audio-capture': 'No encontre un microfono disponible en este dispositivo.',
      'no-speech': 'No escuche voz. Pulsa el microfono e intenta hablar un poco mas cerca.',
      network: 'El reconocimiento de voz necesita conexion del navegador. Intenta de nuevo.',
    };
    setVoiceStatus(messages[event.error] || 'No pude iniciar el reconocimiento de voz en este navegador.', 'error');
  });

  return speech;
}

async function toggleSpeechRecognition() {
  setMode('voice');
  
  if (isListening) {
    stopRecording();
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    startRecording(stream);
  } catch (error) {
    isListening = false;
    micBtn.classList.remove('listening');
    setVoiceStatus(
      error?.name === 'NotAllowedError'
        ? 'Permiso de microfono bloqueado. Activalo en el navegador y vuelve a intentar.'
        : 'No pude activar el microfono. Revisa permisos o disponibilidad del dispositivo.',
      'error'
    );
  }
}

function startRecording(stream) {
  isListening = true;
  audioChunks = [];
  mediaRecorder = new MediaRecorder(stream);
  
  mediaRecorder.ondataavailable = event => {
    audioChunks.push(event.data);
  };

  mediaRecorder.onstop = async () => {
    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    setVoiceStatus('Procesando voz con Groq...', 'active');
    await sendAudioToBackend(audioBlob);
    
    // Stop all tracks to release microphone
    stream.getTracks().forEach(track => track.stop());
  };

  mediaRecorder.start();
  micBtn.classList.add('listening');
  micBtn.title = 'Stop recording';
  setVoiceStatus('Escuchando... pulsa de nuevo para transcribir.', 'active');
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }
  isListening = false;
  micBtn.classList.remove('listening');
  micBtn.title = 'Dictate';
}

async function sendAudioToBackend(blob) {
  const formData = new FormData();
  formData.append('file', blob, 'recording.wav');

  try {
    const res = await fetch(`${API_BASE}/transcribe`, {
      method: 'POST',
      body: formData
    });

    if (!res.ok) throw new Error('Transcription failed');
    
    const data = await res.json();
    if (data.text) {
      inputEl.value = data.text;
      inputEl.dispatchEvent(new Event('input'));
      setVoiceStatus('Texto reconocido con Groq.', 'ok');
      // Optional: auto-send if you want
      // sendMessage();
    }
  } catch (error) {
    setVoiceStatus('Error al transcribir audio: ' + error.message, 'error');
  }
}

// ── Render User Bubble ─────────────────────────────────────────────────────
function appendUserMessage(text) {
  const row = document.createElement('div');
  row.className = 'message-row user';
  row.innerHTML = `
    <div class="avatar user-avatar" aria-hidden="true">👤</div>
    <div class="bubble-wrapper">
      <div class="bubble user">${escapeHtml(text)}</div>
    </div>
  `;
  chatArea.appendChild(row);
  scrollToBottom();
}

// ── Typing Indicator ───────────────────────────────────────────────────────
function showTyping() {
  const row = document.createElement('div');
  row.className = 'message-row';
  row.id = 'typing-row';
  row.innerHTML = `
    <div class="avatar bot-avatar" aria-hidden="true">✈️</div>
    <div class="bubble-wrapper">
      <div class="typing-indicator" aria-label="ViajeBot is typing">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      </div>
    </div>
  `;
  chatArea.appendChild(row);
  scrollToBottom();
}

function removeTyping() {
  document.getElementById('typing-row')?.remove();
}

// ── Render Bot Bubble ──────────────────────────────────────────────────────
function appendBotMessage(text, toolUsed, toolName, audioBlob, visualContext = null, audioFailed = false) {
  const row = document.createElement('div');
  row.className = 'message-row';

  const bubbleClass = toolUsed ? 'bubble bot tool-active' : 'bubble bot';

  // Tool badge HTML (persists in history)
  const badgeHtml = toolUsed && toolName
    ? `<div class="tool-badge" aria-label="Tool used: ${toolLabel(toolName)}">
         <span class="tool-dot" aria-hidden="true"></span>
         ${toolLabel(toolName)}
       </div>`
    : '';

  // Audio player HTML
  let audioHtml = '';
  let audioUrl = null;
  if (audioBlob) {
    audioUrl = URL.createObjectURL(audioBlob);
    audioHtml = `
      <div class="audio-player-wrap">
        <button class="play-btn" aria-label="Play audio response">▶</button>
        <span class="audio-label">🔊 Voice response</span>
      </div>
    `;
  }
  if (audioFailed) {
    audioHtml = `
      <div class="audio-player-wrap audio-error">
        <span class="audio-label">Voice mode is active, but audio could not be generated. Check the backend /tts endpoint and OPENAI_API_KEY.</span>
      </div>
    `;
  }

  let visualHtml = '';
  if (visualContext?.map) {
    const query = encodeURIComponent(visualContext.destination);
    visualHtml += `
      <div class="map-card">
        <iframe
          title="Google Maps location for ${escapeHtml(visualContext.destination)}"
          src="https://www.google.com/maps?q=${query}&output=embed"
          loading="lazy"
          referrerpolicy="no-referrer-when-downgrade"></iframe>
        <div class="map-actions">
          <a href="https://www.google.com/maps/search/?api=1&query=${query}" target="_blank" rel="noopener">Open map</a>
          <a href="https://www.google.com/maps/dir/?api=1&destination=${query}" target="_blank" rel="noopener">Directions</a>
        </div>
      </div>
    `;
  }

  if (visualContext?.images) {
    const images = imageSetFor(visualContext.destination);
    visualHtml += `
      <div class="image-gallery" aria-label="Travel images for ${escapeHtml(visualContext.destination)}">
        ${images.map((src, index) => `
          <img src="${src}" alt="${escapeHtml(visualContext.destination)} travel view ${index + 1}" loading="lazy" />
        `).join('')}
      </div>
    `;
  }

  row.innerHTML = `
    <div class="avatar bot-avatar" aria-hidden="true">✈️</div>
    <div class="bubble-wrapper">
      ${badgeHtml}
      <div class="${bubbleClass}">
        ${formatMarkdown(text)}
        ${visualHtml}
        ${audioHtml}
      </div>
    </div>
  `;
  chatArea.appendChild(row);
  if (visualContext?.images) {
    hydrateImageGallery(row, visualContext.destination);
  }
  if (audioUrl) {
    const playBtn = row.querySelector('.play-btn');
    playBtn.addEventListener('click', () => playAudio(audioUrl, playBtn));
    setTimeout(() => playAudio(audioUrl, playBtn), 200);
  }
  scrollToBottom();
}

// ── Audio Play Button Callback ─────────────────────────────────────────────
function playAudio(url, btn) {
  if (activeAudio?.url === url) {
    if (activeAudio.audio.paused) {
      activeAudio.audio.play().catch(() => {});
      btn.textContent = '⏸';
    } else {
      activeAudio.audio.pause();
      btn.textContent = '▶';
    }
    return;
  }

  if (activeAudio) {
    activeAudio.audio.pause();
    activeAudio.audio.currentTime = 0;
    activeAudio.btn.textContent = '▶';
  }

  const audio = new Audio(url);
  activeAudio = { url, audio, btn };
  btn.textContent = '⏸';
  audio.play().catch(() => { btn.textContent = '▶'; });
  audio.addEventListener('ended', () => {
    btn.textContent = '▶';
    if (activeAudio?.url === url) activeAudio = null;
  });
}

// ── Simple Markdown Formatter ──────────────────────────────────────────────
function formatMarkdown(text) {
  return escapeHtml(text)
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/`(.*?)`/g, '<code style="background:rgba(255,255,255,0.08);padding:1px 5px;border-radius:4px;font-size:0.85em">$1</code>')
    .replace(/\n/g, '<br>');
}

function escapeHtml(text) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Main Send Logic ────────────────────────────────────────────────────────
async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text || isLoading) return;

  hideWelcome();
  appendUserMessage(text);

  // Reset input
  inputEl.value = '';
  inputEl.style.height = 'auto';
  setLoading(true);
  showTyping();

  try {
    // 1. Send to /chat
    const chatRes = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message:    text,
        session_id: sessionId,
        mode:       responseMode,
      }),
    });

    if (!chatRes.ok) {
      const err = await chatRes.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${chatRes.status}`);
    }

    const data = await chatRes.json();
    // data: { text, tool_used, tool_name, tools_used, mode }

    // 2. TTS if voice mode
    let audioBlob = null;
    let audioFailed = false;
    if (responseMode === 'voice') {
      try {
        const ttsRes = await fetch(`${API_BASE}/tts`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: data.text }),
        });
        if (ttsRes.ok) {
          audioBlob = await ttsRes.blob();
        } else {
          audioFailed = true;
        }
      } catch {
        // TTS failure is non-fatal — still show text
        audioFailed = true;
      }
    }

    removeTyping();
    const visualDestination = data.destination || text;
    appendBotMessage(data.text, data.tool_used, data.tool_name, audioBlob, buildVisualContext(text, data.destination), audioFailed);

  } catch (err) {
    removeTyping();
    appendBotMessage(
      `⚠️ Sorry, I encountered an error: ${err.message}\nMake sure the backend is running at ${API_BASE}.`,
      false, null, null
    );
    // Red status dot on error
    statusDot.style.background = '#ff6b6b';
    statusDot.style.boxShadow  = '0 0 0 2px rgba(255,107,107,0.3)';
  } finally {
    setLoading(false);
  }
}

// ── Health Check on Load ───────────────────────────────────────────────────
(async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(4000) });
    if (!res.ok) throw new Error();
    // Green — already default
  } catch {
    statusDot.style.background = '#ff6b6b';
    statusDot.style.boxShadow  = '0 0 0 2px rgba(255,107,107,0.3)';
    statusDot.title = 'Backend offline';
  }
})();
